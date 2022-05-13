import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import logging
import shutil
from time import time, sleep
import simpleaudio as sa
from scipy.interpolate import griddata
import sounddevice as sd
from scipy.io.wavfile import write
from datetime import datetime
from scipy.fft import fft, ifft, fftshift, ifftshift

try:
    import pyrealsense2 as rs
except:
    rs=None


class Sound_Echo_and_Depth_Mapper:
    def __init__(self, user_defined_parameters, mainThreadPlotter):
        self.mainThreadPlotter = mainThreadPlotter
        stereoCalibrateParams = np.load(user_defined_parameters["stereo_calibrate_param_path_echo_depth"])

        stereoRectifyParams = np.load(user_defined_parameters["stereo_rectify_param_path_echo_depth"])
        self.KL = stereoCalibrateParams["KL"]
        self.DL = stereoCalibrateParams["DL"]
        self.KR = stereoCalibrateParams["KR"]
        self.DR = stereoCalibrateParams["DR"]
        self.R = stereoCalibrateParams["R"]
        self.T = stereoCalibrateParams["T"]
        self.E = stereoCalibrateParams["E"]
        self.F = stereoCalibrateParams["F"]

        self.R1 = stereoRectifyParams["R1"]
        self.R2 = stereoRectifyParams["R2"]
        self.P1 = stereoRectifyParams["P1"]
        self.P2 = stereoRectifyParams["P2"]
        self.Q = stereoRectifyParams["Q"]
        self.roi_left = stereoRectifyParams["roi_left"]
        self.roi_right = stereoRectifyParams["roi_right"]

        self.num_samples = user_defined_parameters["sample_num_echo_depth"]
        self.time_between_shots = user_defined_parameters["time_between_shots"]
        self.output_path = user_defined_parameters["echo_depth_output_path"]
        self.frequency = user_defined_parameters["frequency"]
        self.pulse_duration = user_defined_parameters["pulse_duration"]
        self.broadcast_sample_rate = user_defined_parameters["broadcast_sample_rate"]
        self.sigma_duration_ratio = user_defined_parameters["envelope_std_ratio"]
        self.gaussian_envelope = user_defined_parameters["gaussian_envelope"]
        self.recording_sample_rate = user_defined_parameters["recording_sample_rate"]
        self.echo_duration = user_defined_parameters["echo_duration"]
        self.use_average_image = user_defined_parameters["use_average_image"]

        self.widthL=user_defined_parameters["width"]
        self.heightL=user_defined_parameters["height"]
        self.widthR=user_defined_parameters["width"]
        self.heightR=user_defined_parameters["height"]

        self.is_realsense = user_defined_parameters["is_realsense"]
        self.laser= 1 if user_defined_parameters["laser"] else 0
        print(self.laser)
        self.lower_freq_1 = user_defined_parameters["lower_freq_1"]
        self.upper_freq_1 = user_defined_parameters["upper_freq_1"]

        self.lower_freq_2 = user_defined_parameters["lower_freq_2"]
        self.upper_freq_2 = user_defined_parameters["upper_freq_2"]
        self.filter_freqs = user_defined_parameters["filter_freqs"]
        self.show_live_frames = user_defined_parameters["show_live_frames"]


        # Generate Disparity calculator and filters
        self.stereo_left_matcher = cv2.StereoSGBM_create(
            minDisparity=user_defined_parameters["min_disp"],
            numDisparities=user_defined_parameters["num_disp"],
            blockSize=user_defined_parameters["block_size"],
            uniquenessRatio=user_defined_parameters["uniquenessRatio"],
            speckleWindowSize=user_defined_parameters["speckleWindowSize"],
            speckleRange=user_defined_parameters["speckleRange"],
            disp12MaxDiff=user_defined_parameters["disp12MaxDiff"],
            P1=8 * 1 * user_defined_parameters["block_size"]**2,
            P2=32 * 1 * user_defined_parameters["block_size"]**2,
        )

        self.stereo_right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_left_matcher)

        self.disp_roi = cv2.getValidDisparityROI(self.roi_left, self.roi_right,user_defined_parameters["min_disp"],user_defined_parameters["num_disp"], user_defined_parameters["block_size"])

        # Filter
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_left_matcher)
        self.wls_filter.setLambda(user_defined_parameters["lmbda"])
        self.wls_filter.setSigmaColor(user_defined_parameters["sigma"])


    def setCameras(self, cam_L_idx, cam_R_idx):
        logging.info("1. Setting Cameras Ready...")
        if self.is_realsense:
            #%% configure
            ctx = rs.context()
            device = ctx.devices[0]
            if device is None:
                logging.error("\n[ERROR] Unable to Open the realsense Cameras!\n")
                return 1
            serial_number = device.get_info(rs.camera_info.serial_number)
            self.config = rs.config()
            self.config.enable_device(serial_number)

            #%% enable streams
            self.config.enable_stream(rs.stream.infrared, 1, self.widthL, self.heightL, rs.format.y8, 30)
            self.config.enable_stream(rs.stream.infrared, 2, self.widthR, self.heightR, rs.format.y8, 30)

            # disable laser
            stereo_module = device.query_sensors()[0]
            stereo_module.set_option(rs.option.emitter_enabled, self.laser)

        else:
            self.vidStreamL = cv2.VideoCapture(cam_L_idx)  # index of OBS - Nirie
            self.vidStreamR = cv2.VideoCapture(cam_R_idx)  # index of Droidcam camera - Izeko
            if not (self.vidStreamL.isOpened() and self.vidStreamR.isOpened()):
                logging.error("\n[ERROR] Unable to Open the Cameras!\n")
                return 1

            # Change the resolution if needed
            self.vidStreamR.set(cv2.CAP_PROP_FRAME_WIDTH, self.heightR)  # float
            self.vidStreamR.set(cv2.CAP_PROP_FRAME_HEIGHT, self.widthR)  # float

            self.vidStreamL.set(cv2.CAP_PROP_FRAME_WIDTH, self.heightL)  # float
            self.vidStreamL.set(cv2.CAP_PROP_FRAME_HEIGHT, self.widthL)  # float

        return 0

    def clean_directories_build_new(self, remove_old_data):
        # Build directory structure
        logging.info("2. Building Directory Structure...")
        if remove_old_data:
            shutil.rmtree(f"{self.output_path}/SOUND_ECHO_and_DEPTH/", ignore_errors=True)

        os.makedirs(f"{self.output_path}/SOUND_ECHO_and_DEPTH/DEPTH_MAPS/USED_IMAGES", exist_ok=True)
        os.makedirs(f"{self.output_path}/SOUND_ECHO_and_DEPTH/DEPTH_MAPS/FILTERED_DEPTH_MAPS", exist_ok=True)
        os.makedirs(f"{self.output_path}/SOUND_ECHO_and_DEPTH/SOUND_RECORDINGS/WAV", exist_ok=True)
        os.makedirs(f"{self.output_path}/SOUND_ECHO_and_DEPTH/SOUND_RECORDINGS/NPY", exist_ok=True)
        return 0

    def invert_maps_interpol(self, map_x, map_y):
        points =  np.stack([map_x.flatten(), map_y.flatten()], axis=1)
        grid = np.mgrid[:map_x.shape[0], :map_y.shape[1]]
        values = grid.reshape(2, -1).T[..., ::-1]

        grid_y, grid_x = grid
        map_back = griddata(points, values, (grid_x, grid_y), method='cubic').astype(map_x.dtype)
        return map_back[:,:,0], map_back[:,:,1]

    def take_samples(self):
        """
            If it was for the depth maps, we could capture lots of images and only afterwards
            compute their depth maps. However, as the sound signal will take time, we can take
            advantage of it, this will allow us avoid using RAM to keep all the information
            and instead do it dynamically only saving the last results. Ofcourse the last step
            will take its time, but its okay for this verison I guess...

        """
        logging.info("Generating forward and inverse rectification maps...")
        # Generate Rectification Maps
        self.mapLx, self.mapLy	=	cv2.initUndistortRectifyMap( self.KL,
                        self.DL, self.R1, self.P1, (self.widthL, self.heightL), cv2.CV_32FC1)

        self.mapRx, self.mapRy	=	cv2.initUndistortRectifyMap( self.KR,
                        self.DR, self.R2, self.P2, (self.widthR, self.heightR), cv2.CV_32FC1)

        # option one to revert rectification uses no interpolation
        #self.inv_mapLx, self.inv_mapLy = self.invert_maps_noInterpol(self.mapLx, self.mapLy)
        # option two to revert rectification uses interpolation
        #self.inv_mapLx, self.inv_mapLy = self.invert_maps_interpol(self.mapLx, self.mapLy)


        # we prepare the audio pulse as numpy array according to the user desires

        envelope_std = self.pulse_duration/self.sigma_duration_ratio
        t = np.linspace(0, self.pulse_duration,
                int(self.pulse_duration*self.broadcast_sample_rate), False)
        note = np.sin( 2*np.pi*self.frequency*t )
        # if a gaussian envelope is desired, it is applied
        if self.gaussian_envelope==True:
            def gaussian(x, mu, stdv):
                return np.exp(-0.5*(x - mu)**2 / stdv**2)/(np.sqrt(2.0*np.pi)*stdv)
            gaussian_f = gaussian(t, mu=self.pulse_duration/2.0, stdv=envelope_std)
            note = note*gaussian_f

        # Ensure that highest value is in 16-bit range
        audio = np.zeros(int((self.echo_duration+self.pulse_duration)*self.broadcast_sample_rate), dtype=np.float64)
        audio[:len(t)] = (note / np.max(np.abs(note)))*(2**15-1)
        # Convert to 16-bit data
        audio = audio.astype(np.int16)

        #t = np.linspace(0, pulse_duration+echo_recording_duration,
        #        int((pulse_duration+echo_recording_duration)*broadcast_sample_rate), False)


        def normalize_disparity_map(disparity):
            disparity = cv2.normalize(disparity, disparity, alpha=255,
                                          beta=0, norm_type=cv2.NORM_MINMAX)
            return np.uint8(disparity)

        if self.is_realsense:
            pipeline = rs.pipeline()
            profile = pipeline.start(self.config)
            for warm_up in range(20):
                frames = pipeline.wait_for_frames()

        x, y, w, h = self.disp_roi

        # sample spacing is sample_rate smaples per second-> time between samples is:
        T = 1.0 / self.recording_sample_rate

        for j in range(self.num_samples):
            logging.info(f"\n\nTAKING SAMPLE {j+1}/{self.num_samples} ########################")
            frame_n=0
            begin_t = time()

            # Start playback-recording and image capturing
            # if we wanted to do it separately:
            #broadcast = sa.play_buffer(audio, 1, 2, self.broadcast_sample_rate) # play sound
            #recording = sd.rec(int(self.recording_duration * self.recording_sample_rate),
            #   samplerate=self.recording_sample_rate, channels=1) # start recording sound
            # there is an option to do it simultaneously
            img_L=np.zeros((self.heightL,self.widthL), dtype=np.uint32)
            img_R=np.zeros((self.heightR,self.widthR), dtype=np.uint32)
            recording = sd.playrec(audio, self.recording_sample_rate, channels=1)
            if self.is_realsense:
                if self.use_average_image:
                    while((time()-begin_t)<(self.pulse_duration+self.echo_duration)): # grab all the possible images
                        frames = pipeline.wait_for_frames()
                        ir1_frame = frames.get_infrared_frame(1) # Left IR Camera, it allows 0, 1 or no input
                        ir2_frame = frames.get_infrared_frame(2) # Right IR camera
                        img_L += np.asanyarray( ir1_frame.get_data() )
                        img_R += np.asanyarray( ir2_frame.get_data() )
                        frame_n+=1
                    img_R=(img_R/frame_n).astype(np.uint8)
                    img_L=(img_L/frame_n).astype(np.uint8)
                else:
                    sleep(max(0, (self.pulse_duration+self.echo_duration)/2-(time()-begin_t)))
                    frames = pipeline.wait_for_frames()
                    ir1_frame = frames.get_infrared_frame(1) # Left IR Camera, it allows 0, 1 or no input
                    ir2_frame = frames.get_infrared_frame(2) # Right IR camera
                    img_L = np.asanyarray( ir1_frame.get_data() )
                    img_R = np.asanyarray( ir2_frame.get_data() )

            else:
                if self.use_average_image:
                    img_L=np.zeros((self.heightL,self.widthL,3), dtype=np.uint32)
                    img_R=np.zeros((self.heightR,self.widthR,3), dtype=np.uint32)
                    while((time()-begin_t)<(self.pulse_duration+self.echo_duration)): # grab all the possible images
                        if not (self.vidStreamR.grab() and self.vidStreamL.grab()):
                            logging.warning("[Error] Getting the image for this iteration. Retrying...")
                            continue
                        img_R += self.vidStreamR.retrieve()[1]
                        img_L += self.vidStreamL.retrieve()[1]
                        frame_n+=1
                    img_R=(img_R/frame_n).astype(np.uint8)
                    img_L=(img_L/frame_n).astype(np.uint8)
                else:
                    sleep(max(0, (self.pulse_duration+self.echo_duration)/2-(time()-begin_t)))
                    if not (self.vidStreamR.grab() and self.vidStreamL.grab()):
                        logging.warning("[Error] Getting the image for this iteration. Retrying...")
                        continue
                    img_R = self.vidStreamR.retrieve()[1]
                    img_L = self.vidStreamL.retrieve()[1]

            sd.wait()
            middle_t=time()
            # postprocess sound
            # erase first zeros
            difs = np.abs(recording[1:]-recording[:-1])
            valid_sound =  recording[np.argwhere(difs!=0)[0,0]:,0]
            #valid_times=t[:len(valid)]

            if self.filter_freqs:
                # number of total signal points
                N = len(valid_sound)
                samplesFourierCoefs = fft(valid_sound.astype(np.float64), N)
                # frecs = np.concatenate((np.linspace(0, 1.0/(2.0*T), N//2), np.linspace(-1.0/(2.0*T), 0, N//2))) # before the shifting for plotting
                frecs = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N) # this is after applying the shift for ploting
                samplesFourierCoefs = fftshift(samplesFourierCoefs)
                print(self.lower_freq_1, self.upper_freq_1, self.lower_freq_2, self.upper_freq_2)
                # apply filtering
                samplesFourierCoefs[ (np.abs(frecs)<=self.upper_freq_1) & (np.abs(frecs)>=self.lower_freq_1) ]=0
                samplesFourierCoefs[ (np.abs(frecs)<=self.upper_freq_2) & (np.abs(frecs)>=self.lower_freq_2) ]=0

                valid_sound = ifft(ifftshift(samplesFourierCoefs)).real


            logging.info(f"Captured {frame_n} images while broadcasting sound")
            date = datetime.now()

            # save recording as npy and as wav
            np.save(f"{self.output_path}/SOUND_ECHO_and_DEPTH/SOUND_RECORDINGS/NPY/Echo_{date}.npy", valid_sound)
            write(f"{self.output_path}/SOUND_ECHO_and_DEPTH/SOUND_RECORDINGS/WAV/Echo_{date}.wav", self.recording_sample_rate, valid_sound)  # Save as WAV file

            # save images
            resulting_images = np.concatenate((img_L, img_R), axis=1)
            cv2.imwrite(f"{self.output_path}/SOUND_ECHO_and_DEPTH/DEPTH_MAPS/USED_IMAGES/LRimage_{date}.png", resulting_images)


            # RECTIFY THE IMAGES
            rect_img_L = cv2.remap(img_L, self.mapLx, self.mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
            rect_img_R = cv2.remap(img_R, self.mapRx, self.mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

            # GRAYSCALE THE IMAGES
            if not self.is_realsense:
                grayL = cv2.cvtColor(rect_img_L, cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(rect_img_R, cv2.COLOR_BGR2GRAY)
            else:
                grayL = rect_img_L
                grayR = rect_img_R


            # COMPUTE DISPARITIES
            disparity_L = self.stereo_left_matcher.compute(grayL, grayR).astype(np.float32)/16
            disparity_R = self.stereo_right_matcher.compute(grayR, grayL).astype(np.float32)/16
            #disparity_L = np.int16(disparity_L)
            #disparity_R = np.int16(disparity_R)
            filtered_disparity = self.wls_filter.filter(disparity_L, grayL, None, disparity_R).astype(np.float32)/16  # important to put "imgL" here!!! Maybe can use the colored image here!

            # We invert the rectification to obtain back an image with the original sizes
            #filtered_disparity = cv2.remap(filtered_disparity, self.inv_mapLx, self.inv_mapLy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            # Only valid refion after rectification and disparity computation
            filtered_disparity = filtered_disparity[ y:y+h, x:x+w]
            rect_img_L = rect_img_L[ y:y+h, x:x+w]

            result = normalize_disparity_map(filtered_disparity)
            cv2.imwrite(f"{self.output_path}/SOUND_ECHO_and_DEPTH/DEPTH_MAPS/FILTERED_DEPTH_MAPS/DepthMap_{date}.png", result)

            if self.show_live_frames:
                self.mainThreadPlotter.emit(result,
                            max(100, self.time_between_shots-(time()-middle_t)*1000),
                                            f'Live Test {j}')
            else:
                sleep(max(0, self.time_between_shots-(time()-middle_t))) # wait for user movement change
        if self.is_realsense:
            pipeline.stop()

        logging.info("DONE!!!")
