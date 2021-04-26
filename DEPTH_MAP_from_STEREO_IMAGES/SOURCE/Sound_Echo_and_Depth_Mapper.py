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
        self.envelope_std = user_defined_parameters["envelope_std"]
        self.gaussian_envelope = user_defined_parameters["gaussian_envelope"]
        self.recording_sample_rate = user_defined_parameters["recording_sample_rate"]
        self.recording_duration = user_defined_parameters["recording_duration"]
        self.use_average_image = user_defined_parameters["use_average_image"]
        self.delay_broadcast_recording = user_defined_parameters["delay_broadcast_recording"]

        self.widthL=640
        self.heightL=480
        self.widthR=640
        self.heightR=480


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

        # Filter
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_left_matcher)
        self.wls_filter.setLambda(user_defined_parameters["lmbda"])
        self.wls_filter.setSigmaColor(user_defined_parameters["sigma"])


    def setCameras(self, cam_L_idx, cam_R_idx):
        logging.info("1. Setting Cameras Ready...")
        # The left camera image will be the final deoth map so better the good resolution one

        self.vidStreamL = cv2.VideoCapture(cam_L_idx)  # index of OBS - Nirie
        self.vidStreamR = cv2.VideoCapture(cam_R_idx)  # index of Droidcam camera - Izeko
        if not (self.vidStreamL.isOpened() and self.vidStreamR.isOpened()):
            logging.error("\n[ERROR] Unable to Open the Cameras!\n")
            return 1

        # Change the resolution if needed
        self.vidStreamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
        self.vidStreamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

        self.vidStreamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
        self.vidStreamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

        self.CAMS=["CAM_LEFT", "CAM_RIGHT"]
        return 0

    def clean_directories_build_new(self, remove_old_data):
        # Build directory structure
        logging.info("2. Building Directory Structure...")
        if remove_old_data:
            shutil.rmtree(f"{self.output_path}/SOUND_ECHO_and_DEPTH/", ignore_errors=True)

        os.makedirs(f"{self.output_path}/SOUND_ECHO_and_DEPTH/DEPTH_MAPS/USED_IMAGES", exist_ok=True)
        os.makedirs(f"{self.output_path}/SOUND_ECHO_and_DEPTH/DEPTH_MAPS/FILTERED_DEPTH_MAPS", exist_ok=True)
        os.makedirs(f"{self.output_path}/SOUND_ECHO_and_DEPTH/SOUND_RECORDINGS", exist_ok=True)
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
        self.inv_mapLx, self.inv_mapLy = self.invert_maps_interpol(self.mapLx, self.mapLy)


        # we prepare the audio pulse as numpy array according to the user desires
        t = np.linspace(0, self.pulse_duration,
                self.pulse_duration*self.broadcast_sample_rate, False)
        note = np.sin( 2*np.pi*self.frequency*t )
        # if a gaussian envelope is desired, it is applied
        if self.gaussian_envelope==True:
            def gaussian(x, mu, stdv):
                return np.exp(-0.5*(x - mu)**2 / stdv**2)/(np.sqrt(2.0*np.pi)*stdv)
            gaussian_f = gaussian(t, mu=self.pulse_duration/2.0, stdv=self.envelope_std)
            note = note*gaussian_f

        # Ensure that highest value is in 16-bit range
        audio = note * (2**15 - 1) / np.max(np.abs(note))
        # Convert to 16-bit data
        audio = audio.astype(np.int16)


        def normalize_disparity_map(disparity):
            disparity = cv2.normalize(disparity, disparity, alpha=255,
                                          beta=0, norm_type=cv2.NORM_MINMAX)
            return np.uint8(disparity)

        for j in range(self.num_samples):
            logging.info(f"\n\nTAKING SAMPLE {j+1}/{self.num_samples} ########################")
            frames=0
            begin_t = time()

            # Start playback-recording and image capturing
            # if we wanted to do it separately:
            #broadcast = sa.play_buffer(audio, 1, 2, self.broadcast_sample_rate) # play sound
            #recording = sd.rec(int(self.recording_duration * self.recording_sample_rate),
            #   samplerate=self.recording_sample_rate, channels=1) # start recording sound
            # there is an option to do it simultaneously
            img_L=np.zeros((self.heightL,self.widthL,3))
            img_R=np.zeros((self.heightR,self.widthR,3))

            recording = sd.playrec(audio, self.recording_sample_rate, channels=1)
            if self.use_average_image:
                while((time()-begin_t)<self.pulse_duration): # grab all the possible images
                    if not (self.vidStreamR.grab() and self.vidStreamL.grab()):
                        logging.warning("[Error] Getting the image for this iteration. Retrying...")
                        continue
                    img_R += self.vidStreamR.retrieve()[1]
                    img_L += self.vidStreamL.retrieve()[1]
                    frames+=1
                img_R=(img_R/frames).astype(np.uint8)
                img_L=(img_L/frames).astype(np.uint8)
            else:
                sleep(max(0, self.pulse_duration/2-(time()-begin_t)))
                if not (self.vidStreamR.grab() and self.vidStreamL.grab()):
                    logging.warning("[Error] Getting the image for this iteration. Retrying...")
                    continue
                img_R = self.vidStreamR.retrieve()[1]
                img_L = self.vidStreamL.retrieve()[1]

            sd.wait()
            middle_t=time()
            logging.info(f"Captured {frames} images while broadcasting sound")
            date = datetime.now()

            # save recording
            write(f"{self.output_path}/SOUND_ECHO_and_DEPTH/SOUND_RECORDINGS/Echo_{date}.wav", self.recording_sample_rate, recording)  # Save as WAV file

            # save images
            resulting_images = np.concatenate((img_L, img_R), axis=1)
            cv2.imwrite(f"{self.output_path}/SOUND_ECHO_and_DEPTH/DEPTH_MAPS/USED_IMAGES/LRimage_{date}.png", resulting_images)


            # RECTIFY THE IMAGES
            rect_img_L = cv2.remap(img_L, self.mapLx, self.mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
            rect_img_R = cv2.remap(img_R, self.mapRx, self.mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

            # GRAYSCALE THE IMAGES
            grayL = cv2.cvtColor(rect_img_L, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rect_img_R, cv2.COLOR_BGR2GRAY)

            # COMPUTE DISPARITIES
            disparity_L = self.stereo_left_matcher.compute(grayL, grayR)  # .astype(np.float32)/16
            disparity_R = self.stereo_right_matcher.compute(grayR, grayL)  # .astype(np.float32)/16
            disparity_L = np.int16(disparity_L)
            disparity_R = np.int16(disparity_R)
            filtered_disparity = self.wls_filter.filter(disparity_L, grayL, None, disparity_R)  # important to put "imgL" here!!! Maybe can use the colored image here!

            # We invert the rectification to obtain back an image with the original sizes
            filtered_disparity = cv2.remap(filtered_disparity, self.inv_mapLx, self.inv_mapLy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            result = normalize_disparity_map(filtered_disparity)
            cv2.imwrite(f"{self.output_path}/SOUND_ECHO_and_DEPTH/DEPTH_MAPS/FILTERED_DEPTH_MAPS/DepthMap_{date}.png", result)


            sleep(max(0, self.time_between_shots-(time()-middle_t))) # wait for user movement change

        logging.info("DONE!!!")
