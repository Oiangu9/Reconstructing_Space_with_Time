import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import logging
import shutil
from time import time, sleep
import simpleaudio as sa
from scipy.interpolate import griddata


class Depth_Mapper:
    def __init__(self, user_defined_parameters, mainThreadPlotter):
        self.mainThreadPlotter = mainThreadPlotter
        stereoCalibrateParams = np.load(user_defined_parameters["stereo_calibrate_param_path"])
        stereoRectifyParams = np.load(user_defined_parameters["stereo_rectify_param_path"])
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

        self.num_frames = user_defined_parameters["frame_number"]
        self.time_between_frames = user_defined_parameters["time_between_frames"]
        self.show_life_frames = user_defined_parameters["show_life_frames"]
        self.use_taken_photos_life = user_defined_parameters["use_taken_photos_life"]
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
            shutil.rmtree(f"./OUTPUTS/LIFE_TAKE/", ignore_errors=True)
        else: # then use old data -> expects OUTPUTS laready exists
            if not os.path.isdir("./OUTPUTS/LIFE_TAKE"):
                #logging.error("\n[ERROR] If old data is to use, there should exist an ./OUTPUT directory in working directory!")
                return 1
        os.makedirs("./OUTPUTS/LIFE_TAKE", exist_ok=True)


        for CAM in ["CAM_LEFT", "CAM_RIGHT"]:
            os.makedirs(f"./OUTPUTS/LIFE_TAKE/{CAM}", exist_ok=True)
        os.makedirs(f"./OUTPUTS/LIFE_TAKE/COMMON/", exist_ok=True)
        return 0

    def invert_maps_noInterpol(self, map_x, map_y):
        assert(map_x.shape == map_y.shape)
        rows = map_x.shape[0]
        cols = map_x.shape[1]
        m_x = np.ones(map_x.shape, dtype=map_x.dtype) * -1
        m_y = np.ones(map_y.shape, dtype=map_y.dtype) * -1
        for i in range(rows):
            for j in range(cols):
                ik = round(map_y[i, j])
                jk = round(map_x[i, j])
                if 0 <= ik < rows and 0 <= jk < cols:
                    ik=int(ik)
                    jk=int(jk)
                    m_x[ik, jk] = j
                    m_y[ik, jk] = i
        return m_x, m_y

    def invert_maps_interpol(self, map_x, map_y):
        points =  np.stack([map_x.flatten(), map_y.flatten()], axis=1)
        grid = np.mgrid[:map_x.shape[0], :map_y.shape[1]]
        values = grid.reshape(2, -1).T[..., ::-1]

        grid_y, grid_x = grid
        map_back = griddata(points, values, (grid_x, grid_y), method='cubic').astype(map_x.dtype)
        return map_back[:,:,0], map_back[:,:,1]

    def compute_Disparity_Maps(self):
        # Generate Rectification Maps
        self.mapLx, self.mapLy	=	cv2.initUndistortRectifyMap( self.KL,
                        self.DL, self.R1, self.P1, (self.widthL, self.heightL), cv2.CV_32FC1)

        self.mapRx, self.mapRy	=	cv2.initUndistortRectifyMap( self.KR,
                        self.DR, self.R2, self.P2, (self.widthR, self.heightR), cv2.CV_32FC1)

        # option one to revert rectification uses no interpolation
        #self.inv_mapLx, self.inv_mapLy = self.invert_maps_noInterpol(self.mapLx, self.mapLy)
        # option two to revert rectification uses interpolation
        self.inv_mapLx, self.inv_mapLy = self.invert_maps_interpol(self.mapLx, self.mapLy)


        # Each time a photo will be taken a sound will be emited first
        frequency = 440  # Our played note will be 440 Hz
        fs = 44100  # 44100 samples per second
        seconds = 1  # Note duration of 3 seconds
        # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
        t = np.linspace(0, seconds, seconds * fs, False)
        # Generate a 440 Hz sine wave
        note = np.sin(frequency * t * 2 * np.pi)
        # Ensure that highest value is in 16-bit range
        audio = note * (2**15 - 1) / np.max(np.abs(note))
        # Convert to 16-bit data
        audio = audio.astype(np.int16)


        def normalize_disparity_map(disparity):
            disparity = cv2.normalize(disparity, disparity, alpha=255,
                                          beta=0, norm_type=cv2.NORM_MINMAX)
            return np.uint8(disparity)

        for j in range(self.num_frames):

            # Start playback
            play_obj = sa.play_buffer(audio, 1, 2, fs)
            logging.info(f"\n\nTAKING PHOTOS {j}/{self.num_frames} ########################")
            begin_t = time()
            # instead of using .read() to get an image we decompose it into .grab and then .retrieve
            # so we can maximize the sinchronization
            if not self.use_taken_photos_life:
                if not (self.vidStreamR.grab() and self.vidStreamL.grab()):
                    logging.warning("[Error] Getting the image for this iteration. Retrying...")
                    continue
                _, img_R = self.vidStreamR.retrieve()
                _, img_L = self.vidStreamL.retrieve()
                cv2.imwrite(f"./OUTPUTS/LIFE_TAKE/CAM_LEFT/Life_L_{j}.png", img_L)
                cv2.imwrite(f"./OUTPUTS/LIFE_TAKE/CAM_RIGHT/Life_R_{j}.png", img_R)
            else:
                img_L = cv2.imread(f"./OUTPUTS/LIFE_TAKE/CAM_LEFT/Life_L_{j}.png")
                img_R = cv2.imread(f"./OUTPUTS/LIFE_TAKE/CAM_RIGHT/Life_R_{j}.png")

            # RECTIFY THE IMAGE
            rect_img_L = cv2.remap(img_L, self.mapLx, self.mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
            rect_img_R = cv2.remap(img_R, self.mapRx, self.mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
            result = np.concatenate((rect_img_L,rect_img_R), axis=1)
            cv2.imwrite(f"./OUTPUTS/LIFE_TAKE/COMMON/Life_{j}_color.png", result)

            # GRAYSCALE THE IMAGES
            grayL = cv2.cvtColor(rect_img_L, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rect_img_R, cv2.COLOR_BGR2GRAY)

            # COMPUTE DISPARITIES

            disparity_L = self.stereo_left_matcher.compute(grayL, grayR)  # .astype(np.float32)/16
            disparity_R = self.stereo_right_matcher.compute(grayR, grayL)  # .astype(np.float32)/16
            disparity_L = np.int16(disparity_L)
            disparity_R = np.int16(disparity_R)
            filtered_disparity = self.wls_filter.filter(disparity_L, grayL, None, disparity_R)  # important to put "imgL" here!!! Maybe can use the colored image here!

            total_unfiltered = np.concatenate((normalize_disparity_map(disparity_L), normalize_disparity_map(disparity_R)), axis=1)
            cv2.imwrite(f"./OUTPUTS/LIFE_TAKE/COMMON/Life_{j}_Disparity_Unfiltered.png", total_unfiltered)
            result = normalize_disparity_map(filtered_disparity)
            cv2.imwrite(f"./OUTPUTS/LIFE_TAKE/COMMON/Life_{j}_Disparity_Filtered.png", result)

            # We invert the rectification to obtain back an image with the original sizes
            filtered_disparity = cv2.remap(filtered_disparity, self.inv_mapLx, self.inv_mapLy, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            result = normalize_disparity_map(filtered_disparity)
            cv2.imwrite(f"./OUTPUTS/LIFE_TAKE/COMMON/Life_{j}_Disparity_Filtered_Unrectified.png", result)
            self.mainThreadPlotter.emit(result, 300, f'Life Test {j}')
            #sleep(max(0,time()-begin_t-self.time_between_frames)) # wait for user movement change
