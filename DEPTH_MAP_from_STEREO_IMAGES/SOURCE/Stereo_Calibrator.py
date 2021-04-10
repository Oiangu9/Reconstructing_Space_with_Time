import numpy as np
import cv2
import glob
import os
import shutil
import matplotlib.pyplot as plt
import logging


class Stereo_Calibrator:
    def __init__(self, user_defined_parameters):
        """
        User Defined Parameter Expects a dictionary with the following keys and value types

        """
        self.working_directory = user_defined_parameters["working_directory"]
        self.num_photos_calibration = user_defined_parameters["num_photos_calibration"]
        self.num_photos_test = user_defined_parameters["num_photos_test"]
        self.allow_choosing=user_defined_parameters["allow_choosing"]
        self.previs_ms=user_defined_parameters["previs_ms"]
        self.chess_size_x = user_defined_parameters["chess_size_x"]
        self.chess_size_y = user_defined_parameters["chess_size_y"]
        self.chess_square_side = user_defined_parameters["chess_square_side"]
        self.corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     user_defined_parameters["corner_criteria_its"],
                       user_defined_parameters["corner_criteria_eps"])
        self.camera_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                     user_defined_parameters["camera_criteria_its"],
                       user_defined_parameters["camera_criteria_eps"])
        self.stereocalib_flags = self.set_flags(
            user_defined_parameters['CALIB_USE_INTRINSIC_GUESS'],
            user_defined_parameters['CALIB_FIX_INTRINSIC'],
            user_defined_parameters['CALIB_FIX_PRINCIPAL_POINT'],
            user_defined_parameters['CALIB_FIX_FOCAL_LENGTH'],
            user_defined_parameters['CALIB_FIX_ASPECT_RATIO'],
            user_defined_parameters['CALIB_SAME_FOCAL_LENGTH'],
            user_defined_parameters['CALIB_ZERO_TANGENT_DIST'],
            user_defined_parameters['CALIB_FIX_K1'],
            user_defined_parameters['CALIB_FIX_K2'],
            user_defined_parameters['CALIB_FIX_K3'],
            user_defined_parameters['CALIB_FIX_K4'],
            user_defined_parameters['CALIB_FIX_K5'],
            user_defined_parameters['CALIB_FIX_K6'])
        self.flag_CALIB_ZERO_DISPARITY = user_defined_parameters['CALIB_ZERO_DISPARITY']
        self.stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                user_defined_parameters["stereocalib_criteria_its"],
                  user_defined_parameters["stereocalib_criteria_eps"])
        self.alpha = user_defined_parameters["alpha"]
        self.block_size = user_defined_parameters["block_size"]
        self.min_disp = user_defined_parameters["min_disp"]
        self.max_disp = user_defined_parameters["max_disp"]
        self.num_disp = user_defined_parameters["num_disp"]
        self.uniquenessRatio = user_defined_parameters["uniquenessRatio"]
        self.speckleWindowSize = user_defined_parameters["speckleWindowSize"]
        self.disp12MaxDiff = user_defined_parameters["disp12MaxDiff"]
        self.speckleRange = user_defined_parameters["speckleRange"]
        self.lmbda = user_defined_parameters["lmbda"]
        self.sigma = user_defined_parameters["sigma"]
        self.visual_multiplier = user_defined_parameters["visual_multiplier"]

        """
        # Execution Pipeline
        self.setWorkingDirectory()
        self.setCameras()
        self.clean_directories_build_new()
        self.take_chess_photos_compute_points()
        self.calibrate_cameras()
        self.compute_Fundamental_Matrix()
        self.draw_Epilines()
        self.compute_Stereo_Rectification()
        self.rectify_chess()
        self.compute_Disparity_Map()
        self.do_Test()
        """

    def set_flags(self, CALIB_USE_INTRINSIC_GUESS,
        CALIB_FIX_INTRINSIC,
        CALIB_FIX_PRINCIPAL_POINT,
        CALIB_FIX_FOCAL_LENGTH,
        CALIB_FIX_ASPECT_RATIO,
        CALIB_SAME_FOCAL_LENGTH,
        CALIB_ZERO_TANGENT_DIST,
        CALIB_FIX_K1,
        CALIB_FIX_K2,
        CALIB_FIX_K3,
        CALIB_FIX_K4,
        CALIB_FIX_K5,
        CALIB_FIX_K6):
        return cv2.CALIB_USE_INTRINSIC_GUESS*CALIB_USE_INTRINSIC_GUESS | \
                cv2.CALIB_FIX_INTRINSIC*CALIB_FIX_INTRINSIC | \
                cv2.CALIB_FIX_PRINCIPAL_POINT*CALIB_FIX_PRINCIPAL_POINT | \
                cv2.CALIB_FIX_FOCAL_LENGTH*CALIB_FIX_FOCAL_LENGTH | \
                cv2.CALIB_FIX_ASPECT_RATIO*CALIB_FIX_ASPECT_RATIO | \
                cv2.CALIB_SAME_FOCAL_LENGTH*CALIB_SAME_FOCAL_LENGTH | \
                cv2.CALIB_ZERO_TANGENT_DIST*CALIB_ZERO_TANGENT_DIST | \
                cv2.CALIB_FIX_K1*CALIB_FIX_K1 | \
                cv2.CALIB_FIX_K2*CALIB_FIX_K2 | \
                cv2.CALIB_FIX_K3*CALIB_FIX_K3 | \
                cv2.CALIB_FIX_K4*CALIB_FIX_K4 | \
                cv2.CALIB_FIX_K5*CALIB_FIX_K5 | \
                cv2.CALIB_FIX_K6*CALIB_FIX_K6

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

    def clean_directories_build_new(self):
        # Build directory structure
        logging.info("2. Building Directory Structure...")

        shutil.rmtree(f"./OUTPUTS", ignore_errors=True)
        os.mkdir("./OUTPUTS")

        for CAM in self.CAMS:
            os.makedirs(f"./OUTPUTS/{CAM}/BoardViews/", exist_ok=True)
            os.makedirs(f"./OUTPUTS/{CAM}/Calibrated_Parameters/", exist_ok=True)
            os.makedirs(f"./OUTPUTS/{CAM}/Undistorted_Chess_Samples/", exist_ok=True)
            os.makedirs(f"./OUTPUTS/{CAM}/Rectified_Chess_Samples/", exist_ok=True)
            os.makedirs(f"./OUTPUTS/{CAM}/BoardViews/", exist_ok=True)

        os.makedirs(f"./OUTPUTS/COMMON/BoardViews/", exist_ok=True)
        os.makedirs(f"./OUTPUTS/COMMON/Calibrated_Parameters/", exist_ok=True)
        os.makedirs(f"./OUTPUTS/COMMON/Epilines_Chess_Samples/", exist_ok=True)
        os.makedirs(f"./OUTPUTS/COMMON/Disparities_Chess_Samples/", exist_ok=True)
        os.makedirs(f"./OUTPUTS/COMMON/Life_Test/", exist_ok=True)


    def take_chess_photos_compute_points(self):
        logging.info("3. Ready to take photos and compute points... ##############################")

        # We will try to contrast with a reference subset 9x6 of the chess board image
        # We prepare the world coordinates of those points in the real world
        # We can assume z=0 and only provide x,y-s if in chess board square units:
        # (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        # if it was that the second point was 10 cm away, then (0,0,0), (10,0,0), (20,0,0) etc.
        # In our case the board squares have 2.4 cm each in width and height, so:
        world_points_chess = np.zeros((self.chess_size_y*self.chess_size_x,3),
                                                np.float32)*self.chess_square_side
        world_points_chess[:,:2] = np.mgrid[0:self.chess_size_x,0:self.chess_size_y].T.reshape(-1,2) # using chess square units

        # Arrays to store world points and image points from all the images.
        world_points = [] # 3d point in real world space
        self.image_points = {} # 2d points in image plane for each camera.
        self.image_points[self.CAMS[0]]=[]
        self.image_points[self.CAMS[1]]=[]

        successful_on_both=1

        while successful_on_both <= self.num_photos_calibration:
            # Capture at least 10 different views of the chess board - Better if 20 for example
            # Ideally the same time instance should be captured by each of the cameras
            logging.info(f"\n\nPHOTOS {successful_on_both}/{self.num_photos_calibration} ########################")
            #input("Press ENTER to take the photos:")
            # instead of using .read() to get an image we decompose it into .grab and then .retrieve
            # so we can maximize the sinchronization

            if not (self.vidStreamR.grab() and self.vidStreamL.grab()):
                logging.warning("[Error] Getting the image for this iteration. Retrying...")
                continue
            _, img_R = self.vidStreamR.retrieve()
            _, img_L = self.vidStreamL.retrieve()

            self.heightL, self.widthL, channelsL  = img_L.shape
            self.heightR, self.widthR, channelsR  = img_R.shape

            # Crop images to same size-> Not necessary
            logging.info(f"\nLeft image: {self.widthL}x{self.heightL}x{channelsL}, Right image: {self.widthR}x{self.heightR}x{channelsR}")

            # Save them now
            cv2.imwrite(f"./OUTPUTS/CAM_LEFT/BoardViews/{successful_on_both:06}.png", img_L)
            cv2.imwrite(f"./OUTPUTS/CAM_RIGHT/BoardViews/{successful_on_both:06}.png", img_R)


            found =[True, True]
            detected_corners_in_image=[0,0]
            gray=[0,0]
            for j, CAM in enumerate(self.CAMS):
                # Read the image in numpy format
                full_img = cv2.imread(f"./OUTPUTS/{CAM}/BoardViews/{successful_on_both:06}.png")

                # Grayscale the image
                gray[j] = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)

                # Find the chess board corners in the image-> Telling that it should look for a 9x6 subchess
                # The detected corners will be the pixel numbers in pixel reference (9x6, 1, 2)
                found[j], detected_corners_in_image[j] = cv2.findChessboardCorners(gray[j], (self.chess_size_x,self.chess_size_y), None)

                if found[j]==True:
                    logging.info(f"\nDetected corners {detected_corners_in_image[j].shape} on image shape {full_img.shape}")

                    # Improve/Refine detected corner pixels
                    cv2.cornerSubPix(gray[j], detected_corners_in_image[j], (11,11), (-1,-1), self.corner_criteria)


            # If found, in both cam images found will be True-> add object points, image points (after refining them)
            if found[0] == True and found[1]==True:
                # Draw and display the corners in the image to ccheck if correctly detected -> full_img is modified!
                cv2.drawChessboardCorners(img_L, (self.chess_size_x,self.chess_size_y), detected_corners_in_image[0], found[0])
                cv2.drawChessboardCorners(img_R, (self.chess_size_x,self.chess_size_y), detected_corners_in_image[1], found[1])

                joint_images = np.concatenate((img_L, img_R), axis=1)
                cv2.imshow('Processed pair of images are ok? Press SPACE if yes', joint_images)
                if self.allow_choosing:
                    ok = cv2.waitKey()
                else:
                    ok = cv2.waitKey(self.previs_ms)
                cv2.destroyAllWindows()

                if ((ok==32) or not self.allow_choosing): ################
                    logging.info(f"\nVALIDATED chess table number {successful_on_both}/{self.num_photos_calibration}!\n")
                    cv2.imwrite(f"./OUTPUTS/COMMON/BoardViews/{successful_on_both:06}.png", joint_images)

                    # Valid detection of corners, so we add them to our list for calibration
                    world_points.append(world_points_chess)

                    self.image_points[self.CAMS[0]].append(detected_corners_in_image[0])
                    self.image_points[self.CAMS[1]].append(detected_corners_in_image[1])
                    successful_on_both+=1
                else:
                    logging.info("\nTrying AGAIN!\n")
            else:
                logging.info("\nTrying AGAIN!\n")
        self.different_chess_viewsR=sorted(glob.glob("./OUTPUTS/CAM_RIGHT/BoardViews/*.png"))
        self.different_chess_viewsL=sorted(glob.glob("./OUTPUTS/CAM_LEFT/BoardViews/*.png"))



    def calibrate_cameras(self):

        self.camera_matrices={}
        self.distortion_coefficients={}

        logging.info("\n\n4. CALIBRATING CAMERAS... ##########################################")

        for CAM in self.CAMS:
            # We input the world points for the chess points together with the corresponding image points in each image
            ret, self.camera_matrices[CAM], self.distortion_coefficients[CAM], rot_vecs_views, \
                    trans_vecs_views = cv2.calibrateCamera( world_points, self.image_points[CAM],
                    gray[0].shape[::-1],  # shape tuple (nx, ny) of the images, just used to initialize the intrinsic param matrix
                    None,None, criteria=self.camera_criteria)
            # camera_matrix is the affine reference change from the projected to the pixel frame.
            # it contains focal distances fx, fy and translation from the center of projection c1,c2 to the origin of pixels

            # distoriton coefficients are the parameters to correct the fisheyeness of the pinhole camera

            # Rot vecs and trans vecs are the rotation and translation vectors to go from one image view of the
            # board to the others

            # We save the camera matrices and the distortion coefficents

            np.save(f"./OUTPUTS/{CAM}/Calibrated_Parameters/Camera_Matrix_Cam_to_Pixel_{CAM}.npy", self.camera_matrices[CAM])
            np.save(f"./OUTPUTS/{CAM}/Calibrated_Parameters/Distortion_Parameters_{CAM}.npy", self.distortion_coefficients[CAM])

            # We compute the error if we re-projected it all with the found parameters
            mean_error = 0
            for i in range(len(world_points)):
                aprox_image_Points, _ = cv2.projectPoints(world_points[i], rot_vecs_views[i], trans_vecs_views[i], self.camera_matrices[CAM], self.distortion_coefficients[CAM])
                error = cv2.norm(self.image_points[CAM][i], aprox_image_Points, cv2.NORM_L2)/len(aprox_image_Points)
                mean_error += error
            logging.info( f"\n{CAM} CALIBRATED!\nMSE for reprojection given distortion correction in {CAM}: {mean_error/len(world_points)}" )

    def compute_Fundamental_Matrix(self):

        # OBTAIN ESSENTIAL AND FUNDAMENTAL MATRICES WITH THE CALIBRATION POINTS WE GOT ########
        # Use the calbration images to obtain the Essential and Fundamental matrices
        logging.info(f"\n\n5. OBTAINING FUNDAMENTAL MATRIX ##################################\n")

        rms, self.camera_matrices["CAM_LEFT"],  self.distortion_coefficients["CAM_LEFT"], \
        self.camera_matrices["CAM_RIGHT"],  self.distortion_coefficients["CAM_RIGHT"], \
        self.R, self.T, E, self.F = cv2.stereoCalibrate(
            world_points, self.image_points['CAM_LEFT'], self.image_points['CAM_RIGHT'],
            self.camera_matrices["CAM_LEFT"], self.distortion_coefficients["CAM_LEFT"],
            self.camera_matrices["CAM_RIGHT"], self.distortion_coefficients["CAM_RIGHT"],
            (self.widthL, self.heightL), criteria = self.stereocalib_criteria, flags = self.stereocalib_flags)

        if rms > 1.0:
            logging.warning(f"[ERROR] Fundamental Matrix reproj. RMS error < 1.0 {rms}. Re-try image capture.")
        else:
            logging.info(f"[OK] Fundamental Matrix onbtention successful RMS error= {rms}")

        np.savez("./OUTPUTS/COMMON/Calibrated_Parameters/Stereo_Calibrate_rms_KL_DL_KR_DR_R_T_E_F.npz",\
                 rms=rms, KL=self.camera_matrices["CAM_LEFT"],  DL=self.distortion_coefficients["CAM_LEFT"], \
                 KR=self.camera_matrices["CAM_RIGHT"],  DR=self.distortion_coefficients["CAM_RIGHT"], \
                 R=self.R, T=self.T, E=E, F=self.F)

    def _drawlines(img1src, img2src, lines, pts1src, pts2src):
        ''' img1 - grayscale image on which we draw the epilines for the points in img2 given by pts2src
            lines - corresponding epilines '''
        r, c = img1src.shape
        img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR) # convert the grayscale images to color channels
        img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
        # Edit: use the same random seed so that two images are comparable! same colour for coresponding epilines
        np.random.seed(0)
        for r, pt1, pt2 in zip(lines, pts1src, pts2src):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
            img1color = cv2.circle(img1color, tuple(pt1[0]), 5, color, -1)
            img2color = cv2.circle(img2color, tuple(pt2[0]), 5, color, -1)
        return img1color, img2color

    def draw_Epilines(self):
        # DRAW SOME EPILINES ON THE CHESS BOARDS FOR FUN using the Fundamental Matrix ##########
        logging.info(f"\n6. DRAWING EPILINES FOR SANITY CHECK ################################\n")
        for i, (file_nameL, file_nameR) in enumerate(zip(self.different_chess_viewsL, self.widthR)):
            full_imageL = cv2.imread(file_nameL, cv2.IMREAD_GRAYSCALE)
            full_imageR = cv2.imread(file_nameR, cv2.IMREAD_GRAYSCALE)
            logging.info(i)
            # Find epilines corresponding to points in right image (second image) and
            # drawing its lines on left image
            lines_L = cv2.computeCorrespondEpilines(
                self.image_points['CAM_RIGHT'][i].reshape(-1, 1, 2), 2, self.F)
            lines_L = lines_L.reshape(-1, 3)
            img5, img6 = self._drawlines(full_imageL, full_imageR, lines_L, self.image_points['CAM_LEFT'][i], self.image_points['CAM_RIGHT'][i])

            # Find epilines corresponding to points in left image (first image) and
            # drawing its lines on right image
            lines_R = cv2.computeCorrespondEpilines(
                self.image_points['CAM_LEFT'][i].reshape(-1, 1, 2), 1, self.F)
            lines_R = lines_R.reshape(-1, 3)
            img3, img4 = self._drawlines(full_imageR, full_imageL, lines_R, self.image_points['CAM_RIGHT'][i], self.image_points['CAM_LEFT'][i])

            plt.subplot(121), plt.imshow(img5)
            plt.subplot(122), plt.imshow(img3)
            plt.suptitle("Epilines in both images")
            plt.savefig(f"./OUTPUTS/COMMON/Epilines_Chess_Samples/Epilines_{(i+1):06}.png", dpi=400)
            plt.clf()
    def compute_Stereo_Rectification(self):
        # OBTAIN STERO RECTIFICATION HOMOGRAPHIES #####################################
        logging.info(f"\n\n7. OBTAINING STERO RECTIFICATION HOMOGRAPHIES ######################\n")
        '''
        R1	Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix brings points given in the unrectified first camera's coordinate system to points in the rectified first camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system.
        R2	Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix brings points given in the unrectified second camera's coordinate system to points in the rectified second camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified second camera's coordinate system to the rectified second camera's coordinate system.
        P1	Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified first camera's image.
        P2	Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified second camera's image.
        Q	Output 4×4 disparity-to-depth mapping matrix (see reprojectImageTo3D).
        flags	Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area.
        alpha	Free scaling parameter. If it is -1 or absent, the function performs the default scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Any intermediate value yields an intermediate result between those two extreme cases.
        newImageSize	New image resolution after rectification. The same size should be passed to initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0) is passed (default), it is set to the original imageSize . Setting it to a larger value can help you preserve details in the original image, especially when there is a big radial distortion.
        validPixROI1	Optional output rectangles inside the rectified images where all the pixels are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below).
        validPixROI2	Optional output rectangles inside the rectified images where all the pixels are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below).
        '''
        self.R1, self.R2, self.P1, self.P2, self.Q, roi_left, roi_right = cv2.stereoRectify(
                self.camera_matrices["CAM_LEFT"],  self.distortion_coefficients["CAM_LEFT"], \
                self.camera_matrices["CAM_RIGHT"],  self.distortion_coefficients["CAM_RIGHT"],\
                (self.widthL, self.heightL), self.R, self.T,\
                flags=cv2.CALIB_ZERO_DISPARITY*self.flag_CALIB_ZERO_DISPARITY , alpha=0.9)

        np.savez("./OUTPUTS/COMMON/Calibrated_Parameters/Stereo_Rectify_R1_R2_P1_P2_Q_roi_left_roi_right.npz",
                     R1=self.R1, R2=self.R2, P1=self.P1, P2=self.P2, Q=self.Q, roi_left=roi_left, roi_right=roi_right)

    def rectify_chess(self):
        # DO THE UNDISTORTION BETTER WITH ALL THIS #####################################
        logging.info(f"\n8. RECTIFING IMAGES AND UNDISTORTING THEM #########################")
        # We proceed to rectify and undistort all the images, so they can be processed on the disparity map generator
        # https://docs.opencv.org/4.1.1/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
        # OJO! MAYBE YOU SHOUL DUSE GREYSCALE IMAGES IN THE RECTIFICATION!!!
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        fig.suptitle("Rectified images should have epilines aligned")

        mapLx, mapLy	=	cv2.initUndistortRectifyMap( self.camera_matrices["CAM_LEFT"],
                        self.distortion_coefficients["CAM_LEFT"], self.R1, self.P1, (self.widthL, self.heightL), cv2.CV_32FC1)

        mapRx, mapRy	=	cv2.initUndistortRectifyMap( self.camera_matrices["CAM_RIGHT"],
                        self.distortion_coefficients["CAM_RIGHT"], self.R2, self.P2, (self.widthR, self.heightR), cv2.CV_32FC1)

        for i, (file_nameL, file_nameR) in enumerate(zip(self.different_chess_viewsL, self.widthR)):
            print(i)
            img_L = cv2.imread(file_nameL)
            img_R = cv2.imread(file_nameR)

            rectified_imageL = cv2.remap(img_L, mapLx, mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
            rectified_imageR = cv2.remap(img_R, mapRx, mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

            cv2.imwrite(f'./OUTPUTS/CAM_LEFT/Rectified_Chess_Samples/Rectified_{(i+1):06}.png', rectified_imageL)
            cv2.imwrite(f'./OUTPUTS/CAM_RIGHT/Rectified_Chess_Samples/Rectified_{(i+1):06}.png', rectified_imageR)

            # For sanity check we will plot the rectified images with some horizontal lines which should match the epilines of both images
            rectified_imageL=cv2.cvtColor(rectified_imageL, cv2.COLOR_BGR2GRAY)
            rectified_imageR=cv2.cvtColor(rectified_imageR, cv2.COLOR_BGR2GRAY)
            axes[0].imshow(rectified_imageL, cmap='gray')
            axes[1].imshow(rectified_imageR, cmap='gray')

            for j in range(5,475, 20):
                axes[0].axhline(j)
                axes[1].axhline(j)

            fig.savefig(f"./OUTPUTS/COMMON/Epilines_Chess_Samples/Rectified_Epilines_{(i+1):06}.png", dpi=400)

            axes[0].clear()
            axes[1].clear()

    def compute_Disparity_Map(self):

        # OBTAIN THE DISPARITY MAP AND THE DEPTH MAP! ###################################
        logging.info(f"\n\n9. OBTAINING STERO DISPARITY AND DEPTH MAPS FOR CHESS VIEWS!!#####\n")

        # StereoSGBM Parameter explanations:
        # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size = 11
        min_disp = -1
        max_disp = 63
        # Maximum disparity minus minimum disparity. The value is always greater than zero.
        # In the current implementation, this parameter must be divisible by 16.
        num_disp = max_disp - min_disp
        # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
        # Normally, a value within the 5-15 range is good enough
        uniquenessRatio = 10
        # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
        # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        speckleWindowSize = 200
        # Maximum disparity variation within each connected component.
        # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
        # Normally, 1 or 2 is good enough.
        speckleRange = 2
        disp12MaxDiff = 0

        self.stereo_left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            disp12MaxDiff=disp12MaxDiff,
            P1=8 * 1 * block_size * block_size,
            P2=32 * 1 * block_size * block_size,
        )

        # We will not simply use the stereoSGBM but also a  disparity map post-filtering
        # in order to refine the homogeneous texture regions or some occluded or discontinuous regions of depth
        # For this, instead of only computing the disparity map for the left camera, we will
        # also compute it for the right image using the same method. then a filter will
        # use both to generate a less sparse map. Following:
        # https://docs.opencv.org/master/d3/d14/tutorial_ximgproc_disparity_filtering.html

        self.stereo_right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_left_matcher)
        # FILTER Parameters
        lmbda = 80000
        sigma = 1.3
        visual_multiplier = 6

        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_left_matcher)
        self.wls_filter.setLambda(lmbda)
        self.wls_filter.setSigmaColor(sigma)


        different_chess_views_rectified_R=sorted(glob.glob("./OUTPUTS/CAM_RIGHT/Rectified_Chess_Samples/*.png"))
        different_chess_views_rectified_L=sorted(glob.glob("./OUTPUTS/CAM_LEFT/Rectified_Chess_Samples/*.png"))
        def normalize_and_plot_disparity_map(disparity, tag, i):
            disparity_plot = cv2.normalize(disparity, disparity, alpha=255,
                                          beta=0, norm_type=cv2.NORM_MINMAX)
            disparity_plot = np.uint8(disparity_plot)
            cv2.imwrite(f"./OUTPUTS/COMMON/Disparities_Chess_Samples/Normalized_{tag}_disparity_{(i+1):06}.png", disparity_plot)

        for i, (file_nameL, file_nameR) in enumerate(zip(different_chess_views_rectified_L, different_chess_views_rectified_R)):
            rectified_imageL = cv2.imread(file_nameL)
            rectified_imageR = cv2.imread(file_nameR)
            grayL = cv2.cvtColor(rectified_imageL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(rectified_imageR, cv2.COLOR_BGR2GRAY)

            # Compute DISPARITY MAPS from L and R
            disparity_L = self.stereo_left_matcher.compute(grayL, grayR)  # .astype(np.float32)/16
            disparity_R = self.stereo_right_matcher.compute(grayR, grayL)  # .astype(np.float32)/16
            disparity_L = np.int16(disparity_L)
            disparity_R = np.int16(disparity_R)
            # Use both to generate a filtered disparity map for the left image
            filtered_disparity = self.wls_filter.filter(disparity_L, grayL, None, disparity_R)  # important to put "imgL" here!!! Maybe can use the colored image here!


            # Normalize the values to a range from 0..255 for a grayscale image of the disparity map
            normalize_and_plot_disparity_map(disparity_L, "L", i)
            normalize_and_plot_disparity_map(disparity_R, "R", i)
            normalize_and_plot_disparity_map(filtered_disparity, "Filtered", i)

            # Obtain the DEPTH MAP
            # According to the documentation we should do this division and conversion
            filtered_disparity = (filtered_disparity/16.0).astype(np.float32)

            # And voila, we get an array [h,w,3] with the 3D coordinates (in the units of the chess world points we inputed)
            # of each pixel in the image of the Left! Thus we chose the Left camera to be better
            image_3D = cv2.reprojectImageTo3D(filtered_disparity, self.Q, handleMissingValues=True)
            # handleMissingValues	Indicates, whether the function should handle missing values (i.e. points where the disparity was not computed). If handleMissingValues=true, then pixels with the minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed to 3D points with a very large Z value (currently set to 10000)
            #print(image_3D.shape)
            '''
            # plot the 3D map with depth and maybe a point cloud
            (r, c, b) = np.shape(rectified_imageL)
            # X and Y coordinates of points in the image, spaced by 10.
            (X, Y) = np.meshgrid(range(0, c, 10), range(0, r, 10))
            # Display the image
            plt.imshow(rectified_imageL)
            # Plot points from the image.
            plt.scatter(X, Y, image[Y,X])
            plt.show()
            '''
        # https://stackoverflow.com/questions/58150354/image-processing-bad-quality-of-disparity-image-with-opencv
        # Kriston erreferentzixe!
        # https://www.programcreek.com/python/?code=aliyasineser%2FstereoDepth%2FstereoDepth-master%2Fstereo_depth.py#

        def do_Test(self):
            logging.info(f"\n\n10. Taking photos for life test...#############")
            # NOW WE WILL RECORD A TEST FILM TO CHECK THE CORRECTNESS OF THE MAPS

            def normalize_disparity_map(disparity):
                disparity = cv2.normalize(disparity, disparity, alpha=255,
                                              beta=0, norm_type=cv2.NORM_MINMAX)
                return np.uint8(disparity)

            for j in range(self.num_photos_test):
                logging.info(f"\n\nTAKING PHOTOS {j}/{self.num_photos_test} ########################")
                # instead of using .read() to get an image we decompose it into .grab and then .retrieve
                # so we can maximize the sinchronization

                if not (self.vidStreamR.grab() and self.vidStreamL.grab()):
                    logging.warning("[Error] Getting the image for this iteration. Retrying...")
                    continue
                _, img_R = self.vidStreamR.retrieve()
                _, img_L = self.vidStreamL.retrieve()


                # RECTIFY THE IMAGE
                rect_img_L = cv2.remap(img_L, mapLx, mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
                rect_img_R = cv2.remap(img_R, mapRx, mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
                result = np.concatenate((rect_img_L,rect_img_R), axis=1)
                cv2.imwrite(f"./OUTPUTS/COMMON/Life_Test/Life_{j}_color.png", result)

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
                total_filtered = np.concatenate( (normalize_disparity_map(filtered_disparity), np.zeros(filtered_disparity.shape)), axis=1 )
                joint_images = np.concatenate((total_unfiltered, total_filtered), axis=0)
                cv2.imwrite(f"./OUTPUTS/COMMON/Life_Test/Life_{j}.png", joint_images)
                cv2.imshow(f'Life Test {j}', result)
                ok = cv2.waitKey(2000) #########
                cv2.destroyAllWindows()
