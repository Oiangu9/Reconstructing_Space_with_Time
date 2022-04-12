import numpy as np
import cv2
import glob
import os
import shutil
import matplotlib.pyplot as plt
import logging
try:
    import pyrealsense2 as rs
except:
    rs=None


class Stereo_Calibrator:
    def __init__(self, user_defined_parameters, mainThreadPlotter):
        """
        User Defined Parameter Expects a dictionary with the following keys and value types

        """
        self.mainThreadPlotter = mainThreadPlotter
        self.working_directory = user_defined_parameters["working_directory"]
        self.num_photos_calibration = user_defined_parameters["num_photos_calibration"]
        self.num_photos_test = user_defined_parameters["num_photos_test"]
        self.allow_choosing=user_defined_parameters["allow_choosing"]
        self.previs_ms=user_defined_parameters["previs_ms"]
        self.chess_size_x = user_defined_parameters["chess_size_x"]
        self.chess_size_y = user_defined_parameters["chess_size_y"]
        self.chess_square_side = user_defined_parameters["chess_square_side"]
        self.corner_criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS,
                     user_defined_parameters["corner_criteria_its"],
                       user_defined_parameters["corner_criteria_eps"])
        self.camera_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
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
        self.num_disp = user_defined_parameters["num_disp"]
        self.uniquenessRatio = user_defined_parameters["uniquenessRatio"]
        self.speckleWindowSize = user_defined_parameters["speckleWindowSize"]
        self.disp12MaxDiff = user_defined_parameters["disp12MaxDiff"]
        self.speckleRange = user_defined_parameters["speckleRange"]
        self.lmbda = user_defined_parameters["lmbda"]
        self.sigma = user_defined_parameters["sigma"]
        self.visual_multiplier = user_defined_parameters["visual_multiplier"]

        self.CAMS=["CAM_LEFT", "CAM_RIGHT"]
        self.is_realsense = user_defined_parameters["is_realsense"]
        self.h = user_defined_parameters["height"]
        self.w = user_defined_parameters["width"]

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
        # The left camera image will be the final depth map so better the good resolution one
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
            self.config.enable_stream(rs.stream.infrared, 1, self.w, self.h, rs.format.y8, 30)
            self.config.enable_stream(rs.stream.infrared, 2, self.w, self.h, rs.format.y8, 30)

            # disable laser
            stereo_module = device.query_sensors()[0]
            stereo_module.set_option(rs.option.emitter_enabled, 0)

        else:
            self.vidStreamL = cv2.VideoCapture(cam_L_idx)  # index of OBS - Nirie
            self.vidStreamR = cv2.VideoCapture(cam_R_idx)  # index of Droidcam camera - Izeko
            if not (self.vidStreamL.isOpened() and self.vidStreamR.isOpened()):
                logging.error("\n[ERROR] Unable to Open the Cameras!\n")
                return 1

            # Change the resolution if needed
            self.vidStreamR.set(cv2.CAP_PROP_FRAME_WIDTH, self.h)  # float
            self.vidStreamR.set(cv2.CAP_PROP_FRAME_HEIGHT, self.w)  # float

            self.vidStreamL.set(cv2.CAP_PROP_FRAME_WIDTH, self.h)  # float
            self.vidStreamL.set(cv2.CAP_PROP_FRAME_HEIGHT, self.w)  # float

        return 0

    def clean_directories_build_new(self, remove_old_data):
        # Build directory structure
        logging.info("2. Building Directory Structure...")
        if remove_old_data:
            shutil.rmtree(f"./OUTPUTS", ignore_errors=True)
        else: # then use old data -> expects OUTPUTS laready exists
            if not os.path.isdir("./OUTPUTS"):
                #logging.error("\n[ERROR] If old data is to use, there should exist an ./OUTPUT directory in working directory!")
                return 1
        os.makedirs("./OUTPUTS", exist_ok=True)


        for CAM in self.CAMS:
            os.makedirs(f"./OUTPUTS/CALIBRATION/{CAM}/BoardViews/", exist_ok=True)
            os.makedirs(f"./OUTPUTS/CALIBRATION/{CAM}/Calibrated_Parameters/", exist_ok=True)
            os.makedirs(f"./OUTPUTS/CALIBRATION/{CAM}/Undistorted_Chess_Samples/", exist_ok=True)
            os.makedirs(f"./OUTPUTS/CALIBRATION/{CAM}/Rectified_Chess_Samples/", exist_ok=True)
            os.makedirs(f"./OUTPUTS/CALIBRATION/{CAM}/BoardViews/", exist_ok=True)

        os.makedirs(f"./OUTPUTS/CALIBRATION/COMMON/BoardViews/", exist_ok=True)
        os.makedirs(f"./OUTPUTS/CALIBRATION/COMMON/Calibrated_Parameters/", exist_ok=True)
        os.makedirs(f"./OUTPUTS/CALIBRATION/COMMON/Epilines_Chess_Samples/", exist_ok=True)
        os.makedirs(f"./OUTPUTS/CALIBRATION/COMMON/Disparities_Chess_Samples/", exist_ok=True)
        os.makedirs(f"./OUTPUTS/CALIBRATION/COMMON/Life_Test/", exist_ok=True)
        return 0


    def take_chess_photos_compute_points(self):
        logging.info("3. Ready to take photos and compute points... ##############################")

        # We will try to contrast with a reference subset 9x6 of the chess board image
        # We prepare the world coordinates of those points in the real world
        # We can assume z=0 and only provide x,y-s if in chess board square units:
        # (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        # This way we are asssuming that the chess plane is the z=0 plane, and we will consider
        # each angle+position of the camera relative to it, as if the board was still and it was
        # the camera that moved
        # if it was that the second point was 10 cm away, then (0,0,0), (10,0,0), (20,0,0) etc.
        # In our case the board squares have 2.4 cm each in width and height, so:
        self.world_points_chess = np.zeros((self.chess_size_y*self.chess_size_x,3),
                                                np.float32)
        self.world_points_chess[:,:2] = np.mgrid[0:self.chess_size_x,0:self.chess_size_y].T.reshape(-1,2)*self.chess_square_side # using chess square side units

        # Arrays to store world points and image points from all the images.
        self.world_points = [] # 3d point in real world space with reference/origin on chess board
        self.image_points = {} # 2d points in image plane for each camera.
        self.image_points[self.CAMS[0]]=[]
        self.image_points[self.CAMS[1]]=[]

        successful_on_both=1
        if self.is_realsense:
            pipeline = rs.pipeline()
            profile = pipeline.start(self.config)
            for warm_up in range(20):
                frames = pipeline.wait_for_frames()

        while successful_on_both <= self.num_photos_calibration:
            # Capture at least 10 different views of the chess board - Better if 20 for example
            # Ideally the same time instance should be captured by each of the cameras
            logging.info(f"\n\nPHOTOS {successful_on_both}/{self.num_photos_calibration} ########################")
            #input("Press ENTER to take the photos:")
            # instead of using .read() to get an image we decompose it into .grab and then .retrieve
            # so we can maximize the sinchronization
            if self.is_realsense:

                frames = pipeline.wait_for_frames()
                ir1_frame = frames.get_infrared_frame(1) # Left IR Camera, it allows 0, 1 or no input
                ir2_frame = frames.get_infrared_frame(2) # Right IR camera
                img_L = np.asanyarray( ir1_frame.get_data() )
                img_R = np.asanyarray( ir2_frame.get_data() )
                self.heightL, self.widthL  = img_L.shape
                self.heightR, self.widthR  = img_R.shape
                channelsL,channelsR=1,1

            else:
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
            cv2.imwrite(f"./OUTPUTS/CALIBRATION/CAM_LEFT/BoardViews/{successful_on_both:06}.png", img_L)
            cv2.imwrite(f"./OUTPUTS/CALIBRATION/CAM_RIGHT/BoardViews/{successful_on_both:06}.png", img_R)


            found =[True, True]
            detected_corners_in_image=[0,0]
            gray=[0,0]
            for j, CAM in enumerate(self.CAMS):
                # Read the image in numpy format
                full_img = cv2.imread(f"./OUTPUTS/CALIBRATION/{CAM}/BoardViews/{successful_on_both:06}.png")

                # Grayscale the image if necessary
                if not self.is_realsense:
                    gray[j] = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
                else:
                    gray[j] = full_img[:,:,0] # it turns out images are saved or read with 3 channels even if they were grayscale! So we choose any of the channels, it will not matter. In fact we could simply do cvtColor but pointless averaging


                # Find the chess board corners in the image-> Telling that it should look for a 9x6 subchess
                # The detected corners will be the pixel numbers in pixel reference (9x6, 1, 2)
                found[j], detected_corners_in_image[j] = cv2.findChessboardCorners(gray[j], (self.chess_size_x,self.chess_size_y), flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE+ cv2.CALIB_CB_FAST_CHECK)

                if found[j]==True:
                    logging.info(f"\nDetected corners {detected_corners_in_image[j].shape} on image shape {full_img.shape}")

                    # Improve/Refine detected corner pixels
                    detected_corners_in_image[j] = cv2.cornerSubPix(gray[j], detected_corners_in_image[j], winSize=(11,11), zeroZone=(-1,-1), criteria=self.corner_criteria)


            # If found, in both cam images found will be True-> add object points, image points (after refining them)
            if found[0] == True and found[1]==True:
                # Draw and display the corners in the image to ccheck if correctly detected -> full_img is modified!
                cv2.drawChessboardCorners(img_L, (self.chess_size_x,self.chess_size_y), detected_corners_in_image[0], found[0])
                cv2.drawChessboardCorners(img_R, (self.chess_size_x,self.chess_size_y), detected_corners_in_image[1], found[1])

                joint_images = np.concatenate((img_L, img_R), axis=1)


                ok = self.mainThreadPlotter.emit(joint_images,
                    0 if self.allow_choosing else self.previs_ms,
                        'Processed pair of images are ok? Press SPACE if yes')


                logging.info(f"SI QUE FUNCIONA ok={ok}")
                if ((ok==32) or not self.allow_choosing): ################
                    logging.info(f"\nVALIDATED chess table number {successful_on_both}/{self.num_photos_calibration}!\n")
                    cv2.imwrite(f"./OUTPUTS/CALIBRATION/COMMON/BoardViews/{successful_on_both:06}.png", joint_images)

                    # Valid detection of corners, so we add them to our list for calibration
                    self.world_points.append(self.world_points_chess) # with origin in the chess board!

                    self.image_points[self.CAMS[0]].append(detected_corners_in_image[0])
                    self.image_points[self.CAMS[1]].append(detected_corners_in_image[1])
                    successful_on_both+=1
                else:
                    logging.info("\nTrying AGAIN!\n")
            else:
                logging.info("\nTrying AGAIN!\n")
        if self.is_realsense:
            pipeline.stop()

        self.different_chess_viewsR=sorted(glob.glob("./OUTPUTS/CALIBRATION/CAM_RIGHT/BoardViews/*.png"))
        self.different_chess_viewsL=sorted(glob.glob("./OUTPUTS/CALIBRATION/CAM_LEFT/BoardViews/*.png"))

    def use_given_photos_compute_points(self):
        # Capture at least 10 different views of the chess board - Better if 20 for example
        # Ideally the same time instance should be captured by each of the cameras
        logging.info("3. Ready to take photos and compute points... ##############################")

        # We will try to contrast with a reference subset 9x6 of the chess board image
        # We prepare the world coordinates of those points in the real world
        # We can assume z=0 and only provide x,y-s if in chess board square units:
        # (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        # if it was that the second point was 10 cm away, then (0,0,0), (10,0,0), (20,0,0) etc.
        # In our case the board squares have 2.4 cm each in width and height, so:
        self.world_points_chess = np.zeros((self.chess_size_y*self.chess_size_x,3),
                                                np.float32)
        self.world_points_chess[:,:2] = np.mgrid[0:self.chess_size_x,0:self.chess_size_y].T.reshape(-1,2)*self.chess_square_side # using chess square units

        # Arrays to store world points and image points from all the images.
        self.world_points = [] # 3d point in real world space
        self.image_points = {} # 2d points in image plane for each camera.
        self.image_points[self.CAMS[0]]=[]
        self.image_points[self.CAMS[1]]=[]
        successful_on_both=1

        while successful_on_both <= self.num_photos_calibration:
            found =[True, True]
            detected_corners_in_image=[0,0]
            gray=[0,0]
            full_img=[0,0]
            for j, CAM in enumerate(self.CAMS):
                # Read the image in numpy format
                full_img[j] = cv2.imread(f"./OUTPUTS/CALIBRATION/{CAM}/BoardViews/{successful_on_both:06}.png")
                # Grayscale the image
                if not self.is_realsense:
                    gray[j] = cv2.cvtColor(full_img[j], cv2.COLOR_BGR2GRAY)
                else:
                    gray[j] = full_img[j][:,:,0]

                # Find the chess board corners in the image-> Telling that it should look for a 9x6 subchess
                # The detected corners will be the pixel numbers in pixel reference (9x6, 1, 2)
                found[j], detected_corners_in_image[j] = cv2.findChessboardCorners(gray[j], (self.chess_size_x,self.chess_size_y), None)

                if found[j]==True:
                    logging.info(f"\nDetected corners {detected_corners_in_image[j].shape} on image shape {full_img[j].shape}")

                    # Improve/Refine detected corner pixels
                    detected_corners_in_image[j] = cv2.cornerSubPix(gray[j], detected_corners_in_image[j], (11,11), (-1,-1), self.corner_criteria)


            # If found, in both cam images found will be True-> add object points, image points (after refining them)
            if found[0] == True and found[1]==True:
                # Draw and display the corners in the image to ccheck if correctly detected -> full_img is modified!
                cv2.drawChessboardCorners(full_img[0], (self.chess_size_x,self.chess_size_y), detected_corners_in_image[0], found[0])
                cv2.drawChessboardCorners(full_img[1], (self.chess_size_x,self.chess_size_y), detected_corners_in_image[1], found[1])

                joint_images = np.concatenate((full_img[0], full_img[1]), axis=1)
                ok=self.mainThreadPlotter.emit(joint_images,
                    0 if self.allow_choosing else self.previs_ms,
                        'Processed pair of images are ok? Press SPACE if yes')

                if ((ok==32) or not self.allow_choosing): ################
                    logging.info(f"\nVALIDATED chess table number {successful_on_both}/{self.num_photos_calibration}!\n")
                    cv2.imwrite(f"./OUTPUTS/CALIBRATION/COMMON/BoardViews/{successful_on_both:06}.png", joint_images)

                    # Valid detection of corners, so we add them to our list for calibration
                    self.world_points.append(self.world_points_chess)

                    self.image_points[self.CAMS[0]].append(detected_corners_in_image[0])
                    self.image_points[self.CAMS[1]].append(detected_corners_in_image[1])
                    successful_on_both+=1
                else:
                    logging.info("\nTrying AGAIN!\n")
            else:
                logging.info("\nTrying AGAIN!\n")
        self.heightL, self.widthL, channelsL  = full_img[0].shape
        self.heightR, self.widthR, channelsR  = full_img[1].shape
        self.different_chess_viewsR=sorted(glob.glob("./OUTPUTS/CALIBRATION/CAM_RIGHT/BoardViews/*.png"))
        self.different_chess_viewsL=sorted(glob.glob("./OUTPUTS/CALIBRATION/CAM_LEFT/BoardViews/*.png"))

    def calibrate_cameras(self):
        '''
         The following recipe is used to calibrate the cameras:
         https://ieeexplore.ieee.org/document/888718
         https://ieeexplore.ieee.org/document/791289
         which essentially first considers that since world points have z=0, in reality the transf
         taking the plane of the refrence to the image is a homography! So we have that the
         intrinsic * extrinsic matrices with the z erased are equal to a homography for the
         given pairs of points. Then using that the extrinsiic rotation vecs must be orthogonal
         they derive two matrix equations for the intrinisc matrix, they re-write this as a
         homogeneous equation system that can be solved with the smallest left singular vlaue's
         singular vector as in dlt precisamente. Usando que they know several views, y ke la
         intrinsic matrix pa todas debe ser igual. Then using the fact that it was orginally a
         homography, they get the extrinsic rotation and translation vectors for each image view.
         Esto es la closed digamos formula analitica. Tras obtener con esto las estimaciones
         iniciales, lo que hacen ahora es una non-linar optimization de la maximum likelihood, que
         no es mas ke la MSE entre la proyeccion en la imagen y la prediccion de la proyeccion
         usando la extrinsic e intrinsic obtenidas. Así acaban de ajustarse el noise que pudo haber
         estadístico en las correspondencias indicadas.
         De paso usando el maximum likelihood step pueden introducir más parametros, los de
         distorsión, que también ajustarán, y así ya lo tienes todo!

         Asike efectivamente, en realidad con el output de esto, podrías obtener los vectores
         posición y rotación relativos entre las dos cámaras stereo directamente, y con ello ya
         tendrías para hacer disparity, rectification etc. Es más podrías plotear un plot toh
         rexulón de las cámaras o el board fijo y la posición en 3D de cada view del otro en movmto
         como en matlab.
        '''
        self.camera_matrices={} # intrinsic camera matrices, change of affine basis from projected point to image basis and change to pixel units
        self.distortion_coefficients={} # intrinsic distortion coefficients

        logging.info("\n\n4. CALIBRATING CAMERAS... ##########################################")

        for CAM in self.CAMS:
            # We input the world points for the chess points together with the corresponding image points in each image
            ret, self.camera_matrices[CAM], self.distortion_coefficients[CAM], rot_vecs_views, \
                    trans_vecs_views = cv2.calibrateCamera( self.world_points, # in calibration object reference
                     self.image_points[CAM], # the projected and shifted/scaled points in the camera
                    (self.widthL, self.heightL),  # shape tuple (nx, ny) of the images, just used to initialize the intrinsic param matrix
                    None,None, None, None, None,
                    criteria=self.camera_criteria)
            # camera_matrix is the affine reference change from the projected to the pixel frame.
            # it contains focal distances fx, fy and translation from the center of projection c1,c2 to the origin of pixels

            # distoriton coefficients are the parameters to correct the fisheyeness of the pinhole camera

            # Rot vecs and trans vecs are the rotation and translation vectors to go from one image view of the
            # board to the others! they are the extrinsic parameters of each view, since each
            # chess board view is seen as a different position and angle for the camera!

            # We save the camera matrices and the distortion coefficents

            np.save(f"./OUTPUTS/CALIBRATION/{CAM}/Calibrated_Parameters/Camera_Matrix_Cam_to_Pixel_{CAM}.npy", self.camera_matrices[CAM])
            np.save(f"./OUTPUTS/CALIBRATION/{CAM}/Calibrated_Parameters/Distortion_Parameters_{CAM}.npy", self.distortion_coefficients[CAM])

            # we save the camera and board view 3D plot for the obtained extrinsics:
            #self.()

            # We compute the MSE if we re-projected it all with the found parameters
            mean_error = 0
            for i in range(len(self.world_points)):
                aprox_image_Points, _ = cv2.projectPoints(self.world_points[i], rot_vecs_views[i], trans_vecs_views[i], self.camera_matrices[CAM], self.distortion_coefficients[CAM])
                error = cv2.norm(self.image_points[CAM][i], aprox_image_Points, cv2.NORM_L2)/len(aprox_image_Points)
                mean_error += error
            logging.info( f"\n{CAM} CALIBRATED!\nMSE for reprojection given distortion correction in {CAM}: {mean_error/len(self.world_points)}" )

    def compute_Fundamental_Matrix(self):
        '''
        Computa el vector y rotación reltaivos entre las dos cámaras estereo. Es aquí por primera
        vez donde consideramos que son dos cámaras sacando fotos de la misma escena, y que las
        cámaras entre ellas están a una distancia y rotación relativas fijas!
        Que en verdad con los Rk,Tk de cada view relativos al chess board plane que obtuvimos en
        calibrateCamera podrías sacar haciendo pair-wise distancias y tal, pero mejor usar esta
        función que lo hará ad-hoc para sacar el R,T relativos (dejando las intrinsic en paz, donde
        las intrinisic se calculaban más conciencudamente en el calibrateCamera).
        Pa sacar est R,T relativas entre las dos cámaras se usa el método de la matriz fundamental.
        La gracia de la matriz fundamental (que con R,T está definida) es que puedes sacar epilineas
        super facilmente
        '''
        # OBTAIN ESSENTIAL AND FUNDAMENTAL MATRICES WITH THE CALIBRATION POINTS WE GOT ########
        # Use the calbration images to obtain the Essential and Fundamental matrices
        logging.info(f"\n\n5. OBTAINING FUNDAMENTAL MATRIX ##################################\n")

        rms, self.camera_matrices["CAM_LEFT"],  self.distortion_coefficients["CAM_LEFT"], \
        self.camera_matrices["CAM_RIGHT"],  self.distortion_coefficients["CAM_RIGHT"], \
        self.R, self.T, E, self.F = cv2.stereoCalibrate(
            self.world_points, self.image_points['CAM_LEFT'], self.image_points['CAM_RIGHT'],
            self.camera_matrices["CAM_LEFT"], self.distortion_coefficients["CAM_LEFT"],
            self.camera_matrices["CAM_RIGHT"], self.distortion_coefficients["CAM_RIGHT"],
            (self.widthL, self.heightL), criteria = self.stereocalib_criteria, flags = self.stereocalib_flags)
        # highly recommended to use CV_CALIB_FIX_INTRINSIC flag, since otherwise cameras will
        # be calibrated again, but now not in such good conditions as before! ASi solo se tienen
        # que buscar R,T y la E y F

        if rms > 1.0:
            logging.warning(f"[ERROR] Fundamental Matrix reproj. RMS error < 1.0 {rms}. Re-try image capture.")
        else:
            logging.info(f"[OK] Fundamental Matrix onbtention successful RMS error= {rms}")

        np.savez("./OUTPUTS/CALIBRATION/COMMON/Calibrated_Parameters/Stereo_Calibrate_rms_KL_DL_KR_DR_R_T_E_F.npz",\
                 rms=rms, KL=self.camera_matrices["CAM_LEFT"],  DL=self.distortion_coefficients["CAM_LEFT"], \
                 KR=self.camera_matrices["CAM_RIGHT"],  DR=self.distortion_coefficients["CAM_RIGHT"], \
                 R=self.R, T=self.T, E=E, F=self.F)

    def _drawlines(self, img1src, img2src, lines, pts1src, pts2src):
        ''' img1 - grayscale image on which we draw the epilines for the points in img2 given by pts2src
            lines - corresponding epilines '''
        r, c = img1src.shape
        img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR) # convert the grayscale images to color channels
        img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
        # Edit: use the same random seed so that two images are comparable! same colour for coresponding epilines
        np.random.seed(0)
        for r, pt1, pt2 in zip(lines, pts1src, pts2src):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # y=-x*a/b-c
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
            img1color = cv2.circle(img1color, tuple(pt1[0]), 5, color, -1)
            img2color = cv2.circle(img2color, tuple(pt2[0]), 5, color, -1)
        return img1color, img2color

    def draw_Epilines(self):
        # DRAW SOME EPILINES ON THE CHESS BOARDS FOR FUN using the Fundamental Matrix ##########
        logging.info(f"\n6. DRAWING EPILINES FOR SANITY CHECK ################################\n")
        for i, (file_nameL, file_nameR) in enumerate(zip(self.different_chess_viewsL, self.different_chess_viewsR)):
            if i==5:
                break
            full_imageL = cv2.imread(file_nameL, cv2.IMREAD_GRAYSCALE)
            full_imageR = cv2.imread(file_nameR, cv2.IMREAD_GRAYSCALE)
            # Find epilines corresponding to points in right image (second image) and
            # drawing its lines on left image
            lines_L = cv2.computeCorrespondEpilines(
                self.image_points['CAM_RIGHT'][i].reshape(-1, 1, 2), 2, self.F)
            '''
            For every point in one of the two images of a stereo pair, the function finds the
            equation of the corresponding epipolar line in the other image.

            From the fundamental matrix definition, line l^{(2)}_i in
            the second image for the point p^{(1)}_i in the first image (when whichImage=1 ) is
            computed as:
                l^{(2)}_i = F p^{(1)}_i
            And viceversa changing indices.

            Claro, puedes definir una recta afín en P^2 con 3 números, que serán el vector normal
            del plano que en el proyectivizado (aka en z=1) es la recta esa. tq si es l=(a,b,c)
            será la recta ptos ortgonales a ese vector (x,y,1)*(a,b,c)=0 <-> ax+by+c=0
            y=-x*a/b-c
            La función elige las a,b,c tq a^2+b^2=1, pero no importa la norma en realidad claro.
            '''
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
            plt.savefig(f"./OUTPUTS/CALIBRATION/COMMON/Epilines_Chess_Samples/Epilines_{(i+1):06}.png", dpi=400)
            plt.clf()

    def compute_Stereo_Rectification(self):
        # OBTAIN STERO RECTIFICATION HOMOGRAPHIES #####################################
        logging.info(f"\n\n7. OBTAINING STERO RECTIFICATION HOMOGRAPHIES ######################\n")
        '''
        The function computes the rotation matrices for each camera that (virtually) make both
        camera image planes the same plane. Aka encuentra la homografía que manda uno a ser
        paralelo al otro rotacionalmente. De forma que por una parte es trivial hacer disparity
        maps si tienes puntos de uno que correspondan con los del otro.
        Pero no solo eso, sino que encima, para hacer un disparity map neceistas poner en
        correspondencia practicamente cada par de pixeles de las imágenes, por lo ke hay que
        encontrar mazo de correspondencias. Así, como poner ambas cameras en el mismo plano hace
        ke all the epipolar lines be parallel (ze el polo de uno está en el infinito del otro),
        thus simplifies the dense stereo correspondence problem, ze solo hay que buscar sobre esas
        rectas cuál es el punto de uno que corresponde con el otro, y es un grid regular al ser
        paralelas!

        R1	Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix brings points given in the unrectified first camera's coordinate system to points in the rectified first camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system. En plan homografía.
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
        self.R1, self.R2, self.P1, self.P2, self.Q, self.roi_left, self.roi_right = cv2.stereoRectify(
                self.camera_matrices["CAM_LEFT"],  self.distortion_coefficients["CAM_LEFT"], \
                self.camera_matrices["CAM_RIGHT"],  self.distortion_coefficients["CAM_RIGHT"],\
                (self.widthL, self.heightL), self.R, self.T,\
                flags=cv2.CALIB_ZERO_DISPARITY*self.flag_CALIB_ZERO_DISPARITY , alpha=self.alpha)

        np.savez("./OUTPUTS/CALIBRATION/COMMON/Calibrated_Parameters/Stereo_Rectify_R1_R2_P1_P2_Q_roi_left_roi_right.npz",
                     R1=self.R1, R2=self.R2, P1=self.P1, P2=self.P2, Q=self.Q, roi_left=self.roi_left, roi_right=self.roi_right)

    def rectify_chess(self):
        # DO THE RECTIFICATION #####################################
        logging.info(f"\n8. RECTIFING IMAGES AND UNDISTORTING THEM #########################")
        # We proceed to rectify and undistort all the images, so they can be processed on the disparity map generator
        # https://docs.opencv.org/4.1.1/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
        # OJO! MAYBE YOU SHOULD USE GRAYSCALE IMAGES IN THE RECTIFICATION!!!
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        fig.suptitle("Rectified images should have epilines parallel and aligned")

        # es un output to input mapper
        self.mapLx, self.mapLy	=	cv2.initUndistortRectifyMap( self.camera_matrices["CAM_LEFT"],
                        self.distortion_coefficients["CAM_LEFT"], self.R1, self.P1, (self.widthL, self.heightL), cv2.CV_32FC1)

        self.mapRx, self.mapRy	=	cv2.initUndistortRectifyMap( self.camera_matrices["CAM_RIGHT"],
                        self.distortion_coefficients["CAM_RIGHT"], self.R2, self.P2, (self.widthR, self.heightR), cv2.CV_32FC1)

        for i, (file_nameL, file_nameR) in enumerate(zip(self.different_chess_viewsL, self.different_chess_viewsR)):
            if i==5:
                break
            img_L = cv2.imread(file_nameL)
            img_R = cv2.imread(file_nameR)

            rectified_imageL = cv2.remap(img_L, self.mapLx, self.mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT) # output to input
            rectified_imageR = cv2.remap(img_R, self.mapRx, self.mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

            cv2.imwrite(f'./OUTPUTS/CALIBRATION/CAM_LEFT/Rectified_Chess_Samples/Rectified_{(i+1):06}.png', rectified_imageL)
            cv2.imwrite(f'./OUTPUTS/CALIBRATION/CAM_RIGHT/Rectified_Chess_Samples/Rectified_{(i+1):06}.png', rectified_imageR)

            # For sanity check we will plot the rectified images with some horizontal lines which should match the epilines of both images
            if not self.is_realsense:
                rectified_imageL=cv2.cvtColor(rectified_imageL, cv2.COLOR_BGR2GRAY)
                rectified_imageR=cv2.cvtColor(rectified_imageR, cv2.COLOR_BGR2GRAY)
            #x, y, w, h = self.roi_left
            #rectified_imageL = rectified_imageL[ x:x+w, y:y+h]
            #x, y, w, h = self.roi_right
            #rectified_imageR = rectified_imageR[y:y+h, x:x+w]
            axes[0].imshow(rectified_imageL, cmap='gray')
            axes[1].imshow(rectified_imageR, cmap='gray')

            for j in range(5,475, 20):
                axes[0].axhline(j)
                axes[1].axhline(j)

            fig.savefig(f"./OUTPUTS/CALIBRATION/COMMON/Epilines_Chess_Samples/Rectified_Epilines_{(i+1):06}.png", dpi=400)

            axes[0].clear()
            axes[1].clear()

    def compute_Disparity_Map(self):
        # OBTAIN THE DISPARITY MAP AND THE DEPTH MAP! ###################################
        logging.info(f"\n\n9. OBTAINING STERO DISPARITY AND DEPTH MAPS FOR CHESS VIEWS!!#####\n")
        '''
        # StereoSGBM Parameter explanations:
        # https://docs.opencv.org/4.5.0/d2/d85/classcv_1_1StereoSGBM.html

    minDisparity – Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
    numDisparities – Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
    blockSize – Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    P1 – The first parameter controlling the disparity smoothness. See below.
    P2 – The second parameter controlling the disparity smoothness. The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 > P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*SADWindowSize*SADWindowSize and 32*number_of_image_channels*SADWindowSize*SADWindowSize , respectively).
    disp12MaxDiff – Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
    preFilterCap – Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
    uniquenessRatio – Margin in percentage by which the best (minimum) computed cost function value should “win” the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
    speckleWindowSize – Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleRange – Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
    mode – Set it to true to run the full-scale two-pass dynamic programming algorithm. It will consume O(W*H*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false .

        P1, P2 The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 > P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8*number_of_image_channels*SADWindowSize*SADWindowSize and 32*number_of_image_channels*SADWindowSize*SADWindowSize , respectively).
        '''
        #self.square = cv2.GetValidDisparityROI(self.roi_left, self.roi_right, self.min_disp, self.num_disp)
        self.stereo_left_matcher = cv2.StereoSGBM_create(
            minDisparity=self.min_disp,
            numDisparities=self.num_disp,
            blockSize=self.block_size,
            uniquenessRatio=self.uniquenessRatio,
            speckleWindowSize=self.speckleWindowSize,
            speckleRange=self.speckleRange,
            disp12MaxDiff=self.disp12MaxDiff,
            P1=8 * 1 * self.block_size * self.block_size,
            P2=32 * 1 * self.block_size * self.block_size,
            mode=cv2.StereoSGBM_MODE_HH
        )
        # cv2.StereoSGBM_MODE_SGBM
        # cv2.StereoSGBM_MODE_HH         cv2.StereoSGBM_MODE_SGBM_3WAY
        # cv2.StereoSGBM_MODE_HH4

        # We will not simply use the stereoSGBM but also a  disparity map post-filtering
        # in order to refine the homogeneous texture regions or some occluded or discontinuous regions of depth
        # For this, instead of only computing the disparity map for the left camera, we will
        # also compute it for the right image using the same method. then a filter will
        # use both to generate a less sparse map. Following:
        # https://docs.opencv.org/master/d3/d14/tutorial_ximgproc_disparity_filtering.html

        self.stereo_right_matcher = cv2.ximgproc.createRightMatcher(self.stereo_left_matcher)
        '''
        # FILTER Parameters
        Lambda is a parameter defining the amount of regularization during filtering. Larger values force filtered disparity map edges to adhere more to source image edges. Typical value is 8000.
        SigmaColor is a parameter defining how sensitive the filtering process is to source image edges. Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0.
        visual_multiplier = 6
        '''
        self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.stereo_left_matcher)
        self.wls_filter.setLambda(self.lmbda)
        self.wls_filter.setSigmaColor(self.sigma)


        different_chess_views_rectified_R=sorted(glob.glob("./OUTPUTS/CALIBRATION/CAM_RIGHT/Rectified_Chess_Samples/*.png"))
        different_chess_views_rectified_L=sorted(glob.glob("./OUTPUTS/CALIBRATION/CAM_LEFT/Rectified_Chess_Samples/*.png"))
        def normalize_and_plot_disparity_map(disparity, tag, i):
            disparity_plot = cv2.normalize(disparity, disparity, alpha=255,
                                          beta=0, norm_type=cv2.NORM_MINMAX)
            disparity_plot = np.uint8(disparity_plot)
            #disparity_plot = (255*(disparity-disparity.min())/(disparity.max()-disparity.min())).astype(np.uint8)
            cv2.imwrite(f"./OUTPUTS/CALIBRATION/COMMON/Disparities_Chess_Samples/Normalized_{tag}_disparity_{(i+1):06}.png", disparity_plot)

        for i, (file_nameL, file_nameR) in enumerate(zip(different_chess_views_rectified_L, different_chess_views_rectified_R)):
            rectified_imageL = cv2.imread(file_nameL)
            rectified_imageR = cv2.imread(file_nameR)
            if not self.is_realsense:
                grayL = cv2.cvtColor(rectified_imageL, cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(rectified_imageR, cv2.COLOR_BGR2GRAY)
            else:
                grayL = rectified_imageL[:,:,0]
                grayR = rectified_imageR[:,:,0]

            # Compute DISPARITY MAPS from L and R
            disparity_L = self.stereo_left_matcher.compute(grayL, grayR)  # .astype(np.float32)/16
            disparity_R = self.stereo_right_matcher.compute(grayR, grayL)  # .astype(np.float32)/16
            logging.info(f"dtype disparity_L {disparity_L.dtype} ({disparity_L.min()}, {disparity_L.max()}) {disparity_R.dtype} ({disparity_R.min()}, {disparity_R.max()}) ")
            #disparity_L = np.int16(disparity_L)
            #disparity_R = np.int16(disparity_R)
            # Use both to generate a filtered disparity map for the left image
            filtered_disparity = self.wls_filter.filter(disparity_L, grayL, disparity_map_right=disparity_R)  # important to put "imgL" here!!! Maybe can use the colored image here!
            logging.info(f"filtered_disparity {filtered_disparity.dtype} ({filtered_disparity.min()}, {filtered_disparity.max()})")


            # Normalize the values to a range from 0..255 for a grayscale image of the disparity map
            normalize_and_plot_disparity_map(disparity_L, "L", i)
            normalize_and_plot_disparity_map(disparity_R, "R", i)
            normalize_and_plot_disparity_map(filtered_disparity, "Filtered", i)

            # Obtain the DEPTH MAP
            # According to the documentation we should do this division and conversion
            filtered_disparity = (filtered_disparity/16.0).astype(np.float32)

            # And voila, we get an array [h,w,3] with the 3D coordinates (in the units of the chess world points we inputed)
            # of each pixel in the image of the Left! Thus we chose the Left camera to be better
            # Input single-channel 8-bit unsigned, 16-bit signed, 32-bit signed or 32-bit floating-point disparity image
            image_3D = cv2.reprojectImageTo3D(filtered_disparity, self.Q, handleMissingValues=True)
            # handleMissingValues	Indicates, whether the function should handle missing values (i.e. points where the disparity was not computed). If handleMissingValues=true, then pixels with the minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed to 3D points with a very large Z value (currently set to 10000)
            image_3D = np.where(image_3D>9000, 0, image_3D)
            self._plot_disparity_and_3D(f"./OUTPUTS/CALIBRATION/COMMON/Disparities_Chess_Samples/3D_point_cloud_{(i+1):06}.png", filtered_disparity, rectified_imageL, image_3D)

        # https://stackoverflow.com/questions/58150354/image-processing-bad-quality-of-disparity-image-with-opencv
        # Kriston erreferentzixe!
        # https://www.programcreek.com/python/?code=aliyasineser%2FstereoDepth%2FstereoDepth-master%2Fstereo_depth.py#

    def _plot_disparity_and_3D(self, path, filtered_disparity, rectified_imageL, image_3D):
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(221)
        ax.imshow(rectified_imageL[:,:,::-1])
        ax.set_title("Rectified L image")
        ax = fig.add_subplot(222)
        c = ax.imshow(filtered_disparity, cmap='gray')
        fig.colorbar(c, ax=ax, orientation='horizontal')
        ax.set_title("Smoothed disparity range image")
        # plot the 3D map with depth and maybe a point cloud
        (h, w, c) = np.shape(rectified_imageL)
        # X and Y coordinates of points in the image, spaced by 10.
        (X, Y) = np.meshgrid(range(0, w,5), range(0, h,5))
        ax = fig.add_subplot(223, projection='3d')
        ax.scatter3D(X, Y, filtered_disparity[Y,X], c=rectified_imageL[Y,X,::-1].reshape(-1,3)/255.0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=30, azim=93)
        ax.set_title("Disparity in 3D")

        ax = fig.add_subplot(224, projection='3d')
        image_3D = image_3D[Y,X].reshape(-1, 3)
        ax.scatter3D(image_3D[:,0], image_3D[:,1], image_3D[:,2], c=rectified_imageL[Y,X,::-1].reshape(-1,3)/255.0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim((0,650))
        ax.set_ylim((0, 480))
        ax.set_zlim((0, 900))
        ax.view_init(elev=30, azim=93)
        ax.set_title("Image reprojected to 3D")
        plt.savefig(path)


    def do_Test(self, use_taken_photos_test):
        logging.info(f"\n\n10. Taking photos for life test...#############")
        # NOW WE WILL RECORD A TEST FILM TO CHECK THE CORRECTNESS OF THE MAPS

        def normalize_disparity_map(disparity):
            disparity = cv2.normalize(disparity, disparity, alpha=255,
                                          beta=0, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
            #disparity = (255*(disparity-disparity.min())/(disparity.max()-disparity.min())).astype(np.uint8)
            return disparity

        if self.is_realsense:
            pipeline = rs.pipeline()
            profile = pipeline.start(self.config)
            for warm_up in range(20):
                frames = pipeline.wait_for_frames()

        for j in range(self.num_photos_test):
            logging.info(f"\n\nTAKING PHOTOS {j}/{self.num_photos_test} ########################")
            # instead of using .read() to get an image we decompose it into .grab and then .retrieve
            # so we can maximize the sinchronization
            if not use_taken_photos_test:
                if self.is_realsense:
                    frames = pipeline.wait_for_frames()
                    ir1_frame = frames.get_infrared_frame(1) # Left IR Camera, it allows 0, 1 or no input
                    ir2_frame = frames.get_infrared_frame(2) # Right IR camera
                    img_L = np.asanyarray( ir1_frame.get_data() )
                    img_R = np.asanyarray( ir2_frame.get_data() )

                else:
                    if not (self.vidStreamR.grab() and self.vidStreamL.grab()):
                        logging.warning("[Error] Getting the image for this iteration. Retrying...")
                        continue
                    _, img_R = self.vidStreamR.retrieve()
                    _, img_L = self.vidStreamL.retrieve()
                cv2.imwrite(f"./OUTPUTS/CALIBRATION/COMMON/Life_Test/Life_L_{j}.png", img_L)
                cv2.imwrite(f"./OUTPUTS/CALIBRATION/COMMON/Life_Test/Life_R_{j}.png", img_R)

            else:
                img_L = cv2.imread(f"./OUTPUTS/CALIBRATION/COMMON/Life_Test/Life_L_{j}.png")
                img_R = cv2.imread(f"./OUTPUTS/CALIBRATION/COMMON/Life_Test/Life_R_{j}.png")

            # RECTIFY THE IMAGE
            rect_img_L = cv2.remap(img_L, self.mapLx, self.mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
            rect_img_R = cv2.remap(img_R, self.mapRx, self.mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
            result = np.concatenate((rect_img_L,rect_img_R), axis=1)
            cv2.imwrite(f"./OUTPUTS/CALIBRATION/COMMON/Life_Test/Life_{j}_color.png", result)

            # GRAYSCALE THE IMAGES
            if not self.is_realsense:
                grayL = cv2.cvtColor(rect_img_L, cv2.COLOR_BGR2GRAY)
                grayR = cv2.cvtColor(rect_img_R, cv2.COLOR_BGR2GRAY)
            else:
                if not use_taken_photos_test:
                    grayL = rect_img_L
                    grayR = rect_img_R
                else:
                    grayL = rect_img_L[:,:,0]
                    grayR = rect_img_R[:,:,0]

            # COMPUTE DISPARITIES

            disparity_L = self.stereo_left_matcher.compute(grayL, grayR)  # .astype(np.float32)/16
            disparity_R = self.stereo_right_matcher.compute(grayR, grayL)  # .astype(np.float32)/16
            #disparity_L = np.int16(disparity_L)
            #disparity_R = np.int16(disparity_R)
            filtered_disparity = self.wls_filter.filter(disparity_L, grayL, disparity_map_right=disparity_R)  # important to put "imgL" here!!! Maybe can use the colored image here!

            total_unfiltered = np.concatenate((normalize_disparity_map(disparity_L), normalize_disparity_map(disparity_R)), axis=1)
            total_filtered = np.concatenate( (normalize_disparity_map(filtered_disparity), np.zeros(filtered_disparity.shape)), axis=1 )
            joint_images = np.concatenate((total_unfiltered, total_filtered), axis=0)
            cv2.imwrite(f"./OUTPUTS/CALIBRATION/COMMON/Life_Test/Life_{j}.png", joint_images)
            self.mainThreadPlotter.emit(result, 2000, f'Life Test {j}')

            image_3D = cv2.reprojectImageTo3D(filtered_disparity, self.Q, handleMissingValues=True)
            image_3D = np.where(image_3D>9000, 0, image_3D)
            if self.is_realsense:
                rect_img_L = np.stack((rect_img_L, rect_img_L, rect_img_L), axis=-1)
            self._plot_disparity_and_3D(f"./OUTPUTS/CALIBRATION/COMMON/Life_Test/Life_point_cloud_{j}.png", filtered_disparity, rect_img_L, image_3D)
        if self.is_realsense:
            pipeline.stop()




    def plot_camera_and_chess_boards(self):
        '''
        WIP adaption of:
            https://github.com/opencv/opencv/pull/10354
        '''
        def inverse_homogeneoux_matrix(M):
            R = M[0:3, 0:3]
            T = M[0:3, 3]
            M_inv = np.identity(4)
            M_inv[0:3, 0:3] = R.T
            M_inv[0:3, 3] = -(R.T).dot(T)
            return M_inv
        def transform_to_matplotlib_frame(cMo, X, inverse=False):
            M = np.identity(4)
            M[1,1] = 0
            M[1,2] = 1
            M[2,1] = -1
            M[2,2] = 0
            if inverse:
                return M.dot(inverse_homogeneoux_matrix(cMo).dot(X))
            else:
                return M.dot(cMo.dot(X))
        def create_camera_model(camera_matrix, width, height, scale_focal, draw_frame_axis=False):
            fx = camera_matrix[0,0]
            fy = camera_matrix[1,1]
            focal = 2 / (fx + fy)
            f_scale = scale_focal * focal
            # draw image plane
            X_img_plane = np.ones((4,5))
            X_img_plane[0:3,0] = [-width, height, f_scale]
            X_img_plane[0:3,1] = [width, height, f_scale]
            X_img_plane[0:3,2] = [width, -height, f_scale]
            X_img_plane[0:3,3] = [-width, -height, f_scale]
            X_img_plane[0:3,4] = [-width, height, f_scale]
            # draw triangle above the image plane
            X_triangle = np.ones((4,3))
            X_triangle[0:3,0] = [-width, -height, f_scale]
            X_triangle[0:3,1] = [0, -2*height, f_scale]
            X_triangle[0:3,2] = [width, -height, f_scale]
            # draw camera
            X_center1 = np.ones((4,2))
            X_center1[0:3,0] = [0, 0, 0]
            X_center1[0:3,1] = [-width, height, f_scale]
            X_center2 = np.ones((4,2))
            X_center2[0:3,0] = [0, 0, 0]
            X_center2[0:3,1] = [width, height, f_scale]
            X_center3 = np.ones((4,2))
            X_center3[0:3,0] = [0, 0, 0]
            X_center3[0:3,1] = [width, -height, f_scale]
            X_center4 = np.ones((4,2))
            X_center4[0:3,0] = [0, 0, 0]
            X_center4[0:3,1] = [-width, -height, f_scale]
            # draw camera frame axis
            X_frame1 = np.ones((4,2))
            X_frame1[0:3,0] = [0, 0, 0]
            X_frame1[0:3,1] = [f_scale/2, 0, 0]
            X_frame2 = np.ones((4,2))
            X_frame2[0:3,0] = [0, 0, 0]
            X_frame2[0:3,1] = [0, f_scale/2, 0]
            X_frame3 = np.ones((4,2))
            X_frame3[0:3,0] = [0, 0, 0]
            X_frame3[0:3,1] = [0, 0, f_scale/2]
            if draw_frame_axis:
                return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4, X_frame1, X_frame2, X_frame3]
            else:
                return [X_img_plane, X_triangle, X_center1, X_center2, X_center3, X_center4]
        def create_board_model(extrinsics, board_width, board_height, square_size, draw_frame_axis=False):
            width = board_width*square_size
            height = board_height*square_size
            # draw calibration board
            X_board = np.ones((4,5))
            X_board_cam = np.ones((extrinsics.shape[0],4,5))
            X_board[0:3,0] = [0,0,0]
            X_board[0:3,1] = [width,0,0]
            X_board[0:3,2] = [width,height,0]
            X_board[0:3,3] = [0,height,0]
            X_board[0:3,4] = [0,0,0]
            # draw board frame axis
            X_frame1 = np.ones((4,2))
            X_frame1[0:3,0] = [0, 0, 0]
            X_frame1[0:3,1] = [height/2, 0, 0]
            X_frame2 = np.ones((4,2))
            X_frame2[0:3,0] = [0, 0, 0]
            X_frame2[0:3,1] = [0, height/2, 0]
            X_frame3 = np.ones((4,2))
            X_frame3[0:3,0] = [0, 0, 0]
            X_frame3[0:3,1] = [0, 0, height/2]
            if draw_frame_axis:
                return [X_board, X_frame1, X_frame2, X_frame3]
            else:
                return [X_board]
        def draw_camera_boards(ax, camera_matrix, cam_width, cam_height, scale_focal,
                               extrinsics, board_width, board_height, square_size,
                               patternCentric):
            min_values = np.zeros((3,1))
            min_values = np.inf
            max_values = np.zeros((3,1))
            max_values = -np.inf
            if patternCentric:
                X_moving = create_camera_model(camera_matrix, cam_width, cam_height, scale_focal)
                X_static = create_board_model(extrinsics, board_width, board_height, square_size)
            else:
                X_static = create_camera_model(camera_matrix, cam_width, cam_height, scale_focal, True)
                X_moving = create_board_model(extrinsics, board_width, board_height, square_size)
            cm_subsection = linspace(0.0, 1.0, extrinsics.shape[0])
            colors = [ cm.jet(x) for x in cm_subsection ]
            for i in range(len(X_static)):
                X = np.zeros(X_static[i].shape)
                for j in range(X_static[i].shape[1]):
                    X[:,j] = transform_to_matplotlib_frame(np.eye(4), X_static[i][:,j])
                ax.plot3D(X[0,:], X[1,:], X[2,:], color='r')
                min_values = np.minimum(min_values, X[0:3,:].min(1))
                max_values = np.maximum(max_values, X[0:3,:].max(1))
            for idx in range(extrinsics.shape[0]):
                R, _ = cv.Rodrigues(extrinsics[idx,0:3])
                cMo = np.eye(4,4)
                cMo[0:3,0:3] = R
                cMo[0:3,3] = extrinsics[idx,3:6]
                for i in range(len(X_moving)):
                    X = np.zeros(X_moving[i].shape)
                    for j in range(X_moving[i].shape[1]):
                        X[0:4,j] = transform_to_matplotlib_frame(cMo, X_moving[i][0:4,j], patternCentric)
                    ax.plot3D(X[0,:], X[1,:], X[2,:], color=colors[idx])
                    min_values = np.minimum(min_values, X[0:3,:].min(1))
                    max_values = np.maximum(max_values, X[0:3,:].max(1))
            return min_values, max_values

        parser = argparse.ArgumentParser(description='Plot camera calibration extrinsics.',
                                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--calibration', type=str, default="../data/left_intrinsics.yml",
                            help='YAML camera calibration file.')
        parser.add_argument('--cam_width', type=float, default=0.064/2,
                            help='Width/2 of the displayed camera.')
        parser.add_argument('--cam_height', type=float, default=0.048/2,
                            help='Height/2 of the displayed camera.')
        parser.add_argument('--scale_focal', type=float, default=40,
                            help='Value to scale the focal length.')
        parser.add_argument('--patternCentric', action='store_true',
                            help='The calibration board is static and the camera is moving.')
        args = parser.parse_args()

        skip_lines = 2
        with open(args.calibration) as infile:
            for i in range(skip_lines):
                _ = infile.readline()
            data = yaml.load(infile)

        board_width = data['board_width']
        board_height = data['board_height']
        square_size = data['square_size']
        camera_matrix = data['camera_matrix']
        extrinsics = data['extrinsic_parameters']

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_aspect("equal")
        cam_width = args.cam_width
        cam_height = args.cam_height
        scale_focal = args.scale_focal
        min_values, max_values = draw_camera_boards(ax, camera_matrix, cam_width, cam_height,
                                                    scale_focal, extrinsics, board_width,
                                                    board_height, square_size, args.patternCentric)
        X_min = min_values[0]
        X_max = max_values[0]
        Y_min = min_values[1]
        Y_max = max_values[1]
        Z_min = min_values[2]
        Z_max = max_values[2]
        max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / 2.0
        mid_x = (X_max+X_min) * 0.5
        mid_y = (Y_max+Y_min) * 0.5
        mid_z = (Z_max+Z_min) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('-y')
        ax.set_title('Extrinsic Parameters Visualization')
        plt.show()
