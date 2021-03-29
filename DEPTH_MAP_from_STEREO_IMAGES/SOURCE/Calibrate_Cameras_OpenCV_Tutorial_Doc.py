import numpy as np
import cv2
import glob
import os

num_photos = 10

vidStreamL = cv2.VideoCapture(0)  # index of your camera
vidStreamR = cv2.VideoCapture(1)  # index of your camera

CAMS=["CAM_RIGHT", "CAM_LEFT"]

# termination criteria for the optimizations
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# We will try to contrast with a reference subset 7x6 of the chess board image
# We prepare the world coordinates of those points in the real world
# We can assume z=0 and only provide x,y-s if in chess board square units:
# (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# if it was that the second point was 10 cm away, then (0,0,0), (10,0,0), (20,0,0) etc.
world_points_chess = np.zeros((6*7,3), np.float32)
world_points_chess[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2) # using chess square units

# Arrays to store world points and image points from all the images.
world_points = [] # 3d point in real world space
image_points = {} # 2d points in image plane for each camera.

# Build directory structure
for CAM in CAMS:
    try:
        os.mkdir(f"{CAM}")
    except OSError as error:
        pass
    try:
        os.mkdir(f"{CAM}/BoardViews/")
    except OSError as error:
        pass
    image_points[CAM] = [] 
    
successful_on_both=1

while successful_on_both <= num_photos:
    # Capture at least 10 different views of the chess board - Better if 20 for example
    # Ideally the same time instance should be captured by each of the cameras
    retL, img_L = vidStreamL.read()
    heightL, widthL, channelsL  = imgL.shape
    retR, img_R = vidStreamR.read()
    heightR, widthR, channelsR  = imgR.shape
    
    # Crop images to same size
    
    
    # Save them now
    cv2.imwrite(f"CAM_LEFT/BoardView/{successful_on_both}.png", img_L)
    cv2.imwrite(f"CAM_RIGHT/BoardView/{successful_on_both}.png", img_R)
    
    found =[True, True]
    detected_corners_in_image=[0,0]
    gray=[0,0]
    for j, CAM in enumerate(CAMS):
        # Read the image in numpy format
        full_img = cv2.imread(f"{CAM}/BoardView/{successful_on_both}.png")
        # Grayscale the image
        gray[j] = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)
    
        # Find the chess board corners in the image-> Telling that it should look for a 7x6 subchess
        # The detected corners will be the pixel numbers in pixel reference (7x6, 1, 2)
        found[j], detected_corners_in_image[j] = cv2.findChessboardCorners(gray[j], (7,6), None)
        print(f"Detected corners {detected_corners_in_image.shape} on image shape {full_img.shape}")
        
        if found[j]==True:
            # Improve/Refine detected corner pixels
            cv2.cornerSubPix(gray[j], detected_corners_in_image[j], (11,11), (-1,-1), criteria)
 
            
    # If found, in both cam images found wil be True-> add object points, image points (after refining them)
    if found[0] == True and found[1]==True:
        # Draw and display the corners in the image to ccheck if correctly detected -> full_img is modified!
        cv2.drawChessboardCorners(full_img, (7,6), detected_corners_in_image, ret)
        cv2.imshow('img', full_img)
        cv2.imshow('img', img_L)

        ok = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if (ok==32):
        
            print("Valid chess table!")
            # Valid detection of corners, so we add them to our list for calibration
            world_points.append(world_points_chess)        
            # Add them as well
            image_points[CAM[0]].append(detected_corners_in_image[0])
            image_points[CAM[1]].append(detected_corners_in_image[1])
            successful_on_both+=1
    

camera_matrices={}
distortion_coefficients={}

for CAM in CAMS:
    # We input the world points for the chess points together with the corresponfing image points in each image
    ret, camera_matrices[CAM], distortion_coefficients[CAM], rot_vecs_views, trans_vecs_views = cv2.calibrateCamera( world_points, image_points[CAM],  
                                                       gray[0].shape[::-1],  # shape tuple (nx, ny) of the images, just used to initialize the intrinsic param matrix
                                                       None,None)
    # camera_matrix is the affine reference change from the projected to the pixel frame.
    # it contains focal distances fx, fy and translation from the center of projection c1,c2 to the origin of pixels
    
    # distoriton coefficients are the parameters to correct the fisheyeness of the pinhole camera
    
    # Rot vecs and trans vecs are the rotation and translation vectors to go from one image view of the 
    # board to the others
    
    # We save the camera matrices and the distortion coefficents
    try:
        os.mkdir(f"{CAM}/Calibrated_Parameters/")
    except OSError as error:
        pass
        
    np.save(f"{CAM}/Calibrated_Parameters/Camera_Matrix_Cam_to_Pixel_{CAM}.npy", camera_matrices[CAM])
    np.save(f"{CAM}/Calibrated_Parameters/Distortion_Parameters_{CAM}.npy", distortion_coefficients[CAM])
    
    # We compute the error if we re-projected it all with the found parameters
    mean_error = 0
    for i in range(len(world_points)):
        aprox_image_Points, _ = cv2.projectPoints(world_points[i], rot_vecs_views[i], trans_vecs_views[i], camera_matrices[CAM], distortion_coefficients[CAM])
        error = cv2.norm(image_points[CAM][i], aprox_image_Points, cv2.NORM_L2)/len(aprox_image_Points)
        mean_error += error
    print( "MSE for reprojection given distortion correction in {CAM}: {}".format(mean_error/len(world_points)) )
    
    
# UNDISTORING ONE IMAGE ##################################################################

# Now that we have the camera matrix and the distortion parameters we can refine the camera matrix
# for each given new image we want to Undistort. The thing is that when undistorting there will be some
# pixels that will be left in black ze era ojo de pez y ahora es rectangular!

for file_name in different_chess_views:
    full_image = cv2.imread(file_name)
    h,  w = full_image.shape[:2] # take height and width of the image
    # the roi is a shape giving the subset of the image that does not have invented pixels
    refined_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w,h), 1, (w,h))
    
    # undistort the image 
    undistorted = cv2.undistort(full_image, camera_matrix, distortion_coefficients, None, refined_camera_matrix)
    # crop the image toa void black pixels
    x, y, w, h = roi
    undistorted = undistorted[y:y+h, x:x+w]

    try:
        os.mkdir(f"{CAM}/Undistorted_Chess_Samples/")
    except OSError as error:
        pass
    
    cv2.imwrite(f"{CAM}/Undistorted_Chess_Samples/{file_name[13+len(CAM):-4]}_undistorted.png", undistorted)

# OBTAIN ESSENTIAL AND FUNDAMENTAL MATRICES WITH THE CALIBRATION POINTS WE GOT
# Use the claibration images to obtain the Essential and Fundamental matrices

# say we are inputing the camera matrix and the distortions
flag = 0
flag |= cv2.CALIB_FIX_INTRINSIC
flag |= cv2.CALIB_USE_INTRINSIC_GUESS

cam_Matrix_L, distortion_coeffs_L, cam_Matrix_R, distortion_coeffs_R, R, T, E, F = cv2.stereoCalibrate(
    world_points[CAM_LEFT], image_points[CAM_LEFT], image_points[RIGHT_LEFT], cam_Matrix_L, distortion_coeffs_L, cam_Matrix_R, distortion_coeffs_R, image_size)

assert ret < 1.0, "[ERROR] Calibration RMS error < 1.0 (%i). Re-try image capture." % (ret)
print("[OK] Calibration successful w/ RMS error=" + str(ret))

R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F	=	cv.stereoCalibrate(	objectPoints, imagePoints1, imagePoints2, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize[, R[, T[, E[, F[, flags[, criteria]]]]]]




















