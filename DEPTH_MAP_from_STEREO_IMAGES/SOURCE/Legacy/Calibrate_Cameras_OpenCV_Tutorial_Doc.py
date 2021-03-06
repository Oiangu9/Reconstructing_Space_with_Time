import numpy as np
import cv2
import glob
import os
import shutil
import matplotlib.pyplot as plt

num_photos = 30

# The left camera image will be the final deoth map so better the good resolution one
vidStreamL = cv2.VideoCapture(2)  # index of OBS - Nirie
vidStreamR = cv2.VideoCapture(1)  # index of Droidcam camera - Izeko

# Change the resolution if needed
vidStreamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
vidStreamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

vidStreamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
vidStreamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

CAMS=["CAM_LEFT", "CAM_RIGHT"]



# We will try to contrast with a reference subset 9x6 of the chess board image
# We prepare the world coordinates of those points in the real world
# We can assume z=0 and only provide x,y-s if in chess board square units:
# (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
# if it was that the second point was 10 cm away, then (0,0,0), (10,0,0), (20,0,0) etc.
# In our case the board squares have 2.4 cm each in width and height, so:
world_points_chess = np.zeros((6*9,3), np.float32)
world_points_chess[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) # using chess square units

# Arrays to store world points and image points from all the images.
world_points = [] # 3d point in real world space
image_points = {} # 2d points in image plane for each camera.

# Build directory structure

shutil.rmtree(f"../CAM_RIGHT", ignore_errors=True)
shutil.rmtree(f"../CAM_LEFT", ignore_errors=True)
shutil.rmtree(f"../COMMON", ignore_errors=True)

for CAM in CAMS+["COMMON"]:
    image_points[CAM] = []

    try:
        os.mkdir(f"../{CAM}")
    except OSError as error:
        pass

    try:
        os.mkdir(f"../{CAM}/BoardViews/")
    except OSError as error:
        pass

    try:
        os.mkdir(f"../{CAM}/Calibrated_Parameters/")
    except OSError as error:
        pass

    try:
        if CAM!="COMMON":
            os.mkdir(f"../{CAM}/Undistorted_Chess_Samples/")
    except OSError as error:
        pass

    try:
        if CAM!="COMMON":
            os.mkdir(f"../{CAM}/Rectified_Chess_Samples/")
    except OSError as error:
        pass

try:
    os.mkdir(f"../COMMON/Epilines_Chess_Samples/")
except OSError as error:
    pass
try:
    os.mkdir(f"../COMMON/Disparities_Chess_Samples/")
except OSError as error:
    pass

try:
    os.mkdir(f"../COMMON/Live_Test/")
except OSError as error:
    pass


successful_on_both=1
# termination criteria for the optimization of corner detections
corner_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 1e-9)

while successful_on_both <= num_photos:
    # Capture at least 10 different views of the chess board - Better if 20 for example
    # Ideally the same time instance should be captured by each of the cameras
    print(f"\n\nPHOTOS {successful_on_both}/{num_photos} ########################")
    #input("Press ENTER to take the photos:")
    # instead of using .read() to get an image we decompose it into .grab and then .retrieve
    # so we can maximize the sinchronization

    if not (vidStreamR.grab() and vidStreamL.grab()):
        print("[Error] Getting the image for this iteration. Retrying...")
        continue
    _, img_R = vidStreamR.retrieve()
    _, img_L = vidStreamL.retrieve()

    heightL, widthL, channelsL  = img_L.shape
    heightR, widthR, channelsR  = img_R.shape

    # Crop images to same size-> Not necessary
    print(f"\nLeft image: {widthL}x{heightL}x{channelsL}, Right image: {widthR}x{heightR}x{channelsR}")

    # Save them now
    cv2.imwrite(f"../CAM_LEFT/BoardViews/{successful_on_both:06}.png", img_L)
    cv2.imwrite(f"../CAM_RIGHT/BoardViews/{successful_on_both:06}.png", img_R)


    found =[True, True]
    detected_corners_in_image=[0,0]
    gray=[0,0]
    for j, CAM in enumerate(CAMS):
        # Read the image in numpy format
        full_img = cv2.imread(f"../{CAM}/BoardViews/{successful_on_both:06}.png")
        #heightL, widthL, channelsL  = full_img.shape ############
        #heightR, widthR, channelsR  = full_img.shape

        # Grayscale the image
        gray[j] = cv2.cvtColor(full_img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners in the image-> Telling that it should look for a 9x6 subchess
        # The detected corners will be the pixel numbers in pixel reference (9x6, 1, 2)
        found[j], detected_corners_in_image[j] = cv2.findChessboardCorners(gray[j], (9,6), None)

        if found[j]==True:
            print(f"\nDetected corners {detected_corners_in_image[j].shape} on image shape {full_img.shape}")

            # Improve/Refine detected corner pixels
            cv2.cornerSubPix(gray[j], detected_corners_in_image[j], (11,11), (-1,-1), corner_criteria)


    # If found, in both cam images found will be True-> add object points, image points (after refining them)
    if found[0] == True and found[1]==True:
        # Draw and display the corners in the image to ccheck if correctly detected -> full_img is modified!
        cv2.drawChessboardCorners(img_L, (9,6), detected_corners_in_image[0], found[0])
        cv2.drawChessboardCorners(img_R, (9,6), detected_corners_in_image[1], found[1])

        joint_images = np.concatenate((img_L, img_R), axis=1)
        cv2.imshow('Processed pair of images are ok? Press SPACE if yes', joint_images)
        ok = cv2.waitKey(4000) #########
        cv2.destroyAllWindows()

        if (ok!=32): ################
            print(f"\nVALIDATED chess table number {successful_on_both}/{num_photos}!\n")
            cv2.imwrite(f"../COMMON/BoardViews/{successful_on_both:06}.png", joint_images)

            # Valid detection of corners, so we add them to our list for calibration
            world_points.append(world_points_chess)

            image_points[CAMS[0]].append(detected_corners_in_image[0])
            image_points[CAMS[1]].append(detected_corners_in_image[1])
            successful_on_both+=1
        else:
            print("\nTrying AGAIN!\n")
    else:
        print("\nTrying AGAIN!\n")


camera_matrices={}
distortion_coefficients={}

print("\n\nCALIBRATING CAMERAS! ###############################################")

camera_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-5)

for CAM in CAMS:
    # We input the world points for the chess points together with the corresponding image points in each image
    ret, camera_matrices[CAM], distortion_coefficients[CAM], rot_vecs_views, \
            trans_vecs_views = cv2.calibrateCamera( world_points, image_points[CAM],
            gray[0].shape[::-1],  # shape tuple (nx, ny) of the images, just used to initialize the intrinsic param matrix
            None,None, criteria=camera_criteria)
    # camera_matrix is the affine reference change from the projected to the pixel frame.
    # it contains focal distances fx, fy and translation from the center of projection c1,c2 to the origin of pixels

    # distoriton coefficients are the parameters to correct the fisheyeness of the pinhole camera

    # Rot vecs and trans vecs are the rotation and translation vectors to go from one image view of the
    # board to the others

    # We save the camera matrices and the distortion coefficents

    np.save(f"../{CAM}/Calibrated_Parameters/Camera_Matrix_Cam_to_Pixel_{CAM}.npy", camera_matrices[CAM])
    np.save(f"../{CAM}/Calibrated_Parameters/Distortion_Parameters_{CAM}.npy", distortion_coefficients[CAM])

    # We compute the error if we re-projected it all with the found parameters
    mean_error = 0
    for i in range(len(world_points)):
        aprox_image_Points, _ = cv2.projectPoints(world_points[i], rot_vecs_views[i], trans_vecs_views[i], camera_matrices[CAM], distortion_coefficients[CAM])
        error = cv2.norm(image_points[CAM][i], aprox_image_Points, cv2.NORM_L2)/len(aprox_image_Points)
        mean_error += error
    print( f"\n{CAM} CALIBRATED!\nMSE for reprojection given distortion correction in {CAM}: {mean_error/len(world_points)}" )


# UNDISTORTING THE CALIBRATION IMAGES ##################################################################

# Now that we have the camera matrix and the distortion parameters we can refine the camera matrix
# for each given new image we want to Undistort. The thing is that when undistorting there will be some
# pixels that will be left in black ze era ojo de pez y ahora es rectangular!

different_chess_viewsR=sorted(glob.glob("../CAM_RIGHT/BoardViews/*.png"))
different_chess_viewsL=sorted(glob.glob("../CAM_LEFT/BoardViews/*.png"))

# the roi is a shape giving the subset of the image that does not have invented pixels
refined_camera_matrixL, roiL = cv2.getOptimalNewCameraMatrix(camera_matrices["CAM_LEFT"], distortion_coefficients["CAM_LEFT"], (widthL,heightL), 1, (widthL,heightL))
refined_camera_matrixR, roiR = cv2.getOptimalNewCameraMatrix(camera_matrices["CAM_RIGHT"], distortion_coefficients["CAM_RIGHT"], (widthR,heightR), 1, (widthR,heightR))

for file_nameL, file_nameR in zip(different_chess_viewsL, different_chess_viewsR):
    full_imageL = cv2.imread(file_nameL)
    full_imageR = cv2.imread(file_nameR)
    # undistort the image
    undistortedL = cv2.undistort(full_imageL, camera_matrices["CAM_LEFT"], distortion_coefficients["CAM_LEFT"], None, refined_camera_matrixL)
    undistortedR = cv2.undistort(full_imageR, camera_matrices["CAM_RIGHT"], distortion_coefficients["CAM_RIGHT"], None, refined_camera_matrixR)

    # crop the image to avoid black pixels
    x, y, w, h = roiL
    undistortedL = undistortedL[y:y+h, x:x+w]
    x, y, w, h = roiR
    undistortedR = undistortedR[y:y+h, x:x+w]

    cv2.imwrite(f"../CAM_LEFT/Undistorted_Chess_Samples/{file_nameL[13+len(CAM):-4]}_undistorted.png", undistortedL)
    cv2.imwrite(f"../CAM_RIGHT/Undistorted_Chess_Samples/{file_nameL[13+len(CAM):-4]}_undistorted.png", undistortedR)

# OBTAIN ESSENTIAL AND FUNDAMENTAL MATRICES WITH THE CALIBRATION POINTS WE GOT ########
# Use the calbration images to obtain the Essential and Fundamental matrices
print(f"\n\nOBTAINING FUNDAMENTAL MATRIX #######################################\n")

# say we are inputing the camera matrix and the distortions to the calculation
stereocalib_flags = cv2.CALIB_USE_INTRINSIC_GUESS
#stereocalib_flags |= cv2.CALIB_FIX_INTRINSIC
#stereocalib_flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
#stereocalib_flags |= cv2.CALIB_FIX_FOCAL_LENGTH
stereocalib_flags |= cv2.CALIB_FIX_ASPECT_RATIO
#stereocalib_flags |= cv2.CALIB_SAME_FOCAL_LENGTH
#stereocalib_flags |= cv2.CALIB_ZERO_TANGENT_DIST
stereocalib_flags |= cv2.CALIB_FIX_K1
stereocalib_flags |= cv2.CALIB_FIX_K2
stereocalib_flags |= cv2.CALIB_FIX_K3
stereocalib_flags |= cv2.CALIB_FIX_K4
stereocalib_flags |= cv2.CALIB_FIX_K5
stereocalib_flags |= cv2.CALIB_FIX_K6

stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-5)

rms, camera_matrices["CAM_LEFT"],  distortion_coefficients["CAM_LEFT"], \
camera_matrices["CAM_RIGHT"],  distortion_coefficients["CAM_RIGHT"], \
R, T, E, F = cv2.stereoCalibrate(
    world_points, image_points['CAM_LEFT'], image_points['CAM_RIGHT'],
    camera_matrices["CAM_LEFT"], distortion_coefficients["CAM_LEFT"],
    camera_matrices["CAM_RIGHT"], distortion_coefficients["CAM_RIGHT"],
    (widthL, heightL), criteria = stereocalib_criteria, flags = stereocalib_flags)

if rms > 1.0:
    print(f"[ERROR] Fundamental Matrix reproj. RMS error < 1.0 {rms}. Re-try image capture.")
else:
    print(f"[OK] Fundamental Matrix onbtention successful RMS error= {rms}")

np.savez("../COMMON/Calibrated_Parameters/Stereo_Calibrate_rms_KL_DL_KR_DR_R_T_E_F.npz",\
         rms=rms, KL=camera_matrices["CAM_LEFT"],  DL=distortion_coefficients["CAM_LEFT"], \
         KR=camera_matrices["CAM_RIGHT"],  DR=distortion_coefficients["CAM_RIGHT"], \
         R=R, T=T, E=E, F=F)

# DRAW SOME EPILINES ON THE CHESS BOARDS FOR FUN using the Fundamental Matrix ##########
print(f"\n\DRAWING EPILINES FOR SANITY CHECK #######################################\n")
def drawlines(img1src, img2src, lines, pts1src, pts2src):
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

for i, (file_nameL, file_nameR) in enumerate(zip(different_chess_viewsL, different_chess_viewsR)):
    full_imageL = cv2.imread(file_nameL, cv2.IMREAD_GRAYSCALE)
    full_imageR = cv2.imread(file_nameR, cv2.IMREAD_GRAYSCALE)
    print(file_nameL)
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines_L = cv2.computeCorrespondEpilines(
        image_points['CAM_RIGHT'][i].reshape(-1, 1, 2), 2, F)
    lines_L = lines_L.reshape(-1, 3)
    img5, img6 = drawlines(full_imageL, full_imageR, lines_L, image_points['CAM_LEFT'][i], image_points['CAM_RIGHT'][i])

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines_R = cv2.computeCorrespondEpilines(
        image_points['CAM_LEFT'][i].reshape(-1, 1, 2), 1, F)
    lines_R = lines_R.reshape(-1, 3)
    img3, img4 = drawlines(full_imageR, full_imageL, lines_R, image_points['CAM_RIGHT'][i], image_points['CAM_LEFT'][i])

    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.suptitle("Epilines in both images")
    plt.savefig(f"../COMMON/Epilines_Chess_Samples/Epilines_{(i+1):06}.png", dpi=400)
    plt.clf()

# OBTAIN STERO RECTIFICATION HOMOGRAPHIES #####################################
print(f"\n\nOBTAINING STERO RECTIFICATION HOMOGRAPHIES ###########################\n")
'''
R1	Output 3x3 rectification transform (rotation matrix) for the first camera. This matrix brings points given in the unrectified first camera's coordinate system to points in the rectified first camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified first camera's coordinate system to the rectified first camera's coordinate system.
R2	Output 3x3 rectification transform (rotation matrix) for the second camera. This matrix brings points given in the unrectified second camera's coordinate system to points in the rectified second camera's coordinate system. In more technical terms, it performs a change of basis from the unrectified second camera's coordinate system to the rectified second camera's coordinate system.
P1	Output 3x4 projection matrix in the new (rectified) coordinate systems for the first camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified first camera's image.
P2	Output 3x4 projection matrix in the new (rectified) coordinate systems for the second camera, i.e. it projects points given in the rectified first camera coordinate system into the rectified second camera's image.
Q	Output 4??4 disparity-to-depth mapping matrix (see reprojectImageTo3D).
flags	Operation flags that may be zero or CALIB_ZERO_DISPARITY . If the flag is set, the function makes the principal points of each camera have the same pixel coordinates in the rectified views. And if the flag is not set, the function may still shift the images in the horizontal or vertical direction (depending on the orientation of epipolar lines) to maximize the useful image area.
alpha	Free scaling parameter. If it is -1 or absent, the function performs the default scaling. Otherwise, the parameter should be between 0 and 1. alpha=0 means that the rectified images are zoomed and shifted so that only valid pixels are visible (no black areas after rectification). alpha=1 means that the rectified image is decimated and shifted so that all the pixels from the original images from the cameras are retained in the rectified images (no source image pixels are lost). Any intermediate value yields an intermediate result between those two extreme cases.
newImageSize	New image resolution after rectification. The same size should be passed to initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0) is passed (default), it is set to the original imageSize . Setting it to a larger value can help you preserve details in the original image, especially when there is a big radial distortion.
validPixROI1	Optional output rectangles inside the rectified images where all the pixels are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below).
validPixROI2	Optional output rectangles inside the rectified images where all the pixels are valid. If alpha=0 , the ROIs cover the whole images. Otherwise, they are likely to be smaller (see the picture below).
'''
R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        camera_matrices["CAM_LEFT"],  distortion_coefficients["CAM_LEFT"], \
        camera_matrices["CAM_RIGHT"],  distortion_coefficients["CAM_RIGHT"],\
        (widthL, heightL), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.9)

np.savez("../COMMON/Calibrated_Parameters/Stereo_Rectify_R1_R2_P1_P2_Q_roi_left_roi_right.npz",
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q, roi_left=roi_left, roi_right=roi_right)

"""
# This is replaced because my results were always bad. Estimates are
# taken from the OpenCV samples.
width, height = self.image_size
focal_length = 0.8 * width
Q = np.float32([[1, 0, 0, -0.5 * width],
                                      [0, -1, 0, 0.5 * height],
                                      [0, 0, 0, -focal_length],
                                      [0, 0, 1, 0]])
"""

# DO THE UNDISTORTION BETTER WITH ALL THIS #####################################
print(f"RECTIFING IMAGES AND UNDISTORTING THEM ################################")
# We proceed to rectify and undistort all the images, so they can be processed on the disparity map generator
# https://docs.opencv.org/4.1.1/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
# OJO! MAYBE YOU SHOUL DUSE GREYSCALE IMAGES IN THE RECTIFICATION!!!
fig, axes = plt.subplots(1, 2, figsize=(15, 10))
fig.suptitle("Rectified images should have epilines aligned")

mapLx, mapLy	=	cv2.initUndistortRectifyMap( camera_matrices["CAM_LEFT"],
                distortion_coefficients["CAM_LEFT"], R1, P1, (widthL, heightL), cv2.CV_32FC1)

mapRx, mapRy	=	cv2.initUndistortRectifyMap( camera_matrices["CAM_RIGHT"],
                distortion_coefficients["CAM_RIGHT"], R2, P2, (widthR, heightR), cv2.CV_32FC1)

for i, (file_nameL, file_nameR) in enumerate(zip(different_chess_viewsL, different_chess_viewsR)):
    print(i)
    img_L = cv2.imread(file_nameL)
    img_R = cv2.imread(file_nameR)

    rectified_imageL = cv2.remap(img_L, mapLx, mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    rectified_imageR = cv2.remap(img_R, mapRx, mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)

    cv2.imwrite(f'../CAM_LEFT/Rectified_Chess_Samples/Rectified_{(i+1):06}.png', rectified_imageL)
    cv2.imwrite(f'../CAM_RIGHT/Rectified_Chess_Samples/Rectified_{(i+1):06}.png', rectified_imageR)

    # For sanity check we will plot the rectified images with some horizontal lines which should match the epilines of both images
    rectified_imageL=cv2.cvtColor(rectified_imageL, cv2.COLOR_BGR2GRAY)
    rectified_imageR=cv2.cvtColor(rectified_imageR, cv2.COLOR_BGR2GRAY)
    axes[0].imshow(rectified_imageL, cmap='gray')
    axes[1].imshow(rectified_imageR, cmap='gray')

    for j in range(5,475, 20):
        axes[0].axhline(j)
        axes[1].axhline(j)

    fig.savefig(f"../COMMON/Epilines_Chess_Samples/Rectified_Epilines_{(i+1):06}.png", dpi=400)

    axes[0].clear()
    axes[1].clear()


# OBTAIN THE DISPARITY MAP AND THE DEPTH MAP! ###################################
print(f"\n\nOBTAINING STERO DISPARITY AND DEPTH MAPS FOR CHESS VIEWS!!###########\n")

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

stereo_left_matcher = cv2.StereoSGBM_create(
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

stereo_right_matcher = cv2.ximgproc.createRightMatcher(stereo_left_matcher)
# FILTER Parameters
lmbda = 80000
sigma = 1.3
visual_multiplier = 6

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=stereo_left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)


different_chess_views_rectified_R=sorted(glob.glob("../CAM_RIGHT/Rectified_Chess_Samples/*.png"))
different_chess_views_rectified_L=sorted(glob.glob("../CAM_LEFT/Rectified_Chess_Samples/*.png"))
def normalize_and_plot_disparity_map(disparity, tag, i):
    disparity_plot = cv2.normalize(disparity, disparity, alpha=255,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
    disparity_plot = np.uint8(disparity_plot)
    cv2.imwrite(f"../COMMON/Disparities_Chess_Samples/Normalized_{tag}_disparity_{(i+1):06}.png", disparity_plot)

for i, (file_nameL, file_nameR) in enumerate(zip(different_chess_views_rectified_L, different_chess_views_rectified_R)):
    rectified_imageL = cv2.imread(file_nameL)
    rectified_imageR = cv2.imread(file_nameR)
    grayL = cv2.cvtColor(rectified_imageL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectified_imageR, cv2.COLOR_BGR2GRAY)

    # Compute DISPARITY MAPS from L and R
    disparity_L = stereo_left_matcher.compute(grayL, grayR)  # .astype(np.float32)/16
    disparity_R = stereo_right_matcher.compute(grayR, grayL)  # .astype(np.float32)/16
    disparity_L = np.int16(disparity_L)
    disparity_R = np.int16(disparity_R)
    # Use both to generate a filtered disparity map for the left image
    filtered_disparity = wls_filter.filter(disparity_L, grayL, None, disparity_R)  # important to put "imgL" here!!! Maybe can use the colored image here!


    # Normalize the values to a range from 0..255 for a grayscale image of the disparity map
    normalize_and_plot_disparity_map(disparity_L, "L", i)
    normalize_and_plot_disparity_map(disparity_R, "R", i)
    normalize_and_plot_disparity_map(filtered_disparity, "Filtered", i)

    # Obtain the DEPTH MAP
    # According to the documentation we should do this division and conversion
    filtered_disparity = (filtered_disparity/16.0).astype(np.float32)

    # And voila, we get an array [h,w,3] with the 3D coordinates (in the units of the chess world points we inputed)
    # of each pixel in the image of the Left! Thus we chose the Left camera to be better
    image_3D = cv2.reprojectImageTo3D(filtered_disparity, Q, handleMissingValues=True)
    # handleMissingValues	Indicates, whether the function should handle missing values (i.e. points where the disparity was not computed). If handleMissingValues=true, then pixels with the minimal disparity that corresponds to the outliers (see StereoMatcher::compute ) are transformed to 3D points with a very large Z value (currently set to 10000)
    print(image_3D.shape)
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

# NOW WE WILL RECORD A TEST FILM TO CHECK THE CORRECTNESS OF THE MAPS

def normalize_disparity_map(disparity):
    disparity = cv2.normalize(disparity, disparity, alpha=255,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
    return np.uint8(disparity)

input("\n\nPress ENTER to take the Live Test photos:")
num_photos=15
for j in range(num_photos):
    print(f"\n\nTAKING PHOTOS {j}/{num_photos} ########################")
    # instead of using .read() to get an image we decompose it into .grab and then .retrieve
    # so we can maximize the sinchronization

    if not (vidStreamR.grab() and vidStreamL.grab()):
        print("[Error] Getting the image for this iteration. Retrying...")
        continue
    _, img_R = vidStreamR.retrieve()
    _, img_L = vidStreamL.retrieve()


    # RECTIFY THE IMAGE
    rect_img_L = cv2.remap(img_L, mapLx, mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    rect_img_R = cv2.remap(img_R, mapRx, mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    result = np.concatenate((rect_img_L,rect_img_R), axis=1)
    cv2.imwrite(f"../COMMON/Live_Test/Live_{j}_color.png", result)

    # GRAYSCALE THE IMAGES
    grayL = cv2.cvtColor(rect_img_L, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rect_img_R, cv2.COLOR_BGR2GRAY)

    # COMPUTE DISPARITIES

    disparity_L = stereo_left_matcher.compute(grayL, grayR)  # .astype(np.float32)/16
    disparity_R = stereo_right_matcher.compute(grayR, grayL)  # .astype(np.float32)/16
    disparity_L = np.int16(disparity_L)
    disparity_R = np.int16(disparity_R)
    filtered_disparity = wls_filter.filter(disparity_L, grayL, None, disparity_R)  # important to put "imgL" here!!! Maybe can use the colored image here!

    total_unfiltered = np.concatenate((normalize_disparity_map(disparity_L), normalize_disparity_map(disparity_R)), axis=1)
    total_filtered = np.concatenate( (normalize_disparity_map(filtered_disparity), np.zeros(filtered_disparity.shape)), axis=1 )
    joint_images = np.concatenate((total_unfiltered, total_filtered), axis=0)
    cv2.imwrite(f"../COMMON/Live_Test/Live_{j}.png", joint_images)
    cv2.imshow(f'Live Test {j}', result)
    ok = cv2.waitKey(2000) #########
    cv2.destroyAllWindows()

'''
    x, y, w, h = roiL
    undistortedL = undistortedL[y:y+h, x:x+w]
    x, y, w, h = roiR
    undistortedR = undistortedR[y:y+h, x:x+w]
'''


'''
Point cloud class generated from stereo image pairs.
Classes:
    * ``PointCloud`` - Point cloud with RGB colors
.. image:: classes_point_cloud.svg
'''



class PointCloud(object):

    """3D point cloud generated from a stereo image pair."""

    #: Header for exporting point cloud to PLY
    ply_header = (
'''ply
format ascii 1.0
element vertex {vertex_count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
''')

    def __init__(self, coordinates, colors):
        """
        Initialize point cloud with given coordinates and associated colors.
        ``coordinates`` and ``colors`` should be numpy arrays of the same
        length, in which ``coordinates`` is made of three-dimensional point
        positions (X, Y, Z) and ``colors`` is made of three-dimensional spectral
        data, e.g. (R, G, B).
        """
        self.coordinates = coordinates.reshape(-1, 3)
        self.colors = colors.reshape(-1, 3)

    def write_ply(self, output_file):
        """Export ``PointCloud`` to PLY file for viewing in MeshLab."""
        points = np.hstack([self.coordinates, self.colors])
        with open(output_file, 'w') as outfile:
            outfile.write(self.ply_header.format(
                                            vertex_count=len(self.coordinates)))
            np.savetxt(outfile, points, '%f %f %f %d %d %d')

    def filter_infinity(self):
        """Filter infinite distances from ``PointCloud.``"""
        mask = self.coordinates[:, 2] > self.coordinates[:, 2].min()
        coords = self.coordinates[mask]
        colors = self.colors[mask]
        return PointCloud(coords, colors)
