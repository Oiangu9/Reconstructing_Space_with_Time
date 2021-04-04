import numpy as np
import cv2
import glob
import os
import shutil
import matplotlib.pyplot as plt

num_photos = 30

try:
    os.mkdir(f"../COMMON/ExtraPhotos/")
except OSError as error:
    pass

# The left camera image will be the final deoth map so better the good resolution one
vidStreamL = cv2.VideoCapture(2)  # index of OBS - Nirie
vidStreamR = cv2.VideoCapture(1)  # index of Droidcam camera - Izeko

# Change the resolution if needed
vidStreamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
vidStreamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

vidStreamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # float
vidStreamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # float

# We import the Camera and Distortion Matrices together with the fundamental matrix etc.

stereoCalibrators = np.load("../COMMON/Calibrated_Parameters/Stereo_Calibrate_rms_K1_D1_K2_D2_R_T_E_F.npz")
rms, KL, DL, KR, DR, R, T, E, F= stereoCalibrators['rms'], stereoCalibrators['K1'],\
    stereoCalibrators['D1'], stereoCalibrators['K2'], stereoCalibrators['D2'], \
    stereoCalibrators['R'], stereoCalibrators['T'], stereoCalibrators['E'], stereoCalibrators['F']

stereoRectificators = np.load("../COMMON/Calibrated_Parameters/Stereo_Rectify_R1_R2_P1_P2_Q_roi_left_roi_right.npz")
R1,R2, P1, P2, Q, roi_left, roi_right = stereoRectificators['R1'], stereoRectificators['R2'],\ 
    stereoRectificators['P1'], stereoRectificators['P2'], stereoRectificators['Q'], \
    stereoRectificators['roi_left'], stereoRectificators['roi_right']
    
# A sequence of 30 photos will be taken and processed
mapLx, mapLy = cv2.initUndistortRectifyMap( KL, DL, R1, P1, (640, 480), cv2.CV_32FC1)

mapRx, mapRy = cv2.initUndistortRectifyMap( KR, DR, R2, P2, (640, 480), cv2.CV_32FC1) 

    
def normalize_disparity_map(disparity):
    disparity = cv2.normalize(disparity, disparity, alpha=255,
                                  beta=0, norm_type=cv2.NORM_MINMAX)
    return np.uint8(disparity)

input("\n\nPress ENTER to take the Life Test photos:")

for j in range(num_photos):
    print(f"\n\nTAKING PHOTOS {j}/{num_photos} ########################")
    # instead of using .read() to get an image we decompose it into .grab and then .retrieve
    # so we can maximize the sinchronization
    
    if not (vidStreamL.grab() and vidStreamR.grab()):
        print("[Error] Getting the image for this iteration. Retrying...")
        continue
    _, img_L = vidStreamL.retrieve()
    _, img_R = vidStreamR.retrieve()
    
    
    # RECTIFY THE IMAGE
    rect_img_L = cv2.remap(img_L, mapLx, mapLy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    rect_img_R = cv2.remap(img_R, mapRx, mapRy, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT)
    result = np.concatenate((rect_img_L,rect_img_R), axis=1)
    cv2.imwrite(f"../COMMON/ExtraPhotos/Extra_{j}_color.png", result)
    
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
    cv2.imwrite(f"../COMMON/Life_Test/Life_{j}.png", joint_images)
    cv2.imshow(f'Life Test {j}', result)
    ok = cv2.waitKey(2000) #########
    cv2.destroyAllWindows()
    