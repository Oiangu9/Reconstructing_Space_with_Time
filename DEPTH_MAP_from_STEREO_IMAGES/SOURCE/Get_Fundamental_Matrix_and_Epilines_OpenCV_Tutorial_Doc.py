import cv2
import numpy as np
from matplotlib import pyplot as plt



img_L = cv2.imread('../CAM_LEFT/Life_Images/left_img.png', cv2.IMREAD_GRAYSCALE)  
img_R = cv2.imread('../CAM_RIGHT/Life_Images/right_img.png', cv2.IMREAD_GRAYSCALE)

# LOOK FOR KEYPOINTS ####################################
sift = cv2.SIFT_create() # The keypoint detector algorithm

# find the keypoints and descriptors with SIFT
keypoints_L, kp_descriptors_L = sift.detectAndCompute(img_L, None)
keypoints_R, kp_descriptors_R = sift.detectAndCompute(img_R, None)

# Visualize keypoints
imgSift = cv2.drawKeypoints(
    img_L, keypoints_L, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("SIFT Keypoints", imgSift)
cv2.waitKey(500)

cv2.destroyAllWindows()

# Using the FLANN Nearest Nioghtbourgh approach look for keypoint matches
# If we had that the two cameras are always fixed relative to each other, then
# it would be better to use points selected at hand maybe because if fixed only one
# fundamental matrix would be required!!

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(kp_descriptors_L,kp_descriptors_R,k=2)

# Keep good matches: calculate distinctive image features
# Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
# https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf

good_kps = []
pts_L = []
pts_R = []
matchesMask = [[0, 0] for i in range(len(matches))] # just to plot the good keypoints

for i,(m,n) in enumerate(matches):
    if m.distance < 0.8*n.distance: # Then save these that have got good enough scores
        matchesMask[i] = [1, 0] # this is just to plot them
        good_kps.append(m)
        pts_R.append(keypoints_R[m.trainIdx].pt)
        pts_L.append(keypoints_L[m.queryIdx].pt)

# Draw the keypoint matches between both pictures
# Still based on: https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
draw_params = dict(matchColor=(0, 255, 0),
                   singlePointColor=(255, 0, 0),
                   matchesMask=matchesMask[300:500], # we will still only plot 200 of the good enough keypts
                   flags=cv2.DrawMatchesFlags_DEFAULT)

keypoint_matches = cv2.drawMatchesKnn(
    img_L, keypoints_L, img_R, keypoints_R, matches[300:500], None, **draw_params)
cv2.imshow("Keypoint matches", keypoint_matches)
cv2.waitKey(500)
cv2.destroyAllWindows()

# Now we have the list of best matches from both the images. Let's find the Fundamental Matrix.

# FIND FUNDAMENTAL MATRIX! ####################################################
pts_L = np.int32(pts_L) # Pixel coordinates of the corresponding keypoints between images
pts_R = np.int32(pts_R)

'''
 OpenCV includes a function that calculates the fundamental matrix based on the 
 matched keypoint pairs. It needs at least 7 pairs but works best with 8 or more. 
 We have more than enough matches. This is where the RanSaC method (Random Sample Consensus) 
 works well. RANSAC also considers that not all matched features are reliable. 
 It takes a random set of point correspondences, uses these to compute the fundamental
 matrix and then checks how well it performs. When doing this for different random sets 
 (usually, 8-12), the algorithm chooses its best estimate. According to OpenCV’s source code,
 you should have at least fifteen feature pairs to give the algorithm enough data.
'''
# inliers is a mask to select the not outliers que se han cribado con el algoritmo
Fundamental_Matrix, inliers = cv2.findFundamentalMat(pts_L, pts_R, cv2.FM_RANSAC) # you can also use cv2.FM_LMEDS instead of RANSAC
# It is computed based on the fact that it should happen that
# [pts_L;1]^T * F * [pts_R;1] = 0 
# We select only inlier points
pts_R = pts_R[inliers.ravel()==1]
pts_L = pts_L[inliers.ravel()==1]

np.save(f"../COMMON/Fundamental_Matrix.npy", Fundamental_Matrix)

# DRAW EPILINES ############################################################
# Visualize epilines
# Adapted from: https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
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
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    return img1color, img2color


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines_L = cv2.computeCorrespondEpilines(
    pts_R.reshape(-1, 1, 2), 2, Fundamental_Matrix)
lines_L = lines_L.reshape(-1, 3)
img5, img6 = drawlines(img_L, img_R, lines_L, pts_L, pts_R)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
lines_R = cv2.computeCorrespondEpilines(
    pts_L.reshape(-1, 1, 2), 1, Fundamental_Matrix)
lines_R = lines_R.reshape(-1, 3)
img3, img4 = drawlines(img_R, img_L, lines_R, pts_R, pts_L)

plt.subplot(121), plt.imshow(img6)
plt.subplot(122), plt.imshow(img4)
plt.suptitle("Epilines in both images")
plt.savefig("../COMMON/Epilines.png", dpi=400)


# If the images were not undistorted, one could find the fundamental matrix using those params


# STEREO RECTIFICATION ######################################################
# Rectification is essentially achieveing that corresponding  epilines that 
# itnersect in the same point p, are parallel and at the same height in the image
# Stereo rectification (uncalibrated variant)
# Adapted from: https://stackoverflow.com/a/62607343
h1, w1 = img1.shape
h2, w2 = img2.shape

# We get the homographies HL and HR that map the image points to the rectified images
_, H1, H2 = cv.stereoRectifyUncalibrated(
    np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
)
