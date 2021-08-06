from __future__ import print_function
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


img1 = cv.imread('Barn2/im2.ppm', 0)
img2 = cv.imread('Barn2/im6.ppm', 0)
dispartyIm = cv.imread("Barn2/disp2.pgm", 0)

print(img1.shape)

ORBDetector = cv.ORB_create()
keypoints1 = ORBDetector.detect(img1,None)
keypoints1, descriptors1 = ORBDetector.compute(img1, keypoints1)


keypoints2 = ORBDetector.detect(img2,None)
keypoints2, descriptors2 = ORBDetector.compute(img2, keypoints2)


FLANN_INDEX_LSH= 6

index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

keyPointMatcher = cv.FlannBasedMatcher(index_params, {})
matchedPoints = keyPointMatcher.knnMatch(descriptors1, descriptors2, 2)


firstKeyPoints = []
secondKeyPoints = []
firstSpecialKeyPoints = []
secondSpecialKeyPoints = []
for i,(m,n) in enumerate(matchedPoints):
    if m.distance < 0.7 * n.distance:
        secondSpecialKeyPoints.append(keypoints2[m.trainIdx].pt)
        firstSpecialKeyPoints.append(keypoints1[m.queryIdx].pt)
    secondKeyPoints.append(keypoints2[m.trainIdx].pt)
    firstKeyPoints.append(keypoints1[m.queryIdx].pt)

firstKeyPoints = np.int32(firstKeyPoints)
secondKeyPoints = np.int32(secondKeyPoints)
firstSpecialKeyPoints = np.int32(firstKeyPoints)
secondSpecialKeyPoints = np.int32(secondKeyPoints)

counter = 0
total = 0

print("Feature Selection: Founded Disparity ------- Ground Truth")
for i in range(len(firstKeyPoints)):
    distance = (firstKeyPoints[i][0]-secondKeyPoints[i][0])
    trueVal = dispartyIm[firstKeyPoints[i][1],secondKeyPoints[i][0]]/8
    total = total + abs(distance-trueVal)
    print(distance," ---- ",trueVal," Difference: ",distance-trueVal)
    counter += 1
print("Avg: " ,total/counter)
print("Number of points: ",counter)
F1, mask1 = cv.findFundamentalMat(firstSpecialKeyPoints,secondSpecialKeyPoints,cv.FM_LMEDS)

print("Fundamental Matrix: " , F1)

zeroIndex = []
print("Epipolar Correspondences: Founded Disparity ------- Ground Truth")
for i in range(len(firstKeyPoints)):
    u1 = firstKeyPoints[i][0]
    v1 = firstKeyPoints[i][1]
    u2 = secondKeyPoints[i][0]
    v2 = secondKeyPoints[i][1]
    Ar1=[u1*u2,v2*u1,u1,u2*v1,v1*v2,v1,u2,v2,1]
    Ar2=[[F1[0][0]],[F1[0][1]],[F1[0][2]],[F1[1][0]],[F1[1][1]],[F1[1][2]],[F1[2][0]],[F1[2][1]],[F1[2][2]]]
    if((np.matmul(Ar1,Ar2) == 0)):
        zeroIndex.append(i)

counter = 0
total = 0
for i in zeroIndex:
    distance = (firstKeyPoints[i][0] - secondKeyPoints[i][0])
    trueVal = dispartyIm[firstKeyPoints[i][1], secondKeyPoints[i][0]] / 8
    total = total + abs(distance - trueVal)
    print(distance, " ---- ", trueVal, " Difference: ", distance - trueVal)
    counter += 1
print("Avg: ", total / counter)
print("Number of points: ", counter)
############################################## Corseponce Found Ends ###################################################

firstKeyPoints = firstKeyPoints[mask1.ravel() == 1]
secondKeyPoints = secondKeyPoints[mask1.ravel() == 1]

height1, width1 = img1.shape
height2, width2 = img2.shape

_, H1, H2 = cv.stereoRectifyUncalibrated(
    np.float32(firstKeyPoints), np.float32(secondKeyPoints), F1, imgSize=(width1, height1)
)
img1_rectified = cv.warpPerspective(img1, H1, (width1, height1))
img2_rectified = cv.warpPerspective(img2, H2, (width2, height2))
cv.imwrite("img1_rectified.png", img1_rectified)
cv.imwrite("img2_rectified.png", img2_rectified)


block_size = 1
min_disp = -20
max_disp = 20
num_disp = max_disp - min_disp
uniquenessRatio = 10
speckleWindowSize = 200
speckleRange = 2
disp12MaxDiff = 0

stereo = cv.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
)
disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)
disparity_SGBM = cv.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                              beta=0, norm_type=cv.NORM_MINMAX)
disparity_SGBM = np.uint8(disparity_SGBM)
cv.imwrite("dispartyMapWithNormalization.png", disparity_SGBM)




