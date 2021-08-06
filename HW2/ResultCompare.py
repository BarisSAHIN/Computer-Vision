from __future__ import print_function
import cv2 as cv

img1 = cv.imread('Barn2/im2.ppm', 0)
dispartyIm = cv.imread("Barn2/disp2.pgm", 0)

height1, width1 = img1.shape

imcompare = cv.imread("dispartyMapWithNormalization.png",0)
print(imcompare.shape)
total = 0
count= 0
for i in range(height1):
    for j in range(width1):
        print(dispartyIm[i, j])
        print(imcompare[i,j])
        total += abs(dispartyIm[i, j] - imcompare[i,j])
        count += 1

print(count)
print("Avg: " , total/count)