import numpy as np 
import cv2
import os
from scipy.linalg import svd,lstsq


flag = 0
a =[]
b = []

def on_click1(event, x, y, p1, p2):
        
        if event == cv2.EVENT_LBUTTONDOWN:

                cv2.circle(inputImg, (x, y), 3, (0, 0, 255), -1)
                a.append([x,y])
                cv2.imshow('inputImg',inputImg)
                
def findInfPoint():
        hm1 = b[0]
        hm2 = b[1]
        hm3 = b[2]
        hm4 = b[3]

        hm1.append(1)
        hm2.append(1)
        hm3.append(1)
        hm4.append(1)

        cp1 = np.cross(hm1,hm2)
        cp2 = np.cross(hm3,hm4)

        print(np.cross(cp1,cp2))
        HT = np.transpose(homographyCV())
        print(cp1,'--',cp2)
        print(np.matmul(HT,cp1))
        print(np.matmul(HT,cp2))
        
def homographyCV():
        c = np.array(a)
        d = np.array(b)
        h,status = cv2.findHomography(c,d)

        print('OpenCV homography matrix:\n',h)
        return h


def homographyMy(modelImg):
        
        x_1 = [a[0][0],b[0][0]]
        y_1 = [a[0][1],b[0][1]]
        x_2 = [a[1][0],b[1][0]]
        y_2 = [a[1][1],b[1][1]]
        x_3 = [a[2][0],b[2][0]]
        y_3 = [a[2][1],b[2][1]]
        x_4 = [a[3][0],b[3][0]]
        y_4 = [a[3][1],b[3][1]]

        PH = np.array([
            [-x_1[0], -y_1[0], -1, 0, 0, 0, x_1[0]*x_1[1], y_1[0]*x_1[1], x_1[1]],
            [0, 0, 0, -x_1[0], -y_1[0], -1, x_1[0]*y_1[1], y_1[0]*y_1[1], y_1[1]],
            [-x_2[0], -y_2[0], -1, 0, 0, 0, x_2[0]*x_2[1], y_2[0]*x_2[1], x_2[1]],
            [0, 0, 0, -x_2[0], -y_2[0], -1, x_2[0]*y_2[1], y_2[0]*y_2[1], y_2[1]],
            [-x_3[0], -y_3[0], -1, 0, 0, 0, x_3[0]*x_3[1], y_3[0]*x_3[1], x_3[1]],
            [0, 0, 0, -x_3[0], -y_3[0], -1, x_3[0]*y_3[1], y_3[0]*y_3[1], y_3[1]],
            [-x_4[0], -y_4[0], -1, 0, 0, 0, x_4[0]*x_4[1], y_4[0]*x_4[1], x_4[1]],
            [0, 0, 0, -x_4[0], -y_4[0], -1, x_4[0]*y_4[1], y_4[0]*y_4[1], y_4[1]],
            ])
                
        
        U,s,Vt = svd(PH)
        del U,s
        h = Vt[-1]
        H = h.reshape(3,3)
        
        print('My homography matrix:\n',H)
        return H

print(os.getcwd())
inputImg = cv2.imread('photo3.jpg')
modelImg = cv2.imread('photo.jpg')
height, width, channels = modelImg.shape
b.append([0,0])
b.append([width,0])
b.append([0,height])
b.append([width,height])
cv2.imshow("inputImg", inputImg)
cv2.namedWindow('inputImg')
cv2.setMouseCallback('inputImg', on_click1)

for i in range(4):
        pressedkey=cv2.waitKey(0)
        if pressedkey==27:
                copy = cv2.imread('photo3.jpg')
                resultImg = cv2.warpPerspective(copy, homographyMy(modelImg), (1000, 1000))
                resultImg2 = cv2.warpPerspective(copy, homographyCV(), (1000, 1000))
                cv2.imshow('MyresultImg',resultImg)
                cv2.imshow('CVresultImg',resultImg2)
                pressedkey=cv2.waitKey(0)
                if pressedkey==27:
                        cv2.destroyAllWindows()
                        break

