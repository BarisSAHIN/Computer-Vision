import cv2
import numpy as np


class Detector:
    def __init__(self):
        self.line_y = 810
        self.line_x_left = 585
        self.line_x_right = 800
        self.detec = []
        self.rect_min_width = 80
        self.rect_min_height = 80
        self.offset = 4
        self.subtracted = None
        self.video = None
    def backgroundExtract(self,videoName):
        self.video = cv2.VideoCapture(videoName)
        self.subtracted = cv2.createBackgroundSubtractorKNN()
    def mainloop(self):
        car_count= 0
        while True:
            ret, frame = self.video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.blur(gray, (5, 5))
            gray_blur_foreground = self.subtracted.apply(gray_blur)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morphed = cv2.morphologyEx(cv2.dilate(gray_blur_foreground, np.ones((5, 5))), cv2.MORPH_CLOSE, kernel)
            contour, hier = cv2.findContours(morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            cv2.line(frame, (self.line_x_left, self.line_y), (self.line_x_right,self.line_y), (0, 0, 255), 5)
            for (i, c) in enumerate(contour):
                (x, y, w, h) = cv2.boundingRect(c)
                if w >= self.rect_min_width and h >= self.rect_min_height:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    center = (x + int(w / 2), y + int(h / 2))
                    self.detec.append(center)

                    for (x, y) in self.detec:
                        if x < self.line_x_right and x > self.line_x_left and y < (self.line_y + self.offset) and y > (self.line_y - self.offset):
                            car_count += 1
                            self.detec.remove((x, y))

            text = str(car_count) + " car"
            if car_count > 1:
                text += "s"
            cv2.putText(frame, text, (960, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow("baris", frame)
            if cv2.waitKey(1) == 27:
                break
        self.video.release()
        cv2.destroyAllWindows()

hwDetector = Detector()
hwDetector.backgroundExtract('videoplayback.mp4')
hwDetector.mainloop()
