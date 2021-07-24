import cv2
import numpy as np
import time
from datetime import datetime

class Detector:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    def faces(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cap.release()

    def eyes(self):
        cap = cv2.VideoCapture(0)
        while True:
            _, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            eyes = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in eyes:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow('img', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cap.release()
    def selfie(self):
        cap = cv2.VideoCapture(0)
        while True:
            _,frame = cap.read()
            original_frame = frame.copy()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face = self.face_cascade.detectMultiScale(gray,1.3,5)
            for x,y,w,h in face:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
                face_roi = frame[y:y+h,x:x+w]
                gray_roi = gray[y:y+h,x:x+w]
                smile = self.smile_cascade.detectMultiScale(gray_roi,1.3,25)
                for x1,y1,w1,h1 in smile:
                    cv2.rectangle(face_roi,(x,y),(x+w,y+h),(0,255,255),2)
                    time_stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    file_name = f'selfie-{time_stamp}.png'
                    cv2.imwrite(file_name,original_frame)
            cv2.imshow('cam star',frame)
            if cv2.waitKey(10)==ord('q'):
                break

if __name__ == '__main__':
    Detect = Detector()
    Detect.selfie()