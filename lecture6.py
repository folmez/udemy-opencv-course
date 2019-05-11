import numpy as np
import cv2
import sys
import time

subsection = int(sys.argv[1])

if subsection==51:
    body_classifier = cv2.CascadeClassifier('course_files/Haarcascades/haarcascade_fullbody.xml')
    cap = cv2.VideoCapture('course_files/images/walking.avi')

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1.0, fy=1.0, interpolation = cv2.INTER_LINEAR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        bodies = body_classifier.detectMultiScale(gray, 1.2, 5)
        for (x,y,w,h) in bodies:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 2)
            cv2.imshow('Pedestrians', frame)
        time.sleep(.05)
        if cv2.waitKey(1) == 13: # enter key
            break
    cap.release()
    cv2.destroyAllWindows()

    car_classifier = cv2.CascadeClassifier('course_files/Haarcascades/haarcascade_car.xml')
    cap = cv2.VideoCapture('course_files/images/cars.avi')

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation = cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cars = car_classifier.detectMultiScale(gray, 1.2, 5)
        for (x,y,w,h) in cars:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 255), 2)
            cv2.imshow('cars', frame)
        time.sleep(.10)
        if cv2.waitKey(1) == 13: # enter key
            break
    cap.release()
    cv2.destroyAllWindows()

elif subsection==50:
    face_classifier = cv2.CascadeClassifier('course_files/Haarcascades/haarcascade_frontalface_default.xml')
    eye_classifier = cv2.CascadeClassifier('course_files/Haarcascades/haarcascade_eye.xml')

    image = cv2.imread('course_files/images/Trump.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces)==0:
        print('No faces have been found')
    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x,y), (x+w,y+h), (127,0,255), 2)
            cv2.imshow('Face detection', image)
            cv2.waitKey()

            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_classifier.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,255,0), 2)
                cv2.imshow('Eyes', image)
                cv2.waitKey()

    cv2.destroyAllWindows()

    # Live video face-eye detection
    def face_detector(image, size=0.5):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if len(faces)==0:
            return image
        else:
            for (x,y,w,h) in faces:
                x, w, y, h = x-50, w+50, y-50, h+50
                cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)

                roi_gray = gray[y:y+h, x:x+w]
                roi_color = image[y:y+h, x:x+w]
                eyes = eye_classifier.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
                    cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,0,255), 2)

            roi_color = cv2.flip(roi_color, 1)
            return roi_color

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('My face and eyes', face_detector(frame))
        if cv2.waitKey(1) == 13: # enter key
            break
    cap.release()
    cv2.destroyAllWindows()
