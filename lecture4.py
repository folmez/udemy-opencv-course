import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

subsection = int(sys.argv[1])

if subsection==40:
    image = cv2.imread('course_files/images/Sunflowers.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow('original', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    detector = cv2.SimpleBlobDetector_create()
    keypoints = detector.detect(image)
    blank = np.zeros((1,1))
    blobs = cv2.drawKeypoints(image, keypoints, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    cv2.imshow('Blobs', blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif subsection==37:
    image = cv2.imread('course_files/images/soduku.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 170, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi/180, 240)

    for rho, theta in lines[0]:
        print(theta)
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a*rho, b*rho
        x1, y1 = int(x0+1000*(-b)), int(y0+1000*a)
        x2, y2 = int(x0-1000*(-b)), int(y0-1000*a)
        cv2.line(image, (x1, y1), (x2, y2), (255,0,0), 2)

    cv2.imshow('Hough lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif subsection==36:
    template = cv2.imread('course_files/images/4star.jpg', 0)
    cv2.imshow('template', template)
    cv2.waitKey(0)

    target = cv2.imread('course_files/images/shapestomatch.jpg')
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    ret, thresh1 = cv2.threshold(template, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    template_contour = sorted_contours[1]

    ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        match = cv2.matchShapes(template_contour, c, 1, 0.0)
        if match < 0.15:
            closest_contour = c
        else:
            closest_contour = []

    cv2.drawContours(target, [closest_contour], -1, (0,255,0), 3)
    cv2.imshow('output', target)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif subsection==35:
    if False:
        image = cv2.imread('course_files/images/house.jpg')
        orig_image = image.copy()

        cv2.imshow("original image", orig_image)
        cv2.waitKey(0)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(orig_image, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.imshow("bounding rectangle", orig_image)

        for c in contours:
            accuracy = 0.03 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, accuracy, True)
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)
            cv2.imshow('Approx Poly DP', image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    image = cv2.imread('course_files/images/hand.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 176, 255, 0)

    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=False)[:-1]

    for c in contours:
        hull = cv2.convexHull(c)
        print(hull)
        cv2.drawContours(image, [hull], 0, (0,255,0), 2)
        cv2.imshow('convex hull', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif subsection==34:
    image = cv2.imread('course_files/images/bunchofshapes.jpg')
    cv2.imshow('Bunch of shapes', image)
    cv2.waitKey(0)

    black_image = np.zeros((image.shape[0], image.shape[1], 3))

    orig_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(gray, 50, 200)

    cv2.imshow('1 - Canny Edges', edged)
    cv2.waitKey(0)

    original_edged = edged.copy()

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(black_image, contours, -1, (0,255,0), 3)
    cv2.imshow('1 - All contours over black image', black_image)
    cv2.waitKey(0)

    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    cv2.imshow('1 - All contours over the original image', image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

    def get_contour_areas(contours):
        return [cv2.contourArea(c) for c in contours]

    print(get_contour_areas(contours))

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in sorted_contours:
        cv2.drawContours(orig_image, [c], -1, (255,0,0), 3)
        cv2.imshow('Contours by area', orig_image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    def x_cord_contour(c):
        if cv2.contourArea(c) > 10:
            M = cv2.moments(c)
            return (int(M['m10']/M['m00']))

    def label_contour_center(image, c, i):
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(image, (cx, cy), 10, (0,0,255), -1)
#        return image

    image = cv2.imread('course_files/images/bunchofshapes.jpg')
    orig_image = image.copy()

    for (i,c) in enumerate(contours):
#        orig = label_contour_center(image, c, i)
        label_contour_center(image, c, i)

    cv2.imshow("Contour Centers", image)
    cv2.waitKey(0)

    contours_left_to_right = sorted(contours, key=x_cord_contour, reverse=False)

    for i, c in enumerate(contours_left_to_right):
        cv2.drawContours(orig_image, [c], -1, (0, 0, 255), 3)
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(orig_image, str(i+1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        cv2.imshow("Left-to-right contours", orig_image)
        cv2.waitKey(0)
        (x, y, w, h) = cv2.boundingRect(c)
        cropped_contour = orig_image[y:y+h, x:x+w]

    cv2.destroyAllWindows()

elif subsection==33:
    if False:
        image = cv2.imread('course_files/images/shapes.jpg')
    else:
        image = cv2.imread('course_files/images/shapes_donut.jpg')
    cv2.imshow('Input image', image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(gray, 30, 200)
    cv2.imshow('Canny edges', edged)
    cv2.waitKey(0)

    if False:
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    else:
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    cv2.imshow('Canny edges after contouring', edged)
    cv2.waitKey(0)

    cv2.drawContours(image, contours, -1, (0,255,0), 3)

    cv2.imshow('Contours', image)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
