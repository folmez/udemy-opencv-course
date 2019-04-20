import cv2
import numpy as np
import matplotlib.pyplot as plt

subsection = 2.8

if subsection==2.8:
    if False:
        empty = np.zeros((512,512,3), np.uint8)
        cv2.line(empty, (0,0), (511,511), (255,127,0), 5)
        cv2.imshow("Blue Line", empty)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if False:
        empty = np.zeros((512,512,3), np.uint8)
        cv2.rectangle(empty, (100,100), (300,250), (127,50,127), 5)
        cv2.imshow("Rectangle", empty)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if False:
        empty = np.zeros((512,512,3), np.uint8)
        cv2.circle(empty, (350,350), 50, (12,50,172), 2)
        cv2.imshow("Rectangle", empty)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if False:
        empty = np.zeros((512,512,3), np.uint8)
        pnts = np.array([[120,14],[412,500], [223,281], [72,432]], np.int32)
        pnts = pnts.reshape(-1,1,2)
        cv2.polylines(empty, [pnts], True, (112,50,172), 6)
        cv2.imshow("Polygon", empty)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if True:
        empty = np.zeros((512,512,3), np.uint8)
        cv2.putText(empty, 'Yay!!!', (100,300), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                                                5, (200,340,100), 8)
        cv2.imshow("My text", empty)
        cv2.waitKey()
        cv2.destroyAllWindows()


elif subsection==2.7:
    if False:
        input = cv2.imread('course_files/images/input.jpg')
    else:
        input = cv2.imread('course_files/images/tobago.jpg')
    hist = cv2.calcHist([input], [0], None, [256], [0, 256])
    plt.figure()
    plt.hist(input.flatten(), 256, [0, 256])
    plt.figure()
    plt.plot(hist, color='b')
    plt.show()

    color = ('b', 'g', 'r')

    for i, col in enumerate(color):
        hist2 = cv2.calcHist([input], [i], None, [256], [0, 256])
        plt.plot(hist2, color=col)
        plt.xlim([0,256])
    plt.show()

elif subsection==2.6:
    input = cv2.imread('course_files/images/input.jpg')

    if False:
        B, G, R = input[0,0]
        print(B, G, R)
        print(input.shape)

        gray_input = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        print(gray_input.shape)

    # convert to hue (HSV)
    if False:
        hsv_input = cv2.cvtColor(input, cv2.COLOR_BGR2HSV)
        cv2.imshow('HSV image', hsv_input)
        cv2.waitKey()
        cv2.imshow('Hue', hsv_input[:,:,0])
        cv2.waitKey()
        cv2.imshow('Saturation', hsv_input[:,:,1])
        cv2.waitKey()
        cv2.imshow('Value', hsv_input[:,:,2])
        cv2.waitKey()
        cv2.destroyAllWindows()

    B, G, R = cv2.split(input)

    if False:
        cv2.imshow('Red', R)
        cv2.waitKey()
        cv2.imshow('Green', G)
        cv2.waitKey()
        cv2.imshow('Blue', B)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if False:
        merged_input = cv2.merge([B+100, G, R])
        cv2.imshow('Merged with blue amplified', merged_input)
        cv2.waitKey()
        cv2.destroyAllWindows()

    empty = np.zeros(input.shape[0:2], dtype="uint8")

    cv2.imshow('Red', cv2.merge([empty, empty, R]))
    cv2.waitKey()
    cv2.imshow('Green', cv2.merge([empty, G, empty]))
    cv2.waitKey()
    cv2.imshow('Blue', cv2.merge([B, empty, empty]))
    cv2.waitKey()
    cv2.destroyAllWindows()


elif subsection==2.5:
    input = cv2.imread('course_files/images/input.jpg')
    cv2.imshow('Original', input)
    cv2.waitKey()

    gray_image = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    grayscale_input = cv2.imread('course_files/images/input.jpg', 0)
    cv2.imshow('Original', grayscale_input)
    cv2.waitKey()

elif subsection==2.4:
    input = cv2.imread('course_files/images/input.jpg')
    cv2.imshow('Hello World', input)
    cv2.waitKey()
    cv2.destroyAllWindows()

    cv2.imwrite('output.jpg', input)
    cv2.imwrite('output.png', input)
