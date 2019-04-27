import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

subsection = int(sys.argv[1])

if subsection==26:
    image = cv2.imread('course_files/images/elephant.jpg')
else:
    image = cv2.imread('course_files/images/input.jpg')

def sketch(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
    canny_edges = cv2.Canny(img_gray_blur, 10, 50)
    ret, mask = cv2.threshold(canny_edges, 50, 255, cv2.THRESH_BINARY_INV)
    return mask

if subsection==32:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv2.imshow('Live skethcer', sketch(frame))
        if cv2.waitKey(1)==13:
            break

    cap.release()
    cv2.destroyAllWindows()

elif subsection==31:
    if False:
        image = cv2.imread('course_files/images/scan.jpg')

        points_A = np.float32([[320,15], [700,215], [85,610], [530,780]])
        points_B = np.float32([[0,0], [420,0], [0,594], [420,594]])

        M = cv2.getPerspectiveTransform(points_A, points_B)
        warped = cv2.warpPerspective(image, M, (420, 594))

        cv2.imshow('original', image)
        cv2.imshow('Warped', warped)
        cv2.waitKey()
        cv2.destroyAllWindows()

    image = cv2.imread('course_files/images/ex2.jpg')

    rows, cols, ch = image.shape

    points_A = np.float32([[320,15], [700,215], [85,610]])
    points_B = np.float32([[0,0], [420,0], [0,594]])

    M = cv2.getAffineTransform(points_A, points_B)
    warped = cv2.warpAffine(image, M, (cols, rows))

    cv2.imshow('original', image)
    cv2.imshow('Warped', warped)
    cv2.waitKey()
    cv2.destroyAllWindows()

elif subsection==30:
    image = cv2.imread('course_files/images/input.jpg', 0)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)

    cv2.imshow('Original', image)
    cv2.imshow('Sobel X', sobel_x)
    cv2.imshow('Sobel Y', sobel_y)

    sobel_OR = cv2.bitwise_or(sobel_x, sobel_y)
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    cv2.imshow('Sobel OR', sobel_OR)
    cv2.imshow('Laplacian', laplacian)

    canny = cv2.Canny(image, 20,170)
    cv2.imshow('Canny', canny)

    cv2.waitKey()
    cv2.destroyAllWindows()

    h, w = image.shape[0:2]

elif subsection==29:
    image = cv2.imread('course_files/images/opencv_inv.png', 0)
    kernel = np.ones((5,5), np.uint8)

    cv2.imshow('Original', image)
    cv2.imshow('Erosion',  cv2.erode(image, kernel, iterations=1))
    cv2.imshow('Dilation', cv2.dilate(image, kernel, iterations=1))
    cv2.imshow('Opening',  cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel))
    cv2.imshow('Closing',  cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel))

    cv2.waitKey()
    cv2.destroyAllWindows()

elif subsection==28:
    # I have had some problems running the second half of this part
    if False:
        image = cv2.imread('course_files/images/gradient.jpg')
        ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
        ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
        ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)

        cv2.imshow('Original', image)
        cv2.imshow('THRESH_BINARY', thresh1)
        cv2.imshow('THRESH_BINARY_INV', thresh2)
        cv2.imshow('THRESH_TRUNC', thresh3)
        cv2.imshow('THRESH_TOZERO', thresh4)
        cv2.imshow('THRESH_TOZERO_INV', thresh5)
        cv2.waitKey()
        cv2.destroyAllWindows()

    image = cv2.imread('course_files/images/Origin_of_Species.jpg', 0)
    cv2.imshow('Original', image)

    ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow('THRESH_BINARY', thresh1)

    image = cv2.GaussianBlur(image, (3,3), 0)
    if False:
        a_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                   cv2.THRESH_BINARY, 3, 5)
        cv2.imshow('ADAPTIVE_THRESH_MEAN_C', a_thresh)

    thresh3 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Otsus thresholding', thresh3)

    blur = cv2.GaussianBlur(image, (5,5), 0)
    thresh4 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Gaussian Otsus thresholding', thresh4)

    cv2.waitKey()
    cv2.destroyAllWindows()

elif subsection==27:
    sharpening_kernel = (-1) * np.ones((3,3))
    sharpening_kernel[1,1] = 9

    sharpened = cv2.filter2D(image, -1, sharpening_kernel)

    cv2.imshow('Sharpened', sharpened)
    cv2.waitKey()
    cv2.destroyAllWindows()

elif subsection==26:
    if False:
        kernel = 1/25 * np.ones((5,5))
        image_blurred = cv2.filter2D(image, -1, kernel)

        cv2.imshow('Elephant', image)
        cv2.imshow('Blurred elephant', image_blurred)
        cv2.waitKey()
        cv2.destroyAllWindows()

    if False:
        blur = cv2.blur(image, (3,3))
        cv2.imshow('Averaging', blur)
        gaussian_blur = cv2.GaussianBlur(image, (7,7), 0)
        cv2.imshow('Gaussian Blurring', gaussian_blur)
        median_blur = cv2.medianBlur(image, 5)
        cv2.imshow('Median Blurring', median_blur)
        bileater_blur = cv2.bilateralFilter(image, 9, 75, 75)
        cv2.imshow('Biletaral Blurring', bileater_blur)
        cv2.waitKey()
        cv2.destroyAllWindows()

    dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
    cv2.imshow('Original', image)
    cv2.imshow('Fast Means Denoising', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

elif subsection==25:
    square = np.zeros((300, 300), np.uint8)
    cv2.rectangle(square, (50,50), (250,250), 255, -2)
    cv2.imshow('Square', square)

    ellipse = np.zeros((300, 300), np.uint8)
    cv2.ellipse(ellipse, (150,150), (150,150), 30, 0, 180, 255, -1)
    cv2.imshow('Ellipse', ellipse)
    cv2.waitKey()
    cv2.destroyAllWindows()

    And = cv2.bitwise_and(square, ellipse)
    cv2.imshow('AND', And)
    Or = cv2.bitwise_or(square, ellipse)
    cv2.imshow('OR', Or)
    XOr = cv2.bitwise_xor(square, ellipse)
    cv2.imshow('XOR', XOr)
    not_sq = cv2.bitwise_not(square)
    cv2.imshow('Not square', not_sq)
    cv2.waitKey()
    cv2.destroyAllWindows()

elif subsection==24:
    if False:
        M = 75 * np.ones(image.shape, dtype=np.uint8)

        added = cv2.add(image, M)
        subtracted = cv2.subtract(image, M)

        cv2.imshow('Added', added)
        cv2.imshow('Subtracted', subtracted)
        cv2.waitKey()
        cv2.destroyAllWindows()

    square = np.zeros((300, 300), np.uint8)
    cv2.rectangle(square, (50,50), (250,250), 255, -2)
    cv2.imshow('Square', square)

    ellipse = np.zeros((300, 300), np.uint8)
    cv2.ellipse(ellipse, (150,150), (150,150), 30, 0, 180, 255, -1)
    cv2.imshow('Ellipse', ellipse)
    cv2.waitKey()
    cv2.destroyAllWindows()

elif subsection==23:
    h, w = image.shape[0:2]

    i, j = 0.45, 0.55
    start_row, start_col = np.int(h*i), np.int(w*i)
    end_row, end_col = np.int(h*j), np.int(w*j)

    cropped = image[start_row:end_row, start_col:end_col, :]

    cv2.imshow('Cropped', cropped)
    cv2.waitKey()
    cv2.destroyAllWindows()

    a = 3
    cropped_scaled = cv2.resize(cropped, None, fx=a, fy=a, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Scaling - Cubic interpolation', cropped_scaled)
    cropped_scaled = cv2.resize(cropped, None, fx=a, fy=a, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('Scaling - Linear interpolation', cropped_scaled)
    cropped_scaled = cv2.resize(cropped, None, fx=a, fy=a, interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow('Scaling - LANCZOS4 interpolation', cropped_scaled)
    cv2.waitKey()
    cv2.destroyAllWindows()

elif subsection==22:
    smaller = cv2.pyrDown(image)
    larger = cv2.pyrUp(image)

    cv2.imshow('Smaller', smaller)
    cv2.waitKey()

    cv2.imshow('Larger', larger)
    cv2.waitKey()

    cv2.imshow('Larger', cv2.pyrUp(smaller))
    cv2.waitKey()

    cv2.destroyAllWindows()

elif subsection==21:
    image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75)
    cv2.imshow('Scaling - Linear interpolation', image_scaled)
    cv2.waitKey()

    image_scaled = cv2.resize(image, None, fx=0.75, fy=0.75, interpolation=cv2.INTER_CUBIC)
    cv2.imshow('Scaling - Cubic interpolation', image_scaled)
    cv2.waitKey()

    image_scaled = cv2.resize(image, (900, 400), interpolation=cv2.INTER_AREA)
    cv2.imshow('Scaling - Cubic interpolation', image_scaled)
    cv2.waitKey()

    cv2.destroyAllWindows()
