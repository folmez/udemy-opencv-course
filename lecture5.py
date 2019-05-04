import numpy as np
import cv2
import sys

subsection = int(sys.argv[1])

if subsection==45:
    image = cv2.imread('course_files/images/chess.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if False:
        # gray = np.float32(gray)

        harris_corners = cv2.cornerHarris(gray, 3, 3, 0.05)

        kernel = np.ones((7,7), np.uint8)
        harris_corners = cv2.dilate(harris_corners, kernel, iterations=2)

        image[harris_corners > 0.025 * harris_corners.max()] = [255, 127, 127]

        cv2.imshow('Harris Corners', image)
    else:
        nr_corners_to_track = 50
        min_distance = 15
        corners = cv2.goodFeaturesToTrack(gray, nr_corners_to_track, 0.01,
                                                min_distance)

        for c in corners:
            (x, y) = c[0]
            x, y = int(x), int(y)
            cv2.rectangle(image, (x-10, y-10), (x+10, y+10), (0,255,0), 2)
        cv2.imshow('Corners found via goodFeaturesToTrack', image)

    cv2.waitKey()
    cv2.destroyAllWindows()


elif subsection==43:
    image = cv2.imread('course_files/images/WaldoBeach.jpg')
    cv2.imshow('Where is Waldo?', image)
    cv2.waitKey()
    cv2.destroyAllWindows()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('course_files/images/waldo.jpg', cv2.IMREAD_GRAYSCALE)

    result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    top_left = max_loc
    bottom_right = (top_left[0] + 50, top_left[1] + 50)
    cv2.rectangle(image, top_left, bottom_right, (0,0,255), 5)

    cv2.imshow('Where is Waldo?', image)
    cv2.waitKey()
    cv2.destroyAllWindows()
