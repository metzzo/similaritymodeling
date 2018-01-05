import cv2

import numpy as np
import cv2
import os
from time import sleep
from matplotlib import pyplot as plt
def detect_skin(frame):
    lower = np.array([0, 2, 80], dtype="uint8")
    upper = np.array([30, 255, 255], dtype="uint8")

    # get skin in range
    skinMask = cv2.inRange(frame, lower, upper)

    # erode and dilate
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, ))
    #skinMask = cv2.erode(skinMask, kernel, iterations=2)
    #skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # blur mask (GAUSS EVERYWHERE \o/)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # mask the frame
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    return skin

target_directory = os.path.abspath("../../dataset/") + '\\'
target_file = target_directory + "1_2015-10-03_13-42-32.mp4"

cap = cv2.VideoCapture(target_file)

cap.set(cv2.CAP_PROP_POS_MSEC , (9*60 +45) * 1000)

def find_mounting(frame):
    lower = np.array([40, 0, 180], dtype="uint8")
    upper = np.array([100, 20, 255], dtype="uint8")

    ropeMask = cv2.inRange(frame, lower, upper)

    ropeMask = cv2.GaussianBlur(ropeMask, (7, 7), 0)
    ropeFrame = cv2.bitwise_and(frame, frame, mask=ropeMask)

    # edges = cv2.Canny(ropeFrame, 150, 200)
    minLineLength = 50
    maxLineGap = 50
    lines = cv2.HoughLinesP(ropeMask, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    return lines

while(cap.isOpened()):
    ret, frame = cap.read()
    original_frame = frame

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #frame = cv2.GaussianBlur(frame, (7, 7), 0)

    skin = detect_skin(frame)

    lines = find_mounting(frame)
    if lines is not None:
        for l in lines:
            for line in l:
                cv2.line(original_frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
            """ for rho, theta in line:

                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))

                cv2.line(original_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)"""

    cv2.imshow("images", np.hstack([frame, original_frame]))
    #cv2.imshow("edges", edges)

    #plt.imshow(frame)
    #plt.show()
    """ plt.subplot(121), plt.imshow(frame, cmap='g ray')
 plt.title('Original Image'), plt.xticks([]), plt.yticks([])
 plt.subplot(122), plt.imshow(edges, cmap='gray')
 plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
 plt.show()
 # histogram
 color = ('r', 'g', 'b')
 for i, col in enumerate(color):
     histr = cv2.calcHist([frame], [i], None, [256], [0, 256])
     plt.plot(histr, color=col)
     plt.xlim([0, 256])
 plt.show()"""

    sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):
        print("wait")

cap.release()
cv2.destroyAllWindows()

