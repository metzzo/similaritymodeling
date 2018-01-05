import cv2

import numpy as np
import cv2
import os
from time import sleep
from matplotlib import pyplot as plt
import SM2.skin_detector as skin_detector

def detect_skin(frame):
    lower = np.array([0, 2, 80], dtype="uint8")
    upper = np.array([20, 210, 240], dtype="uint8")

    # get skin in range
    skinMask = cv2.inRange(frame, lower, upper)

    # blur mask (GAUSS EVERYWHERE \o/)
    skinMask = cv2.GaussianBlur(skinMask, (11, 11), 0)

    # mask the frame
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    return skin, skinMask

target_directory = os.path.abspath("../../dataset/") + '\\'
target_file = target_directory +  "1_2015-10-03_13-42-32.mp4" # "1_2015-10-03_18-08-15.mp4" #

cap = cv2.VideoCapture(target_file)

cap.set(cv2.CAP_PROP_POS_MSEC , (9*60 +45) * 1000)
#cap.set(cv2.CAP_PROP_POS_MSEC , (6*60 +11) * 1000)


def find_mounting(frame):
    lower = np.array([40, 0, 180], dtype="uint8")
    upper = np.array([90, 20, 255], dtype="uint8")

    ropeMask = cv2.inRange(frame, lower, upper)

    ropeMask = cv2.GaussianBlur(ropeMask, (7, 7), 0)
    ropeFrame = cv2.bitwise_and(frame, frame, mask=ropeMask)

    # edges = cv2.Canny(ropeFrame, 150, 200)
    minLineLength = 50
    maxLineGap = 50
    lines = cv2.HoughLinesP(ropeMask, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    return lines

def find_if_close(cnt1,cnt2):
    row1,row2 = cnt1.shape[0],cnt2.shape[0]
    for i in range(row1):
        for j in range(row2):
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            if abs(dist) < 50 :
                return True
            elif i==row1-1 and j==row2-1:
                return False

hand_positions = []
while(cap.isOpened()):
    ret, frame = cap.read()
    original_frame = frame

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



    skin, skinMask = detect_skin(frame)

    ret, thresh = cv2.threshold(skinMask, 180, 255, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contours = [c for c in contours if cv2.contourArea(c) > 50]
    """
    LENGTH = len(contours)
    status = np.zeros((LENGTH, 1))

    for i, cnt1 in enumerate(contours):
        x = i
        if i != LENGTH - 1:
            for j, cnt2 in enumerate(contours[i + 1:]):
                x = x + 1
                dist = find_if_close(cnt1, cnt2)
                if dist == True:
                    val = min(status[i], status[x])
                    status[x] = status[i] = val
                else:
                    if status[x] == status[i]:
                        status[x] = i + 1

    unified = []
    maximum = int(status.max()) + 1
    for i in range(maximum):
        pos = np.where(status == i)[0]
        if pos.size != 0:
            cont = np.vstack(contours[i] for i in pos)
            hull = cv2.convexHull(cont)
            unified.append(hull)

    countours = unified"""

    cv2.drawContours(skin, contours, -1, 255, 3)

    def find_best_contour(c):
        x, y, w, h = cv2.boundingRect(c)
        c_area = cv2.contourArea(c)
        b_area = w * h

        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        dx = cX - 320
        dy = cY - 240

        dist_to_center = np.sqrt(dx*dx + dy*dy)

        return -c_area + dist_to_center


    contours = sorted(contours, key=find_best_contour)

    def history_aware_rect(contour):
        hand_positions.append(contour)

        new_x, new_y, new_w, new_h = cv2.boundingRect(contour)

        final_x = new_x
        final_y = new_y
        final_w = new_w
        final_h = new_h

        count = 0
        for c in hand_positions:
            tmp_x, tmp_y, tmp_w, tmp_h = cv2.boundingRect(c)

            dx = (tmp_x + tmp_w/2) - (new_x + new_w/2)
            dy = (tmp_y + tmp_h/2) - (new_y + new_h/2)

            if np.sqrt(dx * dx + dy * dy) < 20:
                final_x += tmp_x
                final_y += tmp_y
                final_w += tmp_w
                final_h += tmp_h
                count = count + 1

        final_x = int(final_x / count)
        final_y = int(final_y / count)
        final_w = int(final_w / count)
        final_h = int(final_h / count)

        if len(hand_positions) > 60:
            hand_positions.pop(0)

        return (final_x, final_y, final_w, final_h)


    if len(contours) > 0:
        x, y, w, h = history_aware_rect(contours[0])

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(contours) > 1:
        x, y, w, h = history_aware_rect(contours[1])

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)




    """lines = find_mounting(frame)
    if lines is not None:
        for l in lines:
            for line in l:
                cv2.line(original_frame, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
            """


    cv2.imshow("images", np.hstack([frame, original_frame, skin]))

    sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):
        print("wait")


        frames.pop(0)

cap.release()
cv2.destroyAllWindows()

