import cv2

import numpy as np
import cv2
import os
from time import sleep
import tensorflow as tf
from matplotlib import pyplot as plt

from SM2.skin_detection_model import load_skin_detection_model

tf.reset_default_graph()
X, _, y_, _, _, _, _  = load_skin_detection_model()
saver = tf.train.Saver()
sess = tf.Session()
saver.restore(sess, os.path.abspath('.') + "skin_detection_model.tf")

def amount_of_skin(frame):
    h = frame.shape[0]
    w = frame.shape[1]

    input = np.empty((w*h, 3))
    for y in range(0, h):
        for x in range(0, w):
            pixel = frame[y, x]
            input[y*w + x] = np.array([pixel])

    pred = sess.run(tf.argmax(y_, 1), feed_dict={X: input})
    return pred.sum()



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
target_file = target_directory + "1_2015-10-03_18-08-15.mp4" #"1_2015-10-03_13-42-32.mp4" #

cap = cv2.VideoCapture(target_file)

#cap.set(cv2.CAP_PROP_POS_MSEC , (9*60 +45) * 1000)
cap.set(cv2.CAP_PROP_POS_MSEC , (6*60 +11) * 1000)

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

    contours = [c for c in contours if cv2.contourArea(c) > 50 and cv2.contourArea(c) < 100*100]

    cv2.drawContours(skin, contours, -1, 255, 3)

    def find_best_contour(c):
        x, y, w, h = cv2.boundingRect(c)

        # now count number of pixels of skin
        c_area = cv2.contourArea(c)

        # count number of pixels
        candidate = skin[y:y + h, x:x + w]
        skin_area = amount_of_skin(candidate)

        return -skin_area

    contours = sorted(contours, key=find_best_contour)

    def history_aware_rect(contour):
        hand_positions.append(contour)

        new_x, new_y, new_w, new_h = cv2.boundingRect(contour)
        return new_x, new_y, new_w, new_h

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

        if len(hand_positions) > 30:
            hand_positions.pop(0)

        return (final_x, final_y, final_w, final_h)


    if len(contours) > 0:
        x, y, w, h = history_aware_rect(contours[0])

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(contours) > 1:
        x, y, w, h = history_aware_rect(contours[1])

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("images", np.hstack([frame, original_frame, skin]))

    # sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):
        print("wait")


        frames.pop(0)

cap.release()
cv2.destroyAllWindows()
sess.close()

