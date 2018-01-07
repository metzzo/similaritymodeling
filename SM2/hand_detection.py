import cv2
import librosa

import numpy as np
import cv2
import os
import tensorflow as tf

from SM1.jump_detection_model import load_jump_detection_model, extract_feature as jump
from SM2.skin_detection_model import load_skin_detection_model


skin_graph = tf.Graph()
with skin_graph.as_default():
    X_skin, _, y_skin, _, _, _, _ = load_skin_detection_model()

jump_graph = tf.Graph()
with jump_graph.as_default():
    X_jump, _, y_jump, _, _, _, _ = load_jump_detection_model()
    saver_jump = tf.train.Saver()
    sess_jump = tf.Session()
    saver_jump.restore(sess_jump, os.path.abspath('.') + "/../SM1/jump_detection_model.tf")

with skin_graph.as_default():
    saver_skin = tf.train.Saver()
    sess_skin = tf.Session()
    saver_skin.restore(sess_skin, os.path.abspath('.') + "/skin_detection_model.tf")



def amount_of_skin(frame):
    h = frame.shape[0]
    w = frame.shape[1]

    input = np.empty((w*h, 3))
    for y in range(0, h):
        for x in range(0, w):
            pixel = frame[y, x]
            input[y*w + x] = np.array([pixel])

    pred = sess_skin.run(tf.argmax(y_skin, 1), feed_dict={X_skin: input})
    return pred.sum()



def detect_skin(frame):
    lower = np.array([0, 2, 80], dtype="uint8")
    upper = np.array([20, 210, 240], dtype="uint8")

    # get skin in range
    skinMask = cv2.inRange(frame, lower, upper)

    # blur mask (GAUSS EVERYWHERE \o/)
    skinMask = cv2.GaussianBlur(skinMask, (15, 15), 0)

    # mask the frame
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)

    return skin, skinMask

filename = "1_2015-10-03_18-08-15.mp4" #"1_2015-10-03_13-42-32.mp4" #

target_directory = os.path.abspath("../../dataset/") + '\\'
target_file = target_directory + filename
cap = cv2.VideoCapture(target_file)

#cap.set(cv2.CAP_PROP_POS_MSEC , (9*60 +45) * 1000)
cap.set(cv2.CAP_PROP_POS_MSEC , (6*60 + 11) * 1000)

target_directory = os.path.abspath("../../audio_dataset/") + '\\'
target_audio_file = target_directory + filename + '.wav'
audio, sample_rate = librosa.load(target_audio_file)

hand_positions = []
while(cap.isOpened()):
    ret, frame = cap.read()

    current_second = int(cap.get(cv2.CAP_PROP_POS_MSEC)/1000)

    is_jump = 1
    if current_second > 2 and current_second < len(audio)/sample_rate - 2:
        audio_delta = audio[(current_second - 1)*sample_rate:(current_second + 1)*sample_rate]
        input = np.array([jump.extract_feature(X=audio_delta, sample_rate=sample_rate)])
        is_jump = sess_jump.run(tf.argmax(y_jump, 1), feed_dict={X_jump: input})[0]

    original_frame = frame

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    skin, skinMask = detect_skin(frame)

    ret, thresh = cv2.threshold(skinMask, 180, 255, 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    filtered_contours = []
    for c in contours:
        c_area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)

        if c_area > 50 * 50 and c_area < 100 * 100:
            filtered_contours.append(c)

    contours = filtered_contours

    def skin_ratio(c):
        x, y, w, h = cv2.boundingRect(c)

        # count number of pixels
        candidate = skin[y:y + h, x:x + w]
        skin_area = amount_of_skin(candidate)

        return skin_area/(w*h)

    selected_contours = []
    sr_values = [0]*len(contours)
    er_values = [0]*len(contours)

    i = 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        candidate = skin[y:y + h, x:x + w]
        edges = cv2.Canny(candidate, 25, 3*25)

        # the bigger a contour the more edges are allowed
        c_area = cv2.contourArea(c)

        sr = skin_ratio(c)

        # the more pixel detected by canny the rougher the surface is
        roughness = 0
        for y in range(0, h):
            for x in range(0, w):
                pixel = edges[y, x]
                if pixel == 0:
                    roughness += 1

        er = roughness/(w*h)

        # print(er, " Skin: ", sr)

        sr_values[i] = sr
        er_values[i] = er


        i += 1

    retry = 0
    while len(selected_contours) < 2 and retry < 5:
        i = 0
        for c in contours:
            sr = sr_values[i]
            er = er_values[i]
            if sr <= 0.8 + retry*0.05 and sr >= 0.5 - retry*0.05 and er < 0.8 + retry*0.05:
                selected_contours.append(c)

            i += 1
        retry += 1



    edges = cv2.Canny(skin, 25, 3*25)
    cv2.drawContours(skin, contours, -1, 255, 3)
    skin = np.bitwise_or(skin, edges[:, :, np.newaxis])

    def history_aware_rect(contour):
        hand_positions.append(contour)

        new_x, new_y, new_w, new_h = cv2.boundingRect(contour)

        final_x = 0
        final_y = 0
        final_w = 0
        final_h = 0

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

        if len(hand_positions) > 10:
            hand_positions.pop(0)

        return (final_x, final_y, final_w, final_h)


    if len(selected_contours) > 0:
        x, y, w, h = history_aware_rect(selected_contours[0])

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if len(selected_contours) > 1:
        x, y, w, h = history_aware_rect(selected_contours[1])

        cv2.rectangle(original_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_frame, 'Jump' if is_jump == 0 else "No Jump", (20, 80), font, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("images", np.hstack([frame, original_frame, skin]))

    # sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('p'):
        print("wait")

cap.release()
cv2.destroyAllWindows()
sess_skin.close()
sess_jump.close()

