import cv2
import librosa

import numpy as np
import cv2
import os
import tensorflow as tf
from scipy.signal.lti_conversion import cont2discrete

from SM1.jump_detection_model import load_jump_detection_model, extract_feature as jump_extract_feature
from SM2.skin_detection_model import load_skin_detection_model
from SM2.winch_detection_model import load_winch_detection_model
from load_ground_truth import load_data

skin_graph = tf.Graph()
with skin_graph.as_default():
    X_skin, _, y_skin, _, _, _, _ = load_skin_detection_model()

winch_graph = tf.Graph()
with winch_graph.as_default():
    X_winch, _, y_winch, _, _, _, _ = load_winch_detection_model()

jump_graph = tf.Graph()
with jump_graph.as_default():
    X_jump, _, y_jump, _, _, _, _ = load_jump_detection_model()
    saver_jump = tf.train.Saver()
    sess_jump = tf.Session()
    saver_jump.restore(sess_jump, os.path.abspath('.') + "/../SM1/jump_model/jump_detection_model.tf")

with skin_graph.as_default():
    saver_skin = tf.train.Saver()
    sess_skin = tf.Session()
    saver_skin.restore(sess_skin, os.path.abspath('.') + "/skin_model/skin_detection_model.tf")

with winch_graph.as_default():
    winch_skin = tf.train.Saver()
    sess_winch = tf.Session()
    saver_skin.restore(sess_winch, os.path.abspath('.') + "/winch_model/winch_detection_model.tf")


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

ground_truth = load_data(file='ground-truth.csv')

samples = []
EVENT_DELTA = 0.5*60*1000

for _, entry in ground_truth.iterrows():
    source_filename = entry['name']
    j1 = entry['j1']
    j2 = entry['j2']
    w1 = entry['w1']
    w2 = entry['w2']

    any = False
    if j1 >= 0:
        any = True
        samples.append((source_filename, source_filename + ".jump1.avi", j1*1000 - EVENT_DELTA, EVENT_DELTA*2))

    if j2 >= 0:
        any = True
        samples.append((source_filename, source_filename + ".jump2.avi", j2*1000 - EVENT_DELTA, EVENT_DELTA*2))

    if w1 >= 0:
        any = True
        samples.append((source_filename, source_filename + ".winch1.avi", w1 * 1000 - EVENT_DELTA, EVENT_DELTA*2))

    if w2 >= 0:
        any = True
        samples.append((source_filename, source_filename + ".winch2.avi", w2 * 1000 - EVENT_DELTA, EVENT_DELTA * 2))

    if not any:
        samples.append((source_filename, source_filename + ".nothing.avi", 0, EVENT_DELTA * 2))


def extract(sample):
    cont = True

    source_file, target_file, start_time, duration = sample
    start_time = max(0, start_time)

    target_file = '../../result/' + target_file
    if os.path.isfile(target_file):
        return True

    # process winch/jump usages

    target_directory = os.path.abspath("../../audio_dataset/") + '\\'
    target_audio_file = target_directory + source_file + '.wav'
    audio, sample_rate = librosa.load(target_audio_file)

    total_duration_in_seconds = len(audio)/sample_rate
    duration_in_seconds = duration/1000
    start_time_in_seconds = start_time/1000
    input = []
    for time in np.arange(max(start_time_in_seconds, 2), min(start_time_in_seconds + duration_in_seconds, total_duration_in_seconds - 2), 0.25):
        print("extract time: " + str(time))
        audio_delta = audio[int((time - 1)*sample_rate):int((time + 1)*sample_rate)]
        feature = jump_extract_feature(X=audio_delta, sample_rate=sample_rate)
        input.append(feature)

    input = np.array(input)

    print("get jump/winch results")
    is_jump_lookup = sess_jump.run(tf.argmax(y_jump, 1), feed_dict={X_jump: input})
    # to save some time the same input is used, if the feature space of winch vs jump changes the features have to be recalculated
    is_winch_lookup = sess_winch.run(tf.argmax(y_winch, 1), feed_dict={X_winch: input})
    print("start video")
    end_time = start_time + duration

    target_directory = os.path.abspath("../../dataset/") + '\\'
    cap = cv2.VideoCapture(target_directory + source_file)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(target_file, fourcc, 30.0, (640, 480))
    hand_positions = []
    while(cap.isOpened()):
        ret, frame = cap.read()

        if not ret:
            break

        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        if current_ms > end_time:
            break

        lookup_time = int((current_ms - start_time)/(1000/4))

        is_jump = is_jump_lookup[min(lookup_time, len(is_jump_lookup) - 1)]
        is_winch = is_winch_lookup[min(lookup_time, len(is_winch_lookup) - 1)]


        original_frame = frame
        try:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except AssertionError as e:
            print('-' * 60)
            tf.traceback.print_exc(file=tf.sys.stdout)
            print('-' * 60)
            continue

        skin, skinMask = detect_skin(frame)

        ret, thresh = cv2.threshold(skinMask, 180, 255, 0)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60, 60))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        filtered_contours = []
        for c in contours:
            c_area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)

            # area must be fesaible and bounding rect must also be not quite at the bottom (since there is the head)
            if c_area > 25 * 25 and c_area < 100 * 100 and (y + h) < 460:
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
        while len(selected_contours) < 2 and retry < 3:
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
        cv2.putText(original_frame, 'Winch' if is_winch == 0 else "No Winch", (20, 160), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(original_frame, 'Time ' + str(int(current_ms/1000 / 60)) + ":" + str(int(current_ms/1000 % 60)), (20, 420), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        out.write(original_frame)
        cv2.imshow("images", np.hstack([frame, original_frame, skin]))

        # sleep(0.05)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cont = False
            break
        if cv2.waitKey(1) & 0xFF == ord('p'):
            print("wait")

    cap.release()
    cv2.destroyAllWindows()
    out.release()

    return cont


for sample in samples:
    if not extract(sample):
        break

sess_skin.close()
sess_jump.close()
sess_winch.close()