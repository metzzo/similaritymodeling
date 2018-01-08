import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import random
import pickle
import os.path
import matplotlib.pyplot as plt
from load_ground_truth import load_data

import librosa
import librosa.display

def extract_feature(X, sample_rate):
    stft = np.abs(librosa.stft(X))

    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T

    mfccs_mean = np.mean(mfccs, axis=0)
    mfccs_std = np.std(mfccs, axis=0)

    chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate).T
    chroma_mean = np.mean(chroma, axis=0)
    chroma_std = np.std(chroma, axis=0)

    mel = librosa.feature.melspectrogram(X, sr=sample_rate).T
    mel_mean = np.mean(mel, axis=0)
    mel_std = np.std(mel, axis=0)

    contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T
    contrast_mean = np.mean(contrast, axis=0)
    contrast_std = np.std(contrast, axis=0)

    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T
    tonnetz_mean = np.mean(tonnetz, axis=0)
    tonnetz_std = np.std(tonnetz, axis=0)

    zcr = librosa.feature.zero_crossing_rate(y=X).T
    zcr_mean = np.mean(zcr, axis=0)
    zcr_std = np.std(zcr, axis=0)

    rmse = librosa.feature.rmse(y=X).T
    rmse_mean = np.mean(rmse, axis=0)
    rmse_std = np.mean(rmse, axis=0)

    oenv = librosa.onset.onset_strength(y=X, sr=sample_rate, feature=librosa.cqt).T
    oenv = oenv / oenv.max()
    oenv_mean = np.mean(oenv, axis=0)
    oenv_std = np.mean(oenv, axis=0)

    return np.hstack([
        mfccs_mean,
        mfccs_std,
        chroma_mean,
        chroma_std,
        mel_mean,
        mel_std,
        contrast_mean,
        contrast_std,
        tonnetz_mean,
        tonnetz_std,
        zcr_mean,
        zcr_std,
        rmse_mean,
        rmse_std,
        oenv_mean,
        oenv_std
    ])

def load_jump_detection_model():
    with tf.variable_scope('jump'):
        JUMP_DELTA = 1
        BACKGROUND_NOISE_SAMPLES = 6
        BACKGROUND_NOISE_DURATION = JUMP_DELTA * 2

        def get_jump_frame(audio, sample_rate, time):
            start_time = (time - JUMP_DELTA) * sample_rate
            end_time = (time + JUMP_DELTA) * sample_rate
            return audio[start_time:end_time]

        def extract_at_time(audio, sample_rate, time):
            audio = get_jump_frame(audio=audio, sample_rate=sample_rate, time=time)
            return extract_feature(X=audio, sample_rate=sample_rate)

        def extract_background_audio(audio, sample_rate, jump1, jump2):
            snippets = np.array([])

            if jump1 >= 0:
                # load part before jumpes
                snippets = np.append(snippets, audio[0:(jump1 - JUMP_DELTA) * sample_rate])

                if jump2 >= 0:
                    # get part between jump1 and jump2
                    snippets = np.append(snippets,
                                         audio[(jump1 + JUMP_DELTA) * sample_rate:(jump2 - JUMP_DELTA) * sample_rate])
                    # get part after jump2
                    snippets = np.append(snippets, audio[(jump2 + JUMP_DELTA) * sample_rate:])
                else:
                    # no other jump => load remaining file
                    snippets = np.append(snippets, audio[(jump1 + JUMP_DELTA) * sample_rate:])
            else:
                # no jumpes: load entire file
                snippets = np.append(snippets, audio)

            return snippets

        def extract_background_feature(audio, sample_rate):
            pos = random.randint(0, int(len(audio) / sample_rate) - BACKGROUND_NOISE_DURATION) * sample_rate
            audio = audio[pos:pos + BACKGROUND_NOISE_DURATION * sample_rate]
            return extract_feature(X=audio, sample_rate=sample_rate)

        def extract_feature_set(filename, noise_samples=BACKGROUND_NOISE_SAMPLES):

            features = np.empty((0, 392))
            labels = np.empty((0, 2))

            if not os.path.isfile(filename + '.file'):
                df = load_data(file=filename)

                for index, row in df.iterrows():
                    audio_filename = row['name']
                    winch1 = row['w1']
                    winch2 = row['w2']
                    jump1 = row['j1']
                    jump2 = row['j2']

                    target_directory = os.path.abspath("../../audio_dataset/") + '\\'
                    target_file = target_directory + audio_filename + '.wav'
                    print("Extract data for " + target_file)

                    audio, sample_rate = librosa.load(target_file)

                    if jump1 >= 0:
                        print("Extract jump1 feature")
                        jump_feature = extract_at_time(audio=audio, sample_rate=sample_rate, time=jump1)
                        features = np.vstack([features, jump_feature])
                        labels = np.vstack([labels, [1, 0]])

                        jump_feature = extract_at_time(audio=audio, sample_rate=sample_rate, time=jump1 - JUMP_DELTA*2)

                        features = np.vstack([features, jump_feature])
                        labels = np.vstack([labels, [0, 1]])

                        jump_feature = extract_at_time(audio=audio, sample_rate=sample_rate, time=jump1 + JUMP_DELTA*2)
                        features = np.vstack([features, jump_feature])
                        labels = np.vstack([labels, [0, 1]])

                    if jump2 >= 0:
                        print("Extract jump2 feature")
                        jump_feature = extract_at_time(audio=audio, sample_rate=sample_rate, time=jump2)
                        features = np.vstack([features, jump_feature])
                        labels = np.vstack([labels, [1, 0]])

                    if winch1 >= 0:
                        print("Extract winch1 feature")
                        winch_feature = extract_at_time(audio=audio, sample_rate=sample_rate, time=winch1)
                        features = np.vstack([features, winch_feature])
                        labels = np.vstack([labels, [0, 1]])

                    if winch2 >= 0:
                        print("Extract winch2 feature")
                        winch_feature = extract_at_time(audio=audio, sample_rate=sample_rate, time=winch2)
                        features = np.vstack([features, winch_feature])
                        labels = np.vstack([labels, [0, 1]])

                    audio_noise = extract_background_audio(audio=audio, sample_rate=sample_rate, jump1=jump1, jump2=jump2)
                    for i in range(0, noise_samples):
                        print("Extract background feature: " + str(i))
                        noise_feature = extract_background_feature(audio=audio_noise, sample_rate=sample_rate)
                        features = np.vstack([features, noise_feature])
                        labels = np.vstack([labels, [0, 1]])

                with open(filename + '.file', 'wb') as fp:
                    pickle.dump((features, labels), fp)

            else:
                file = open(filename + ".file", 'rb')
                features, labels = pickle.load(file)
                file.close()

            return features, labels

        train_features, train_labels = extract_feature_set(filename='ground-truth.csv')
        print(train_features.shape)
        print(train_labels.shape)

        test_features, test_labels = extract_feature_set(filename='test-truth.csv', noise_samples=1000)
        print(test_features.shape)
        print(test_labels.shape)

        n_dim = train_features.shape[1]
        n_classes = 2
        n_hidden_units_one = 280
        n_hidden_units_two = 300
        sd = 1 / np.sqrt(n_dim)

        X = tf.placeholder(tf.float32, [None, n_dim])
        Y = tf.placeholder(tf.float32, [None, n_classes])

        W_1 = tf.Variable(tf.random_normal([n_dim, n_hidden_units_one], mean=0, stddev=sd))
        b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean=0, stddev=sd))
        h_1 = tf.nn.tanh(tf.matmul(X, W_1) + b_1)

        W_2 = tf.Variable(tf.random_normal([n_hidden_units_one, n_hidden_units_two], mean=0, stddev=sd))
        b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean=0, stddev=sd))
        h_2 = tf.nn.sigmoid(tf.matmul(h_1, W_2) + b_2)

        W = tf.Variable(tf.random_normal([n_hidden_units_two, n_classes], mean=0, stddev=sd))
        b = tf.Variable(tf.random_normal([n_classes], mean=0, stddev=sd))
        y_ = tf.nn.softmax(tf.matmul(h_2, W) + b)

        return X, Y, y_, train_features, train_labels, test_features, test_labels