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

JUMP_DELTA = 1
BACKGROUND_NOISE_SAMPLES = 5
BACKGROUND_NOISE_DURATION = JUMP_DELTA*2

def get_jump_frame(audio, sample_rate, time):
    start_time = (time - JUMP_DELTA) * sample_rate
    end_time = (time + JUMP_DELTA) * sample_rate
    return audio[start_time:end_time]


def extract_feature(X, sample_rate):
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)

    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])


def extract_jump(audio, sample_rate, time):
    audio = get_jump_frame(audio=audio, sample_rate=sample_rate, time=time)
    return extract_feature(X=audio, sample_rate=sample_rate)


def extract_background_audio(audio, sample_rate, jump1, jump2):
    snippets = np.array([])

    if jump1 >= 0:
        # load part before jumpes
        snippets = np.append(snippets, audio[0:(jump1 - JUMP_DELTA) * sample_rate])

        if jump2 >= 0:
            # get part between jump1 and jump2
            snippets = np.append(snippets, audio[(jump1 + JUMP_DELTA) * sample_rate:(jump2 - JUMP_DELTA) * sample_rate])
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
    audio = audio[pos:pos + BACKGROUND_NOISE_DURATION*sample_rate]
    return extract_feature(X=audio, sample_rate=sample_rate)


def extract_feature_set(filename, noise_samples=BACKGROUND_NOISE_SAMPLES):

    features = np.empty((0, 193))
    labels = np.empty((0, 2))

    if not os.path.isfile(filename+'.file'):
        df = load_data(file=filename)

        for index, row in df.iterrows():
            audio_filename = row['name']
            jump1 = row['j1']
            jump2 = row['j2']

            target_directory = os.path.abspath("../../audio_dataset/") + '\\'
            target_file = target_directory + audio_filename + '.wav'
            print("Extract data for " + target_file)

            audio, sample_rate = librosa.load(target_file)

            if jump1 >= 0:
                print("Extract jump1 feature")
                jump_feature = extract_jump(audio=audio, sample_rate=sample_rate, time=jump1)
                features = np.vstack([features, jump_feature])
                labels = np.vstack([labels, [1, 0]])

            if jump2 >= 0:
                print("Extract jump2 feature")
                jump_feature = extract_jump(audio=audio, sample_rate=sample_rate, time=jump2)
                features = np.vstack([features, jump_feature])
                labels = np.vstack([labels, [1, 0]])

            audio_noise = extract_background_audio(audio=audio, sample_rate=sample_rate, jump1=jump1, jump2=jump2)
            for i in range(1, noise_samples):
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

test_features, test_labels = extract_feature_set(filename='test-truth.csv', noise_samples=30)
print(test_features.shape)
print(test_labels.shape)

training_epochs = 5000
n_dim = train_features.shape[1]
n_classes = 2
n_hidden_units_one = 280
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01

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

init = tf.global_variables_initializer()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1], dtype=float)
y_true, y_pred = None, None
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        if epoch % 50 == 0:
            print("Training Epoch " + str(epoch))

        _, cost = sess.run([optimizer, cost_function], feed_dict={X: train_features, Y: train_labels})
        cost_history = np.append(cost_history, cost)

    y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: test_features})
    y_true = sess.run(tf.argmax(test_labels, 1))


fig = plt.figure(figsize=(10,8))
plt.plot(cost_history)
plt.ylabel("Cost")
plt.xlabel("Iterations")
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='binary')

print("Sample Size: " + str(len(y_true)))
print("Precision:", round(p,3))
print("Recall:", round(r,3))
print("F-Score:", round(f,3))

"""
Result on my machine:

Sample Size: 420
Precision: 0.985
Recall: 0.985
F-Score: 0.985

"""
