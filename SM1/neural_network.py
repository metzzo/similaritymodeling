from scipy.io import wavfile
from scipy import signal
from numpy.fft import fft

#from scipy.signal import spectrogram
import scipy
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import pickle
import os.path
from SM1.load_ground_truth import load_data

import librosa
import librosa.display

class GroundTruthEntry:
    def __init__(self, filename, jump1, jump2, data):
        self.filename = filename
        self.jump1 = jump1
        self.jump2 = jump2
        self.data = data

def extract_features(filename):
    training_set = []

    if not os.path.isfile(filename+'.file'):
        df = load_data(file=filename)

        for index, row in df.iterrows():
            audio_filename = row['name']
            jump1 = row['j1']
            jump2 = row['j2']

            target_directory = os.path.abspath("../../audio_dataset/") + '\\'
            target_file = target_directory + audio_filename + '.wav'
            print("Extract data for " + target_file)

            y, sr = librosa.load(target_file, sr=16000)
            S = librosa.stft(y, win_length=20000, n_fft=20000, hop_length=16000, window=scipy.signal.hanning)
            S = np.abs(S)
            S = np.transpose(S)

            T = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=160, n_fft=20000, hop_length=16000).T
            print("Sampling Rate: " + str(sr))
            print("MFCC: " + str(S))
            print("Shape: " + str(S.shape))

            entry = GroundTruthEntry(filename=audio_filename, jump1=jump1, jump2=jump2, data=np.column_stack((S, T)))
            training_set.append(entry)

        with open(filename + '.file', 'wb') as fp:
            pickle.dump(training_set, fp)

    else:

        file = open(filename + ".file", 'rb')
        training_set = pickle.load(file)
        file.close()

    return training_set

def next_batch(training, positives, batch_size):

    batch = []
    labels = []

    ratio = 1/2

    for _ in range(0, int(batch_size * ratio)):
        positive = positives[random.randint(0, len(positives) - 1)]
        batch.append(positive)
        labels.append([1, 0])

    for _ in range(0, int(batch_size * (1 - ratio))):
        entry_index = random.randint(0, len(training) - 1)
        entry = training[entry_index]
        pos = random.randint(0, entry.data.shape[0] - 1)

        if pos != entry.jump1 and pos != entry.jump2:
            batch.append(entry.data[pos])
            labels.append([0, 1])

    return batch, labels

def extract_positives(training):
    positives = []
    for entry in training:
        if entry.jump1 >= 0:
            positives.append(entry.data[entry.jump1])
        if entry.jump2 >= 0:
            positives.append(entry.data[entry.jump2])
    return positives

def test_batch(test):
    batch = []
    labels = []
    for entry in test:
        if entry.jump1 >= 0:
            batch.append(entry.data[entry.jump1])
            labels.append([1, 0])
        if entry.jump2 >= 0:
            batch.append(entry.data[entry.jump2])
            labels.append([1, 0])

        for pos in range(0, entry.data.shape[0]):
            if pos != entry.jump1 and pos != entry.jump2:
                batch.append(entry.data[pos])
                labels.append([0, 1])

    return batch, labels

training = extract_features(filename='ground-truth.csv')
positives = extract_positives(training)

learning_rate = 0.001
num_steps = 1000
batch_size = 200
display_step = 100

n_hidden_1 = 5000
n_hidden_2 = 4000
n_hidden_3 = 3000
n_hidden_4 = 2000
n_hidden_5 = 1000
n_hidden_6 = 500
n_hidden_7 = 100
n_hidden_8 = 10

num_input = 10161
num_classes = 2

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
    'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
    'h8': tf.Variable(tf.random_normal([n_hidden_7, n_hidden_8])),
    'out': tf.Variable(tf.random_normal([n_hidden_8, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'b6': tf.Variable(tf.random_normal([n_hidden_6])),
    'b7': tf.Variable(tf.random_normal([n_hidden_7])),
    'b8': tf.Variable(tf.random_normal([n_hidden_8])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def neural_net(x):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
    layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
    layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
    layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
    out_layer = tf.matmul(layer_8, weights['out']) + biases['out']
    return out_layer

logits = neural_net(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits,
    labels=Y)
)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))


optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

"""
ratio = 1/2
cost = tf.reduce_mean(-tf.reduce_sum((ratio)*tf.log(logits) + (1 - ratio)*tf.log(1-logits), reduction_indices=1))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)"""



# Evaluate model (with test logits, for dropout to be disabled)

correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

argmax_prediction = tf.argmax(logits, 1)
argmax_y = tf.argmax(Y, 1)

TN = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
TP = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
FN = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
FP = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)

prec = tf.divide(TP, tf.add(TP, FP))
rec = tf.divide(TP, tf.add(TP, FN))
fscore = tf.scalar_mul(2.0, tf.divide(tf.multiply(prec, rec), tf.add(prec, rec)))

init = tf.initialize_all_variables()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
with tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True)) as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps+1):
        batch_x, batch_y = next_batch(training=training, positives=positives, batch_size=batch_size)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    #oSaver = tf.train.Saver()
    #oSaver.save(sess, "models/tensorflow.model")

    print("Optimization Finished!")

    test = extract_features(filename='test-truth.csv')
    test_x, test_y = test_batch(test=test)

    prediction = tf.argmax(logits, 1)
    pred = sess.run(prediction, feed_dict={X: test_x,
                                            Y: test_y})
    time = 0
    lastEntry = test[0]
    index = 0
    for i in range(0, len(pred)):
        if time > lastEntry.data.shape[0]:
            time = 0
            lastEntry = test[index]
            index += 1
        else:
            time += 1
        if pred[i] == 0:
            print("Filename " + lastEntry.filename + " jump at " +
                  str(int(time/60)) + ":" + str(time%60) + "s predicted.")


    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run([accuracy], feed_dict={X: test_x,
                                      Y: test_y}))
    print("Testing Precision:", \
        sess.run([prec], feed_dict={X: test_x,
                                      Y: test_y}))
    print("Testing Recall:", \
        sess.run([rec], feed_dict={X: test_x,
                                      Y: test_y}))
    print("Testing F-Score:", \
        sess.run([fscore], feed_dict={X: test_x,
                                      Y: test_y}))
    print("Testing True Positives & True Negatives:", \
        sess.run([TP, TN], feed_dict={X: test_x,
                                      Y: test_y}))
    print("Testing False Positives & False Negatives:", \
        sess.run([FP, FN], feed_dict={X: test_x,
                                      Y: test_y}))