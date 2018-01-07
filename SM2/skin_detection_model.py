import tensorflow as tf
import os
import pandas as pd
import cv2
import pickle
import numpy as np

def load_skin_detection_model():
    with tf.variable_scope('skin'):
        def extract_features(file):
            if not os.path.isfile(file + '.file'):
                def to_hsv(row):
                    b = row[0]
                    g = row[1]
                    r = row[2]

                    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]

                    row[0] = hsv[0]
                    row[1] = hsv[1]
                    row[2] = hsv[2]

                    row = row.append(pd.Series([0], index=[4]))
                    if row[3] == 1:
                        row[3] = 1
                        row[4] = 0
                    else:
                        row[3] = 0
                        row[4] = 1

                    return row

                df = pd.read_csv(file, sep='\t')
                df = df.apply(to_hsv, axis=1)
                df.columns = ("h", "s", "v", "skin", "not_skin")
                df = df.sample(frac=1).reset_index(drop=True)

                with open(file + '.file', 'wb') as fp:
                    pickle.dump(df, fp)
            else:
                file = open(file + ".file", 'rb')
                df = pickle.load(file)
                file.close()

            return df

        def split_data(data):
            features = data.as_matrix(columns=("h", "s", "v"))
            labels = data.as_matrix(columns=["skin", "not_skin"])
            return features, labels

        ground_truth = extract_features(file="Skin_NonSkin.txt")

        num_of_training = int(len(ground_truth) * 0.1)

        test_truth = ground_truth[0:num_of_training]
        ground_truth = ground_truth[:num_of_training]

        train_features, train_labels = split_data(ground_truth)
        test_features, test_labels = split_data(test_truth)

        n_dim = train_features.shape[1]
        n_classes = 2
        n_hidden_units_one = 50
        n_hidden_units_two = 100
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

