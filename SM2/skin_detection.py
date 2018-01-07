import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import os

from SM2.skin_detection_model import load_skin_detection_model

learning_rate = 0.01
training_epochs = 5000

X, Y, y_, train_features, train_labels, test_features, test_labels = load_skin_detection_model()

init = tf.global_variables_initializer()

cost_function = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(y_), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1], dtype=float)
y_true, y_pred = None, None
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        if epoch % 50 == 0:
            print("Training Epoch " + str(epoch))

        _, cost = sess.run([optimizer, cost_function], feed_dict={X: train_features, Y: train_labels})
        cost_history = np.append(cost_history, cost)

    y_pred = sess.run(tf.argmax(y_, 1), feed_dict={X: test_features})
    y_true = sess.run(tf.argmax(test_labels, 1))

    save_path = saver.save(sess, os.path.abspath('.') + "/skin_detection_model.tf")


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

Sample Size: 24505
Precision: 0.998
Recall: 0.987
F-Score: 0.992
"""
