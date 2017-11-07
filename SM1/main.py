from SM1.feature_extraction import extract_features
import pickle
from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np

EXTRACT_FEATURES = False
EXTRACT_TEST_FEATURES = True

#np.set_printoptions(threshold=np.nan)
training_data = None
training_label = None

if EXTRACT_FEATURES:
    training_data, training_label, _ = extract_features(filename='ground-truth.csv')

    with open('training_data.file', 'wb') as fp:
        pickle.dump(training_data, fp)

    with open('label.file', 'wb') as fp:
        pickle.dump(training_label, fp)
else:
    file = open("training_data.file", 'rb')
    training_data = pickle.load(file)
    file.close()

    file = open("label.file", 'rb')
    training_label = pickle.load(file)
    file.close()

print(training_data)
print(training_label)

true_data = []
false_data = []
for index, value in enumerate(training_label):
    if value:
        true_data.append(training_data[index])
    else:
        false_data.append(training_data[index])

x_vals = [x[0] for x in false_data]
y_vals = [x[1] for x in false_data]
plt.scatter(x=x_vals, y=y_vals)

x_vals = [x[0] for x in true_data]
y_vals = [x[1] for x in true_data]

plt.scatter(x=x_vals, y=y_vals)

plt.xlabel("ZCR")
plt.ylabel("RMS")
plt.legend(loc=2)
plt.show()

clf = svm.SVC()
svm = clf.fit(training_data, training_label)


test_data = None
test_label = None
test_time = None
if EXTRACT_TEST_FEATURES:
    test_data, test_label, test_time = extract_features(filename='test_set.csv')
    
    with open('test_data.file', 'wb') as fp:
        pickle.dump(test_data, fp)
    with open('test_label.file', 'wb') as fp:
        pickle.dump(test_label, fp)
    with open('test_times.file', 'wb') as fp:
        pickle.dump(test_time, fp)
else:
    file = open("test_data.file", 'rb')
    test_data = pickle.load(file)
    file.close()

    file = open("test_label.file", 'rb')
    test_label = pickle.load(file)
    file.close()
    
    file = open("test_times.file", 'rb')
    test_time = pickle.load(file)
    file.close()
    
predicted_label = svm.predict(test_data)

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

print("Testing phase")

for index, value in enumerate(test_data):
    predicted = predicted_label[index]
    actual = test_label[index]

    if predicted and not actual:
        false_positives += 1

    if not predicted and not actual:
        true_negatives += 1

    if predicted and actual:
        true_positives += 1

    if not predicted and actual:
        false_negatives += 1

    #print("Predicted: " + str(predicted) + " vs actual: " + str(actual) + " at " + str(test_time[index]))

recall = true_positives / (true_positives + false_negatives)
precision = true_positives / (true_positives + false_positives)
accuracy = (true_positives + true_negatives) / (true_positives + false_positives + true_negatives + false_negatives)
print("Recall " + str(recall))
print("Precision " + str(precision))
print("Accuracy " + str(accuracy))
