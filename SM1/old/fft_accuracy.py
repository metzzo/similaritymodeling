import pickle

from SM1.load_ground_truth import load_data

file = open("result_set.file", 'rb')
result_set = pickle.load(file)
file.close()

total = 0
correct = 0
for row in result_set:
    if row[3]:
        correct += 1
