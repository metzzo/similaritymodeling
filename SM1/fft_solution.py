from scipy.io import wavfile
from scipy import signal
from numpy.fft import fft

#from scipy.signal import spectrogram
import scipy
import numpy as np
import os
import matplotlib.pyplot as plt

from SM1.load_ground_truth import load_data

import librosa
import librosa.display
import pickle

def classify(filename):
    df = load_data(file=filename)

    result_set = []

    count = 0
    for index, row in df.iterrows():
        filename = row['name']
        ground_truth_jump1 = row['j1']
        ground_truth_jump2 = row['j2']

        target_directory = os.path.abspath("../../audio_dataset/") + '\\'
        target_file = target_directory + filename + '.wav'
        print("Extract data for " + target_file)

        y, sr = librosa.load(target_file)
        S = librosa.stft(y, win_length=1024, n_fft=1024, window=scipy.signal.hanning)

        absS = np.abs(S)
        absS = np.transpose(absS)
        sum_bins = []
        for bin in absS:
            sum_bins.append(sum(bin[350:]))

        second_bins = []
        sec = 44100.0/513.0
        bin_index = 0
        while bin_index < len(sum_bins):
            second_bins.append(sum(sum_bins[int(bin_index):int(bin_index + sec)]))

            bin_index += sec

        jumps = []
        for index, value in enumerate(second_bins):
            if value > 30:
                jumps.append((index, value))

        final_jumps = []
        while len(jumps) > 0:
            jumps = sorted(jumps, key=lambda jump: -jump[1])

            first = jumps[0]
            delete_jump = []
            for jump in jumps[1:]:
                delta = abs(first[0] - jump[0])
                if delta <= 2:
                    first = (first[0], first[1] + jump[1])
                    delete_jump.append(jump)
            jumps = [a for a in jumps if a not in delete_jump]

            final_jumps.append(first)
            jumps = jumps[1:]

        final_jumps = sorted(final_jumps, key=lambda jump: jump[1])
        final_jumps = final_jumps[:2]
        print(final_jumps)

        for jump in final_jumps:
            correct = (ground_truth_jump1 - 3 <= jump[0] <= ground_truth_jump1 + 3) or (ground_truth_jump2 - 3 <= jump[0] <= ground_truth_jump2 + 3)
            row = (filename, jump[0], jump[1], correct)
            result_set.append(row)
            print(row)

        count += 1
        print(str(count) + " / 70")

    return result_set


result_set = classify(filename='ground-truth.csv')

with open('result_set.file', 'wb') as fp:
    pickle.dump(result_set, fp)

print("-----------")
print(result_set)
