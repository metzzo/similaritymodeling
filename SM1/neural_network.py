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

def extract_features(filename):
    df = load_data(file=filename)

    training_set = []


    for index, row in df.iterrows():
        filename = row['name']
        jump1 = row['j1']
        jump2 = row['j2']

        target_directory = os.path.abspath("../../audio_dataset/") + '\\'
        target_file = target_directory + filename + '.wav'
        print("Extract data for " + target_file)

        y, sr = librosa.load(target_file)
        y, sr = (librosa.resample(y, sr, 8000), 8000)

        S = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=16000, hop_length=8000)
        print(S)


result_set = extract_features(filename='ground-truth.csv')
