from scipy.io import wavfile
from scipy import signal
from numpy.fft import fft

#from scipy.signal import spectrogram
import numpy as np
import os

from SM1.load_ground_truth import load_data



def extract_features(filename):
    df = load_data(file=filename)

    training_data = []
    training_label = []
    time_lookup = []

    for index, row in df.iterrows():
        filename = row['name']
        jump1 = row['j1']
        jump2 = row['j2']

        target_directory = os.path.abspath("../../audio_dataset/") + '\\'
        target_file = target_directory + filename + '.wav'
        print("Extract data for " + target_file)

        [fs, x] = wavfile.read(target_file)

        audio = np.array(x)
        left_audio = np.take(audio, [0], axis=1)
        #left_audio = left_audio[0*44100:17*44100][:]
        left_audio = np.array([a[0] for a in left_audio])

        # Feature Extraction

        #hist = {}
        for i in range(0, len(left_audio), int(44100/20)):
            chunk = np.array(left_audio[i:i + int(44100/10)].tolist())

            rms = np.sqrt(np.mean(np.square(chunk)))
            zero_crosses = len(np.nonzero(np.diff(chunk > 0))[0])

            if zero_crosses > 100 and rms > 3000:
                #print("Zero Crosses " + str(zero_crosses) + " RMS " + str(rms) + " at " + str(i / 44100))
                # hist[sec] = hist[sec] + 1 if sec in hist else 1

                sec = int(i / 44100)
                training_data.append([zero_crosses, rms])

                jump = False
                if jump1 >= 0 and (jump1 - 2 <= sec <= jump1 + 2):
                    jump = True
                if jump2 >= 0 and (jump2 - 2 <= sec <= jump2 + 2):
                    jump = True

                training_label.append(jump)
                time_lookup.append(sec)

    return (training_data, training_label, time_lookup)

"""
filter_hist = hist.copy()

for key, value in hist.items():
    if key - 1 in hist:
        filter_hist[key] += hist[key - 1]
        filter_hist[key - 1] = 0
"""









# audio_fft = fft(left_audio)

#, nfft=int(44100/25), scaling='spectrum',
"""f, t, Sxx = spectrogram(left_audio, fs=44100, window=('gaussian', 4.5))



plt.pcolor(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()"""