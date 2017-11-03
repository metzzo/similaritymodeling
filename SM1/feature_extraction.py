from scipy.io import wavfile
from scipy import signal
from numpy.fft import fft

from scipy.signal import spectrogram
import numpy as np
import os
import matplotlib.pyplot as plt
target_directory = os.path.abspath("../../audio_dataset/")
target_file = target_directory + '\\1_2015-10-03_13-42-32.mp4.wav' #'\\1_2015-10-03_13-42-32.mp4.wav'
print(target_file)

[fs, x] = wavfile.read(target_file)

audio = np.array(x)

left_audio = np.take(audio, [0], axis=1)
#left_audio = left_audio[0*44100:17*44100][:]
left_audio = np.array([a[0] for a in left_audio])
np.set_printoptions(threshold=np.nan)

print("Calculation")

hist = {}
for i in range(0, len(left_audio), int(44100/20)):
    chunk = np.array(left_audio[i:i + int(44100/10)].tolist())

    rms = np.sqrt(np.mean(np.square(chunk)))
    zero_crosses = len(np.nonzero(np.diff(chunk > 0))[0])

    if zero_crosses > 150 and rms >  :
        print("Zero Crosses " + str(zero_crosses) + " RMS " + str(rms) + " at " + str(i / 44100))

        sec = int(i/44100)
        hist[sec] = hist[sec] + 1 if sec in hist else 1

filter_hist = hist.copy()

for key, value in hist.items():
    if key - 1 in hist:
        filter_hist[key] += hist[key - 1]
        filter_hist[key - 1] = 0

print(filter_hist)



# audio_fft = fft(left_audio)

#, nfft=int(44100/25), scaling='spectrum',
"""f, t, Sxx = spectrogram(left_audio, fs=44100, window=('gaussian', 4.5))



plt.pcolor(t, f, Sxx)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()"""