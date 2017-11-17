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

    plots = []

    result_set = []

    for index, row in df.iterrows():
        filename = row['name']
        jump1 = row['j1']
        jump2 = row['j2']

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
            result_set.append((filename, jump[0], jump[1]))

        #print(second_bins)
        #plots.append(plt.plot(second_bins, label=filename)[0])
        #plt.legend(handles=plots)
        #plt.ylabel('Sum')
        #plt.show()
    
    return result_set

        """
        
        librosa.display.specshow(librosa.amplitude_to_db(S,
                                                         ref=np.max),
                                 y_axis='linear', x_axis='time', cmap='gray_r')
        plt.title('Power spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()
        """
        """
        [fs, x] = wavfile.read(target_file)

        audio = np.array(x)
        left_audio = np.take(audio, [0], axis=1)
        #left_audio = left_audio[0*44100:17*44100][:]
        left_audio = np.array([a[0] for a in left_audio])
        """
        # Feature Extraction

        #hist = {}
        """
        
    training_data = []
    training_label = []
    time_lookup = []

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
                if jump1 >= 0 and (jump1 - 0.25 <= sec <= jump1 + 0.25):
                    jump = True
                if jump2 >= 0 and (jump2 - 0.25 <= sec <= jump2 + 0.25):
                    jump = True

                training_label.append(jump)
                time_lookup.append(sec
        
        # FFT auf 1 sekunden Segments

        step_size = 1024
        window = np.hamming(1024)
        #for i in range(0, len(left_audio), int(step_size)):
        for i in range(44100*14, 44100*17, int(step_size)):
            chunk = np.array(left_audio[i:i + int(step_size)].tolist())

            audio_fft = fft(chunk * window)
            audio_fft = [abs(a) for a in audio_fft]
            #audio_fft = [0 if abs(a) < 1000 else abs(a) for a in audio_fft]
            slice_sum = []
            for j in range(5000, 44100, 500):
                slice = audio_fft[j:j+500]

                slice_sum.append(sum(slice))
            total_sum = sum(slice_sum)
            print("Second " + str(i/44100))
            print("Slice Sum " + str(slice_sum))
            print("Total Sum " + str(total_sum))
        """




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