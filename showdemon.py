import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from scipy import signal

#hello from Mac
# demon parameters
decimation_rate1 = 10
decimation_rate2 = 10
n_pts_fft = 2048
# window_size = 1 # in seconds
# n_pts_windows = window_size*n_pts_fft
overlap = 0.5  # in seconds

datapath = "F:\PythonCode\ShipClassification\Data\ShipsEar\B2.wav"

# bandpass:5k-25k
sample, SAMPLING_RATE = librosa.load(datapath, sr=None)
print("SAMPLING_RATE:", SAMPLING_RATE)


def butter_highpass_filter(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_highpass(data, cutoff, fs, order=5):
    b, a = butter_highpass_filter(cutoff, fs, order=order)
    y = scipy.signal.filtfilt(b, a, data)
    return y


# sample = butter_highpass(sample,500,SAMPLING_RATE,11)
'''
D = librosa.amplitude_to_db(np.abs(librosa.stft(sample)), ref=np.max)
librosa.display.specshow(D, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')
plt.show()
'''

# decimation in 2 steps - first step
sample_decimate1 = scipy.signal.decimate(sample, decimation_rate1)
fs_decimate1 = float(SAMPLING_RATE) / float(decimation_rate1)

# decimation in 2 steps - second step
sample_decimate2 = scipy.signal.decimate(sample_decimate1, decimation_rate1)
fs_decimate2 = float(fs_decimate1) / float(decimation_rate2)

n_pts_overlap = np.floor(n_pts_fft - fs_decimate2 * overlap)
f, t, Sxx = scipy.signal.spectrogram(sample_decimate2.T
                                     - np.mean(sample_decimate2),
                                     fs=fs_decimate2,
                                     window=scipy.signal.windows.hann(n_pts_fft),
                                     nperseg=n_pts_fft,
                                     noverlap=n_pts_overlap,
                                     nfft=n_pts_fft)
data_demon = Sxx
data_demon_abs = np.abs(data_demon)
data_freq = f
data_time = t

plt.pcolormesh(t, f, Sxx)
# plt.colorbar(format='%+2.0f dB')
plt.title('DEMON spectrogram')
plt.show()

print(len(data_demon), "data_demon", data_demon)
# print(len(data_freq),"data_freq:",data_freq)
# print(len(data_time),"data_time:",data_time)
