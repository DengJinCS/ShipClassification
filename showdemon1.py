import librosa
import librosa.display as dis
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal


def band_pass(data, fs, low, high, order=5):
    nyq = 0.5 * fs
    normalized_low = low / nyq
    normalized_high = high / nyq
    sos = signal.butter(order, [normalized_low, normalized_high], btype='bandpass', output='sos')
    y = signal.sosfilt(sos, data)
    return y


def test(sample, fs, low=2000, high=20000, order=9):
    sample = band_pass(sample, fs=fs, low=low, high=high, order=order)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(sample)), ref=np.max)
    dis.specshow(D, y_axis='log', sr=fs, x_axis='s')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-frequency power spectrogram')
    plt.show()


def DemonAnalysis(data, fs, low=2000, high=20000, DEMON_rate=1024, order=9):
    sample = band_pass(data, fs=fs, low=low, high=high, order=order)
    # Band Pass Filter
    h_signal = signal.hilbert(sample)
    envlope = np.sqrt(h_signal * h_signal + sample * sample)

    # window
    length = int(len(envlope) * DEMON_rate / fs)
    arv_envlope = np.empty(length, dtype=float)
    for i in range(length):
        sum = 0
        for j in range(int(fs / DEMON_rate)):
            sum += envlope[i * int(fs / DEMON_rate) + j] ** 2
            print(i)
        arv_envlope[i] = np.sqrt(sum / int(fs / DEMON_rate))
    print("Rate of Env", DEMON_rate)
    print("len of Arv_Envlope", len(arv_envlope))
    # Envlope by Hilbert

    plt.plot(arv_envlope)
    plt.title('arv-Envlope')
    plt.show()

    # data_demon = librosa.stft(arv_envlope)
    # D_demon = librosa.amplitude_to_db(np.abs(data_demon),ref=np.max)
    overlap = np.floor(DEMON_rate / 4)
    n_pts_overlap = np.floor(2048 - DEMON_rate * 0.25)
    # print("Len of D_demon",len(D_demon))
    f, t, Sxx = scipy.signal.spectrogram(arv_envlope.T - np.mean(arv_envlope),
                                         fs=DEMON_rate,
                                         window=scipy.signal.windows.hann(1024))
    abs_D = np.abs(Sxx)
    plt.pcolormesh(t, f, Sxx)
    plt.title('DEMON spectrogram')
    plt.show()

    # print(len(data_demon), "data_demon", data_demon)


datapath = "F:\PythonCode\ShipClassification\Data\ShipsEar\shipA_1.wav"
sample, rate = librosa.load(datapath, sr=None)
print("RATE of sample=", rate)
test(sample, fs=rate, low=2000, high=20000, order=11)
DemonAnalysis(sample, fs=rate, low=2000, high=20000, DEMON_rate=2048, order=9)
