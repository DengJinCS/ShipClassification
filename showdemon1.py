import librosa
import librosa.display as dis
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import signal


def butter_highpass(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    y = scipy.signal.filtfilt(b, a, data)
    return y


def test(sample, fs, cutoff=1000):
    sample = butter_highpass(sample, cutoff, fs=fs, order=9)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(sample)), ref=np.max)
    dis.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Linear-frequency power spectrogram')
    plt.show()


def DemonAnalysis(data, fs, DEMON_rate=1024):
    # sample = butter_highpass(data, cutoff, fs=fs, order=11)
    # Band Pass Filter
    h_signal = signal.hilbert(data)
    envlope = np.abs(h_signal)

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

    data_demon = librosa.stft(arv_envlope)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(data_demon)),
                             sr=DEMON_rate,
                             hop_length=512,
                             y_axis='log',
                             x_axis='s')
    plt.title('DEMON spectrogram')
    plt.colorbar(format='%+2.0f dB')
    plt.show()

    # print(len(data_demon), "data_demon", data_demon)


datapath = "F:\PythonCode\ShipClassification\Data\ShipsEar\shipA.wav"
sample, rate = librosa.load(datapath, sr=None)
print("RATE of sample=", rate)
test(sample, rate)
DemonAnalysis(sample, fs=rate)
