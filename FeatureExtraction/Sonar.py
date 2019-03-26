"""
  This file contents implementation of some sonar functions
"""
import numpy as np
import scipy.signal

from run_config_settings import *


def DemonAnalysis(sample, fs=SAMPLING_RATE, decimation_rate1=5, decimation_rate2=2, n_pts_fft=1024, overlap=0.5):
    # decimation in 2 steps - first step
    print("len of input signal:", len(sample))
    sample_decimate1 = scipy.signal.decimate(sample, decimation_rate1)
    fs_decimate1 = float(fs) / float(decimation_rate1)

    # decimation in 2 steps - second step
    sample_decimate2 = scipy.signal.decimate(sample_decimate1, decimation_rate1)
    fs_decimate2 = float(fs_decimate1) / float(decimation_rate2)

    n_pts_overlap = np.floor(n_pts_fft - fs_decimate2 * overlap)
    print("len of window:", n_pts_fft)
    f, t, Sxx = scipy.signal.spectrogram(sample_decimate2.T
                                         - np.mean(sample_decimate2),
                                         fs=fs_decimate2,
                                         window=scipy.signal.windows.hann(n_pts_fft),
                                         nperseg=n_pts_fft,
                                         noverlap=n_pts_overlap,
                                         nfft=n_pts_fft)
    data_demon = Sxx
    data_demon_abs = np.abs(data_demon)
    return data_demon_abs
