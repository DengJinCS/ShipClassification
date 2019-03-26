import sys
from math import floor

import Sonar
from FeatureExtraction.feature_extractor_base import FeatureExtractorBase

# demon parameters
decimation_rate1 = 25
decimation_rate2 = 25
n_pts_fft = 1024
window_size = 2  # in seconds
n_pts_windows = window_size * n_pts_fft
overlap = 0.5  # in seconds


# test from ubuntu



class DEMON(FeatureExtractorBase):
    def __init__(self):
        self.output_size = 0
        pass

    def extract_features(self, samples):
        processed_samples = []
        print("len of samples:", len(samples))
        for i in range(len(samples)):
            sample = samples[i]
            print("len of sample:", len(sample))
            sample_demon = Sonar.DemonAnalysis(sample)
            processed_samples.append(sample_demon)
            sys.stdout.write("\rExtracting features %d%%" % floor((i + 1) * (100/len(samples))))
            sys.stdout.flush()
        print()
        self.output_size = len(processed_samples[0])
        return processed_samples
