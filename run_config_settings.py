'''
Python 3.4.3
librosa 0.4.3
Tensorflow 0.6.0, upgraded to 0.12.0
'''
from os import path

DATA_PATH = path.dirname(path.realpath(__file__)) + "/Data/ShipsEar"
LOG_PATH = "/tmp/tensorflow/ShipClassification"

NR_OF_CLASSES = 2
# INCLUDED_VESSELS = ["A", "B", "C", "D"]
INCLUDED_VESSELS = ["0", "1"]
# INCLUDED_VESSELS = ["A", "B", "C", "D", "E", "P", "G", "H", "I", "J", "K"]
'''
INCLUDED_VESSELS = ["Dredger",
                    "Fishboat",
                    "Motorboat",
                    "Musselboat",
                    "Oceanliner",
                    "Passengers",
                    "Pilotship",
                    "RORO",
                    "Sailboat",
                    "Trawler",
                    "Tugboat"]
'''

# INCLUDED_VESSELS = ["speedboat", "tanker"]
# INCLUDED_VESSELS = ["ferry", "speedboat", "tanker", "sub"]
TEST_PERCENTAGE = 0.1
SAMPLING_RATE = 2000
NR_OF_FILES = 30000
# 85 files in total
FILE_LENTH = 2
SAVE_MODEL = ''
SAMPLES_PR_FILE = 2
SAMPLE_LENGTH = 1  # sec

FFT_WINDOW_SIZE = 1024
N_MFCC = 20

# Only one of these can be true at once
# Describes what part of the dataset is being used as test data
USE_WHOLE_FILE_AS_TEST = False # This is the only available option for recurrent networks
USE_END_OF_FILE_AS_TEST = True
USE_RANDOM_SAMPLES_AS_TEST = False

if USE_WHOLE_FILE_AS_TEST + USE_END_OF_FILE_AS_TEST + USE_RANDOM_SAMPLES_AS_TEST > 1:
    raise ValueError

# Recurrent NN
RELATED_STEPS = 10

NOISE_ENABLED = False
NR_OF_NOISY_SAMPLES_PR_SAMPLE = 2

RESET_PICKLE = False
MOCK_DATA = False
USE_PRELOADED_DATA = True


# if NR_OF_CLASSES != len(INCLUDED_VESSELS):
#     raise ValueError

# if SAMPLES_PR_FILE * SAMPLE_LENGTH > 10:
#     raise ValueError

BATCH_SIZE = 10
# EPOCS = ceil(NR_OF_FILES * SAMPLES_PR_FILE / BATCH_SIZE) # To ensure all samples being used in training
EPOCS = 100
LEARNING_RATE = 0.0001
BIAS_ENABLED = True
DROPOUT_RATE = 0.9

ACTIVATION_FUNCTIONS = "2 2"
HIDDEN_LAYERS = "256 128"
CNN_FILTERS = "5 5"
CNN_CHANNELS = "32 64"
DCL_SIZE = 128
PADDING = 2
POOLING_STRIDE = 2
POOLING_FILTER_SIZE = 2