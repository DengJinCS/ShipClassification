
import librosa.display
import numpy as np
audio_path = "C:\\Users\MIRLAB\Desktop\piano.wav"

y, sr = librosa.load(audio_path)
librosa.feature.chroma_stft(y=y, sr=sr)


S = np.abs(librosa.stft(y))
chroma = librosa.feature.chroma_stft(S=S, sr=sr)

S = np.abs(librosa.stft(y, n_fft=4096))**2
chroma = librosa.feature.chroma_stft(S=S, sr=sr)

import matplotlib.pyplot as plt
plt.figure(figsize=(20, 10))
librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
plt.colorbar()
plt.title('Chromagram')
plt.tight_layout()
plt.show()

