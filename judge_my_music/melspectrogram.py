import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

#loading the sound
y, sr = librosa.load("Eminem - Stan (Long Version) ft. Dido.mp3",sr=None)


stft = librosa.stft(y)

mel_spec = librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128, fmax = 8000)

mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

plt.figure(figsize=(10,4))
librosa.display.specshow(mel_spec_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000)


plt.colorbar(format="%+2.0f dB")
plt.title("Mel-Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Mel Frequency (Hz)")
plt.tight_layout()
plt.show()





