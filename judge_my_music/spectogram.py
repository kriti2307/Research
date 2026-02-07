import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

 #loading the sound
y , sr = librosa.load("Eminem - Stan (Long Version) ft. Dido.mp3")

#plotting
plt.figure(figsize=(10,4))

stft = librosa.stft(y)

spectogram = librosa.amplitude_to_db(np.abs(stft))

#librosa to plot
librosa.display.specshow(spectogram, sr = sr, x_axis='time', y_axis ='log')

plt.colorbar()
plt.title("Spectogram(db)")
plt.tight_layout()
plt.show()
