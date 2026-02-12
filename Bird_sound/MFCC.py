import numpy as np
import librosa 
import librosa.display
import matplotlib.pyplot as plt

#Load
y , sr = librosa.load("Eminem - Stan (Long Version) ft. Dido.mp3")

mfccs = librosa.feature.mfcc(y = y, sr = sr, n_mfcc = 13, n_fft = 2048, hop_length = 512)


plt.figure(figsize=(10,4))


librosa.display.specshow(mfccs, sr = sr, x_axis= "time")
#Plotting 
plt.colorbar()
plt.title("MFCC")
plt.tight_layout()
plt.show()
