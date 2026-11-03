import librosa
import librosa.display
import numpy as np 
import matplotlib.pyplot as plt

y, sr = librosa.load("Eminem - Stan (Long Version) ft. Dido.mp3")


print("Audio Loaded")
print("Sample Rate: Measurements per second", sr)
print("Audio Length: ",len(y)/sr)

#waveform plot

plt.figure(figsize=(10,4))

librosa.display.waveshow(y, sr=sr,color="red")

plt.title("Waveform")
plt.xlabel("Time")
plt.ylabel("Amplitude")

plt.tight_layout()
plt.show()



