import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = "../Bird_sound"
OUTPUT_PATH = "../spectrograms"

SAMPLE_RATE = 22050
CHUNK_DURATION = 5
SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_DURATION

print("Reading from:", DATASET_PATH)

for root, dirs, files in os.walk(DATASET_PATH):

    cls = os.path.basename(root)

    if root == DATASET_PATH:
        continue

    os.makedirs(os.path.join(OUTPUT_PATH, cls), exist_ok=True)

    for file in files:

        print("👉 Checking file:", file)

        if file.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):

            file_path = os.path.join(root, file)

            try:
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)

                num_chunks = max(1, len(signal) // SAMPLES_PER_CHUNK)

                for i in range(num_chunks):
                    start = i * SAMPLES_PER_CHUNK
                    end = start + SAMPLES_PER_CHUNK

                    chunk = signal[start:end]

                    if len(chunk) < SAMPLES_PER_CHUNK:
                        chunk = np.pad(chunk, (0, SAMPLES_PER_CHUNK - len(chunk)))

                    plt.figure(figsize=(3, 3))

                    S = librosa.feature.melspectrogram(y=chunk, sr=sr)
                    S_dB = librosa.power_to_db(S, ref=np.max)

                    librosa.display.specshow(S_dB, sr=sr)

                    plt.axis('off')

                    save_path = os.path.join(
                        OUTPUT_PATH,
                        cls,
                        f"{file.split('.')[0]}_chunk{i}.png"
                    )

                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()

                    print("✅ Saved:", save_path)

            except Exception as e:
                print("❌ Error:", file, e)

print("🔥 DONE")
