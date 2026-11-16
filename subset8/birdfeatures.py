"""Audio -> model features. Shared by training AND the web app so they never drift apart.

Each 5-second window becomes a (128, 216, 3) array with three channels:
  0  mel_db   mel spectrogram in dB (built on the Fourier transform + mel filterbank)
  1  stft_db  plain linear-frequency Fourier (STFT) spectrogram in dB, 1024 bins averaged down to 128 rows
  2  pcen     per-channel energy normalisation of the mel spectrogram (robust to background noise / gain)

To try another feature later (e.g. wavelet), add a channel here, bump N_CHANNELS and
retrain. Nothing else needs to change.
"""
import numpy as np

SR = 22050
WINDOW_SEC = 5.0
N_FFT = 2048
HOP = 512
N_MELS = 128
N_SAMPLES = int(round(SR * WINDOW_SEC))      # 110250
N_FRAMES = 1 + N_SAMPLES // HOP              # 216
N_CHANNELS = 3
INPUT_SHAPE = (N_MELS, N_FRAMES, N_CHANNELS)
CHANNELS = ["mel_db", "stft_db", "pcen"]


def normalize_recording(y):
    """Scale a WHOLE recording so its loud calls peak near 1.0.

    Done per recording, never per 5 s chunk: a chunk with only background noise must stay
    quiet. (The old pipeline normalised every chunk to its own maximum, which made empty
    windows look like loud bird calls.)
    """
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0:
        return y
    peak = float(np.percentile(np.abs(y), 99.9))
    return (y / max(peak, 1e-6)).astype(np.float32)


def fix_length(chunk):
    """Pad with zeros / cut so a chunk is exactly N_SAMPLES long."""
    chunk = np.asarray(chunk, dtype=np.float32)
    if len(chunk) < N_SAMPLES:
        chunk = np.pad(chunk, (0, N_SAMPLES - len(chunk)))
    return chunk[:N_SAMPLES]


def chunk_to_features(y):
    """5-second waveform (already passed through normalize_recording) -> float32 (128, 216, 3)."""
    import librosa

    y = fix_length(y)

    mel_mag = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=1.0
    )
    mel_db = librosa.power_to_db(mel_mag ** 2, ref=1.0, top_db=None)

    stft_pow = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP)) ** 2      # (1025, T)
    stft_db = librosa.power_to_db(stft_pow[:1024], ref=1.0, top_db=None)
    stft_db = stft_db.reshape(N_MELS, 1024 // N_MELS, -1).mean(axis=1)

    pcen = librosa.pcen(mel_mag * (2 ** 31), sr=SR, hop_length=HOP)

    feats = np.stack([mel_db, stft_db, pcen], axis=-1).astype(np.float32)
    assert feats.shape == INPUT_SHAPE, f"unexpected feature shape {feats.shape}"
    return feats