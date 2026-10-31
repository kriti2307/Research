"""Stand-ins for librosa and the Keras model, so the web layer can be tested
and previewed on a machine that has neither TensorFlow nor librosa.

These are NOT the real signal processing or the real model. Anything served
with them is flagged "demo": true and the page shows a banner saying so.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.io import wavfile

from inference import SAMPLE_RATE, AudioError


class FakeBackend:
    """WAV-only loader and a rough spectrogram built from scipy."""

    def load(self, path, max_seconds):
        sr, data = wavfile.read(path)  # raises ValueError for non-WAV data
        y = data.astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if np.issubdtype(data.dtype, np.integer):
            y = y / float(np.iinfo(data.dtype).max)
        if sr != SAMPLE_RATE:
            y = signal.resample_poly(y, SAMPLE_RATE, sr).astype(np.float32)
        if len(y) == 0:
            raise AudioError("The file contains no audio.")
        if len(y) / SAMPLE_RATE > max_seconds:
            raise AudioError(f"This recording is longer than {int(max_seconds)} seconds. Trim it and try again.")
        return y, SAMPLE_RATE

    def mel_db(self, y, sr):
        _, _, sxx = signal.spectrogram(y, fs=sr, nperseg=2048, noverlap=1536)
        mel = np.stack([g.mean(axis=0) for g in np.array_split(sxx, 128, axis=0)])
        mel = np.maximum(mel, 1e-10)
        return np.maximum(10 * np.log10(mel / mel.max()), -80.0)

    def draw(self, ax, s_db, sr):
        ax.pcolormesh(s_db, cmap="magma", shading="auto")


class StubPredictor:
    """Returns scripted predictions: plan is a list of (class_index, confidence)
    that is cycled through, one entry per window."""

    is_stub = True

    def __init__(self, n_classes: int, plan=None):
        self.n_classes = n_classes
        self.plan = plan or [(0, 0.9), (0, 0.8), (1, 0.7), (2, 0.4)]

    def predict(self, images):
        out = np.zeros((len(images), self.n_classes), dtype=np.float32)
        for i in range(len(images)):
            k, conf = self.plan[i % len(self.plan)]
            out[i, :] = (1.0 - conf) / (self.n_classes - 1)
            out[i, k] = conf
        return out


def make_wav_bytes(seconds: float, sr: int = SAMPLE_RATE) -> bytes:
    """A little synthetic 'recording': noise plus a few chirps."""
    import io

    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * sr)) / sr
    y = 0.05 * rng.standard_normal(len(t))
    for start, f0 in [(1.0, 3000), (4.5, 5200), (8.0, 2400)]:
        m = (t >= start) & (t < start + 1.2)
        y[m] += 0.5 * np.sin(2 * np.pi * (f0 + 600 * (t[m] - start)) * (t[m] - start))
    buf = io.BytesIO()
    wavfile.write(buf, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    return buf.getvalue()
