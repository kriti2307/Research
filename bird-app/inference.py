"""Turn an audio recording into per-window bird-species predictions.

The heavy lifting is split into three small, replaceable pieces:

* an audio backend   - loads audio, makes a Mel spectrogram (dB) and draws it
                       (LibrosaBackend, used in the real app)
* a predictor        - turns a batch of spectrogram images into class
                       probabilities (KerasPredictor, wraps bird_model.h5)
* Analyzer           - cuts the recording into windows, calls the two pieces
                       above and assembles the JSON that the API returns

The spectrogram images fed to the CNN are produced with exactly the same
matplotlib/librosa calls as Dataset/dataset1.py, and they are resized the same
way train.py resizes them (bilinear, to 128 x 128, then divided by 255), so the
model sees the same kind of input at prediction time as it did in training.
"""
from __future__ import annotations

import base64
import io
import json
import os
import threading
from typing import Protocol

import numpy as np
from matplotlib.figure import Figure
from PIL import Image

SAMPLE_RATE = 22050      # same as Dataset/dataset1.py
WINDOW_SEC = 5.0         # same chunk length as the training spectrograms
MODEL_IMG_SIZE = (128, 128)
MAX_SECONDS = 120        # longest recording the app accepts


class AudioError(Exception):
    """The uploaded audio cannot be analysed. The message is shown to the user."""


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class AudioBackend(Protocol):
    def load(self, path: str, max_seconds: float) -> tuple[np.ndarray, int]: ...
    def mel_db(self, y: np.ndarray, sr: int) -> np.ndarray: ...
    def draw(self, ax, s_db: np.ndarray, sr: int) -> None: ...


class LibrosaBackend:
    """Mirrors the preprocessing in Dataset/dataset1.py."""

    def load(self, path, max_seconds):
        import librosa

        # Read at most max_seconds + 1 so a huge upload cannot exhaust memory.
        y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=max_seconds + 1)
        if len(y) == 0:
            raise AudioError("The file contains no audio.")
        if len(y) / sr > max_seconds:
            raise AudioError(
                f"This recording is longer than {int(max_seconds)} seconds. "
                "Trim it and try again."
            )
        return y, sr

    def mel_db(self, y, sr):
        import librosa

        s = librosa.feature.melspectrogram(y=y, sr=sr)
        return librosa.power_to_db(s, ref=np.max)

    def draw(self, ax, s_db, sr):
        import librosa.display

        librosa.display.specshow(s_db, sr=sr, ax=ax)


class KerasPredictor:
    """Wraps bird_model.h5 (the CNN from train.py)."""

    is_stub = False

    def __init__(self, model_path: str, classes: list[str]):
        import tensorflow as tf

        self._tf = tf
        self.model = tf.keras.models.load_model(model_path, compile=False)
        n_out = self.model.output_shape[-1]
        if n_out != len(classes):
            raise ValueError(
                f"The model has {n_out} outputs but classes.json lists {len(classes)} classes. "
                "They must match, in the same order as the training folders."
            )

    def predict(self, images: list[np.ndarray]) -> np.ndarray:
        tf = self._tf
        # train.py loads images with image_dataset_from_directory, which resizes
        # with tf.image.resize (bilinear), then rescales by 1/255.
        batch = np.stack(
            [
                tf.image.resize(im.astype("float32"), MODEL_IMG_SIZE, method="bilinear").numpy()
                for im in images
            ]
        )
        return self.model.predict(batch / 255.0, verbose=0)


# --------------------------------------------------------------------------
# Windowing, spectrogram rendering, summaries
# --------------------------------------------------------------------------
def make_windows(n_samples: int, sr: int, window_sec: float, hop_sec: float):
    """Return [(start_sample, end_sample), ...]. A clip shorter than one window
    gives a single (padded) window; the tail of a longer clip is covered by one
    extra window anchored to the end."""
    win = int(round(window_sec * sr))
    hop = int(round(hop_sec * sr))
    if n_samples <= win:
        return [(0, win)]
    starts = list(range(0, n_samples - win + 1, hop))
    if starts[-1] + win < n_samples - int(0.5 * sr):
        starts.append(n_samples - win)
    return [(s, s + win) for s in starts]


def render_model_image(backend: AudioBackend, chunk: np.ndarray, sr: int) -> np.ndarray:
    """One 5-second chunk -> RGB image, drawn exactly like Dataset/dataset1.py."""
    fig = Figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    backend.draw(ax, backend.mel_db(chunk, sr), sr)
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"))


def render_display_spectrogram(backend: AudioBackend, y: np.ndarray, sr: int) -> str:
    """Whole-clip spectrogram for the web page (base64 PNG, edge to edge, no axes)."""
    width, height = 1600, 320
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    backend.draw(ax, backend.mel_db(y, sr), sr)
    ax.set_aspect("auto")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def waveform_peaks(y: np.ndarray, bins: int = 1200) -> list[list[float]]:
    """Min/max per bin, scaled so the loudest sample is +-1."""
    n = len(y)
    bins = max(1, min(bins, n))
    edges = np.linspace(0, n, bins + 1).astype(int)[:-1]
    mins = np.minimum.reduceat(y, edges)
    maxs = np.maximum.reduceat(y, edges)
    peak = max(float(np.max(np.abs(y))), 1e-9)
    return [[round(float(a) / peak, 3), round(float(b) / peak, 3)] for a, b in zip(mins, maxs)]


def merge_segments(windows: list[dict], threshold: float) -> list[dict]:
    """Join consecutive windows with the same species and confidence >= threshold."""
    segs: list[dict] = []
    cur = None
    for w in windows:
        if w["confidence"] < threshold:
            cur = None
            continue
        if cur and cur["species"] == w["species"] and w["start"] <= cur["end"] + 0.011:
            cur["end"] = max(cur["end"], w["end"])
            cur["_confs"].append(w["confidence"])
        else:
            cur = {"species": w["species"], "start": w["start"], "end": w["end"],
                   "_confs": [w["confidence"]]}
            segs.append(cur)
    return [
        {
            "species": s["species"],
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "mean_confidence": round(sum(s["_confs"]) / len(s["_confs"]), 4),
            "peak_confidence": round(max(s["_confs"]), 4),
        }
        for s in segs
    ]


def _union_seconds(intervals: list[tuple[float, float]]) -> float:
    total, cur_end = 0.0, None
    cur_start = None
    for a, b in sorted(intervals):
        if cur_end is None or a > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = a, b
        else:
            cur_end = max(cur_end, b)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def summarize(segments: list[dict]) -> list[dict]:
    """One row per species heard, most time first."""
    by_species: dict[str, list[dict]] = {}
    for s in segments:
        by_species.setdefault(s["species"], []).append(s)
    rows = [
        {
            "species": sp,
            "segments": len(segs),
            "seconds": round(_union_seconds([(s["start"], s["end"]) for s in segs]), 2),
            "peak_confidence": max(s["peak_confidence"] for s in segs),
        }
        for sp, segs in by_species.items()
    ]
    return sorted(rows, key=lambda r: -r["seconds"])


# --------------------------------------------------------------------------
# Analyzer
# --------------------------------------------------------------------------
class Analyzer:
    def __init__(self, backend: AudioBackend, predictor, classes: list[str],
                 display_names: dict[str, str] | None = None):
        self.backend = backend
        self.predictor = predictor
        self.classes = list(classes)
        self.labels = {c: (display_names or {}).get(c) or c for c in self.classes}
        self._lock = threading.Lock()  # one prediction at a time keeps memory predictable

    @property
    def is_demo(self) -> bool:
        return bool(getattr(self.predictor, "is_stub", False))

    def analyze(self, path: str, hop_sec: float = 2.5, threshold: float = 0.6,
                max_seconds: float = MAX_SECONDS) -> dict:
        try:
            y, sr = self.backend.load(path, max_seconds)
        except AudioError:
            raise
        except Exception as exc:  # decoder errors differ by format and library
            raise AudioError(
                "Could not read this file as audio. Try a WAV, MP3, OGG or FLAC file."
            ) from exc

        n = len(y)
        spans = make_windows(n, sr, WINDOW_SEC, hop_sec)
        win = int(round(WINDOW_SEC * sr))
        images = []
        for s, e in spans:
            chunk = y[s:e]
            if len(chunk) < win:
                chunk = np.pad(chunk, (0, win - len(chunk)))
            images.append(render_model_image(self.backend, chunk, sr))

        with self._lock:
            probs = np.asarray(self.predictor.predict(images))

        windows = []
        for (s, e), p in zip(spans, probs):
            k = int(np.argmax(p))
            windows.append({
                "start": round(s / sr, 2),
                "end": round(min(e, n) / sr, 2),
                "species": self.classes[k],
                "confidence": round(float(p[k]), 4),
                "probs": {c: round(float(v), 4) for c, v in zip(self.classes, p)},
            })

        segments = merge_segments(windows, threshold)
        return {
            "demo": self.is_demo,
            "duration": round(n / sr, 2),
            "sample_rate": sr,
            "window_sec": WINDOW_SEC,
            "hop_sec": hop_sec,
            "threshold": threshold,
            "classes": self.classes,
            "labels": self.labels,
            "windows": windows,
            "segments": segments,
            "summary": summarize(segments),
            "waveform": waveform_peaks(y),
            "spectrogram_png": render_display_spectrogram(self.backend, y, sr),
        }


def build_default_analyzer(model_path: str | None = None, classes_path: str | None = None) -> Analyzer:
    """Load the real model. Paths can also be set with MODEL_PATH / CLASSES_PATH."""
    here = os.path.dirname(os.path.abspath(__file__))
    model_path = model_path or os.environ.get("MODEL_PATH", os.path.join(here, "model", "bird_model.h5"))
    classes_path = classes_path or os.environ.get("CLASSES_PATH", os.path.join(here, "model", "classes.json"))
    if not os.path.exists(model_path):
        raise RuntimeError(
            f"Model file not found at {model_path}. Copy bird_model.h5 into the model/ folder."
        )
    with open(classes_path, encoding="utf-8") as f:
        cfg = json.load(f)
    classes = cfg["classes"]
    return Analyzer(LibrosaBackend(), KerasPredictor(model_path, classes), classes,
                    cfg.get("display_names"))
