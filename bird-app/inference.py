
"""Audio analysis and bird species prediction for the web app."""

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


SAMPLE_RATE = 22050
WINDOW_SEC = 5.0
MODEL_IMG_SIZE = (128, 128)
MAX_SECONDS = 120


class AudioError(Exception):
    """The uploaded audio cannot be analysed."""


# --------------------------------------------------
# AUDIO BACKEND
# --------------------------------------------------

class AudioBackend(Protocol):
    def load(self, path: str, max_seconds: float):
        ...

    def mel_db(self, y: np.ndarray, sr: int):
        ...

    def draw(self, ax, s_db: np.ndarray, sr: int):
        ...


class LibrosaBackend:

    def load(self, path, max_seconds):
        import librosa

        y, sr = librosa.load(
            path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=max_seconds + 1
        )

        if len(y) == 0:
            raise AudioError("The file contains no audio.")

        if len(y) / sr > max_seconds:
            raise AudioError(
                f"Recording exceeds {int(max_seconds)} seconds."
            )

        return y, sr

    def mel_db(self, y, sr):
        import librosa

        s = librosa.feature.melspectrogram(
            y=y,
            sr=sr
        )

        return librosa.power_to_db(s, ref=np.max)

    def draw(self, ax, s_db, sr):
        import librosa.display

        librosa.display.specshow(
            s_db,
            sr=sr,
            ax=ax
        )


# --------------------------------------------------
# MODEL PREDICTOR
# --------------------------------------------------

class KerasPredictor:

    is_stub = False

    def __init__(self, model_path: str, classes: list[str]):
        import tensorflow as tf

        self._tf = tf

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        n_out = self.model.output_shape[-1]

        if n_out != len(classes):
            raise ValueError(
                f"Model has {n_out} outputs but "
                f"classes.json contains {len(classes)} classes."
            )

    def predict(self, images: list[np.ndarray]) -> np.ndarray:

        batch = np.stack([
            self._tf.image.resize(
                im.astype("float32"),
                MODEL_IMG_SIZE,
                method="bilinear"
            ).numpy()
            for im in images
        ])

        return self.model.predict(
            batch / 255.0,
            verbose=0
        )


# --------------------------------------------------
# AUDIO WINDOWING
# --------------------------------------------------

def make_windows(
    n_samples: int,
    sr: int,
    window_sec: float,
    hop_sec: float
):

    win = int(round(window_sec * sr))
    hop = int(round(hop_sec * sr))

    if n_samples <= win:
        return [(0, win)]

    starts = list(range(0, n_samples - win + 1, hop))

    if starts[-1] + win < n_samples - int(0.5 * sr):
        starts.append(n_samples - win)

    return [(s, s + win) for s in starts]


# --------------------------------------------------
# SPECTROGRAM GENERATION
# --------------------------------------------------

def render_model_image(
    backend: AudioBackend,
    chunk: np.ndarray,
    sr: int
):

    fig = Figure(figsize=(3, 3))
    ax = fig.add_subplot(111)

    backend.draw(
        ax,
        backend.mel_db(chunk, sr),
        sr
    )

    ax.axis("off")

    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0
    )

    buf.seek(0)

    with Image.open(buf) as im:
        return np.asarray(im.convert("RGB"))


def render_display_spectrogram(
    backend: AudioBackend,
    y: np.ndarray,
    sr: int
):

    width, height = 1600, 320

    fig = Figure(
        figsize=(width / 100, height / 100),
        dpi=100
    )

    ax = fig.add_axes([0, 0, 1, 1])

    backend.draw(
        ax,
        backend.mel_db(y, sr),
        sr
    )

    ax.set_aspect("auto")
    ax.axis("off")

    buf = io.BytesIO()

    fig.savefig(
        buf,
        format="png",
        dpi=100
    )

    return base64.b64encode(
        buf.getvalue()
    ).decode("ascii")


# --------------------------------------------------
# WAVEFORM
# --------------------------------------------------

def waveform_peaks(y: np.ndarray, bins: int = 1200):

    n = len(y)
    bins = max(1, min(bins, n))

    edges = np.linspace(
        0,
        n,
        bins + 1
    ).astype(int)[:-1]

    mins = np.minimum.reduceat(y, edges)
    maxs = np.maximum.reduceat(y, edges)

    peak = max(
        float(np.max(np.abs(y))),
        1e-9
    )

    return [
        [
            round(float(a) / peak, 3),
            round(float(b) / peak, 3)
        ]
        for a, b in zip(mins, maxs)
    ]


# --------------------------------------------------
# MERGE PREDICTIONS INTO SEGMENTS
# --------------------------------------------------

def merge_segments(
    windows: list[dict],
    threshold: float
):

    segments = []
    current = None

    for w in windows:

        if w["confidence"] < threshold:
            current = None
            continue

        if (
            current
            and current["species"] == w["species"]
            and w["start"] <= current["end"] + 0.011
        ):

            current["end"] = max(
                current["end"],
                w["end"]
            )

            current["_confs"].append(
                w["confidence"]
            )

        else:

            current = {
                "species": w["species"],
                "start": w["start"],
                "end": w["end"],
                "_confs": [w["confidence"]]
            }

            segments.append(current)

    return [
        {
            "species": s["species"],
            "start": round(s["start"], 2),
            "end": round(s["end"], 2),
            "mean_confidence": round(
                sum(s["_confs"]) / len(s["_confs"]),
                4
            ),
            "peak_confidence": round(
                max(s["_confs"]),
                4
            )
        }
        for s in segments
    ]


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

def _union_seconds(intervals):

    total = 0.0
    current_end = None
    current_start = None

    for a, b in sorted(intervals):

        if current_end is None or a > current_end:

            if current_end is not None:
                total += current_end - current_start

            current_start = a
            current_end = b

        else:
            current_end = max(current_end, b)

    if current_end is not None:
        total += current_end - current_start

    return total


def summarize(segments: list[dict]):

    by_species = {}

    for segment in segments:

        by_species.setdefault(
            segment["species"],
            []
        ).append(segment)

    rows = []

    for species, species_segments in by_species.items():

        rows.append({
            "species": species,
            "segments": len(species_segments),

            "seconds": round(
                _union_seconds([
                    (s["start"], s["end"])
                    for s in species_segments
                ]),
                2
            ),

            "peak_confidence": max(
                s["peak_confidence"]
                for s in species_segments
            )
        })

    return sorted(
        rows,
        key=lambda r: -r["seconds"]
    )


# --------------------------------------------------
# MAIN ANALYZER
# --------------------------------------------------

class Analyzer:

    def __init__(
        self,
        backend: AudioBackend,
        predictor,
        classes: list[str],
        display_names: dict[str, str] | None = None
    ):

        self.backend = backend
        self.predictor = predictor
        self.classes = list(classes)

        self.labels = {
            c: (display_names or {}).get(c) or c
            for c in self.classes
        }

        self._lock = threading.Lock()

    @property
    def is_demo(self):
        return bool(
            getattr(self.predictor, "is_stub", False)
        )

    def analyze(
        self,
        path: str,
        hop_sec: float = 2.5,
        threshold: float = 0.25,
        max_seconds: float = MAX_SECONDS
    ) -> dict:

        # Load audio
        try:

            y, sr = self.backend.load(
                path,
                max_seconds
            )

        except AudioError:
            raise

        except Exception as exc:

            raise AudioError(
                "Could not read this audio file. "
                "Try WAV, MP3, OGG or FLAC."
            ) from exc

        n = len(y)

        # Create overlapping windows
        spans = make_windows(
            n,
            sr,
            WINDOW_SEC,
            hop_sec
        )

        win = int(round(WINDOW_SEC * sr))

        images = []

        for s, e in spans:

            chunk = y[s:e]

            if len(chunk) < win:

                chunk = np.pad(
                    chunk,
                    (0, win - len(chunk))
                )

            images.append(
                render_model_image(
                    self.backend,
                    chunk,
                    sr
                )
            )

        # Run model
        with self._lock:

            probs = np.asarray(
                self.predictor.predict(images)
            )

        if probs.ndim != 2 or probs.shape[1] != len(self.classes):
            raise ValueError(
                "Model prediction shape does not match the class list."
            )

        # Multiple candidate predictions per window
        windows = []

        per_species_windows = {
            c: [] for c in self.classes
        }

        for (s, e), p in zip(spans, probs):

            detected = [
                i
                for i, score in enumerate(p)
                if float(score) >= threshold
            ]

            # Always retain the strongest prediction
            if not detected:

                detected = [
                    int(np.argmax(p))
                ]

            detections = []

            for k in detected:

                confidence = float(p[k])
                species = self.classes[k]

                detections.append({
                    "species": species,
                    "confidence": round(
                        confidence,
                        4
                    )
                })

                per_species_windows[species].append({

                    "start": round(s / sr, 2),

                    "end": round(
                        min(e, n) / sr,
                        2
                    ),

                    "species": species,

                    "confidence": confidence
                })

            best = int(np.argmax(p))

            windows.append({

                "start": round(s / sr, 2),

                "end": round(
                    min(e, n) / sr,
                    2
                ),

                "species": self.classes[best],

                "confidence": round(
                    float(p[best]),
                    4
                ),

                "detections": detections,

                "probs": {
                    c: round(float(v), 4)
                    for c, v in zip(
                        self.classes,
                        p
                    )
                }
            })

        # Create separate segments for each species
        segments = []

        for species_windows in per_species_windows.values():

            segments.extend(
                merge_segments(
                    species_windows,
                    threshold
                )
            )

        segments.sort(
            key=lambda x: x["start"]
        )

        # Final response
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

            "spectrogram_png": render_display_spectrogram(
                self.backend,
                y,
                sr
            )
        }


# --------------------------------------------------
# LOAD MODEL AND BUILD ANALYZER
# --------------------------------------------------

def build_default_analyzer(
    model_path: str | None = None,
    classes_path: str | None = None
) -> Analyzer:

    here = os.path.dirname(
        os.path.abspath(__file__)
    )

    model_path = (
        model_path
        or os.environ.get("MODEL_PATH")
        or os.path.join(
            here,
            "model",
            "bird_model.h5"
        )
    )

    classes_path = (
        classes_path
        or os.environ.get("CLASSES_PATH")
        or os.path.join(
            here,
            "model",
            "classes.json"
        )
    )

    if not os.path.exists(model_path):

        raise RuntimeError(
            f"Model file not found at {model_path}"
        )

    with open(
        classes_path,
        encoding="utf-8"
    ) as f:

        cfg = json.load(f)

    classes = cfg["classes"]

    return Analyzer(

        LibrosaBackend(),

        KerasPredictor(
            model_path,
            classes
        ),

        classes,

        cfg.get("display_names")
    )
