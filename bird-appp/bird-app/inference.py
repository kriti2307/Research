"""Audio analysis and bird species prediction for the web app.

The model is the multi-label one trained by subset8/train_multilabel.py: one SIGMOID output
per species, so every species is judged on its own. A window can contain one species,
several species, or none. Nothing is ever forced: a species is reported only when its own
probability reaches the threshold.
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

import birdfeatures as bf


SAMPLE_RATE = bf.SR
WINDOW_SEC = bf.WINDOW_SEC
MAX_SECONDS = 120

# 0.5 is the threshold subset8/bird_out/metrics.json was measured at.
DEFAULT_THRESHOLD = 0.5


class AudioError(Exception):
    """The uploaded audio cannot be analysed."""


# --------------------------------------------------
# AUDIO BACKEND
# --------------------------------------------------

class AudioBackend(Protocol):
    def load(self, path: str, max_seconds: float):
        ...

    def features(self, chunk: np.ndarray):
        """5 s waveform -> the model's input array, shape birdfeatures.INPUT_SHAPE."""
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

    def features(self, chunk):
        # Same function the model was trained with (subset8/birdfeatures.py).
        return bf.chunk_to_features(chunk)

    # The two methods below only draw the spectrogram shown on the page.
    # The model does not see this image.
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

        if tuple(self.model.input_shape[1:]) != tuple(bf.INPUT_SHAPE):
            raise ValueError(
                f"Model expects input {tuple(self.model.input_shape[1:])} but "
                f"birdfeatures.py produces {tuple(bf.INPUT_SHAPE)}. "
                "Use a model trained with this birdfeatures.py."
            )

        # A softmax model would force exactly one species per window.
        activation = getattr(self.model.layers[-1], "activation", None)

        if getattr(activation, "__name__", "") != "sigmoid":
            raise ValueError(
                "The model's last layer must use a sigmoid activation (multi-label). "
                "This looks like a softmax model, which can only pick one species."
            )

    def predict(self, features: list[np.ndarray]) -> np.ndarray:

        batch = np.stack(features).astype("float32")

        return self.model.predict(
            batch,
            batch_size=32,
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
# SPECTROGRAM GENERATION (display only)
# --------------------------------------------------

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
    """Merge consecutive windows of one species into segments.

    windows: [{"species", "start", "end", "confidence"}, ...] in time order.
    """

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


def window_detections(
    window: dict,
    classes: list[str],
    threshold: float
):
    """Species whose own probability reaches the threshold in one window, strongest first.

    An empty list means "no species detected". There is deliberately no fallback to the
    highest-scoring species. Mirrored by detectionsAt() in static/segments.js.
    """

    found = [
        {"species": c, "confidence": window["probs"][c]}
        for c in classes
        if window["probs"][c] >= threshold
    ]

    found.sort(key=lambda d: -d["confidence"])

    return found


def segments_from_probs(
    windows: list[dict],
    classes: list[str],
    threshold: float
):
    """Per-species segments from windows that carry a probability for every species.

    Each species is merged on its own, so segments of different species can overlap in
    time (two birds calling together). Mirrored by segmentsFromProbs() in static/segments.js,
    which lets the page re-draw instantly when the threshold slider moves.
    """

    segments = []

    for c in classes:

        per_species = [
            {
                "species": c,
                "start": w["start"],
                "end": w["end"],
                "confidence": w["probs"][c]
            }
            for w in windows
        ]

        segments.extend(
            merge_segments(per_species, threshold)
        )

    segments.sort(
        key=lambda s: (s["start"], s["species"])
    )

    return segments


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
        threshold: float = DEFAULT_THRESHOLD,
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

        # Scale the WHOLE recording once, exactly like training did, so a quiet
        # stretch stays quiet instead of being stretched up to look like a call.
        y_model = bf.normalize_recording(y)

        features = [
            self.backend.features(
                bf.fix_length(y_model[s:e])
            )
            for s, e in spans
        ]

        # Run model
        with self._lock:

            probs = np.asarray(
                self.predictor.predict(features)
            )

        if probs.ndim != 2 or probs.shape[1] != len(self.classes):
            raise ValueError(
                "Model prediction shape does not match the class list."
            )

        # One probability per species per window. Each species is judged independently.
        windows = []

        for (s, e), p in zip(spans, probs):

            window = {

                "start": round(s / sr, 2),

                "end": round(
                    min(e, n) / sr,
                    2
                ),

                "probs": {
                    c: round(float(v), 4)
                    for c, v in zip(
                        self.classes,
                        p
                    )
                }
            }

            # May hold several species, or none.
            window["detections"] = window_detections(
                window,
                self.classes,
                threshold
            )

            windows.append(window)

        segments = segments_from_probs(
            windows,
            self.classes,
            threshold
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
