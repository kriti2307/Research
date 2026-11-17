
import io
import json
import math
import os
from collections import defaultdict

import numpy as np
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

import birdfeatures as bf


SAMPLE_RATE = bf.SR
WINDOW_SEC = bf.WINDOW_SEC
MAX_SECONDS = 120


class AudioError(Exception):
    pass


class LibrosaBackend:

    def load(self, path, max_seconds=MAX_SECONDS):
        try:
            y, sr = librosa.load(
                path,
                sr=SAMPLE_RATE,
                mono=True,
                duration=max_seconds + 1
            )
        except Exception as exc:
            raise AudioError(f"Could not load audio: {exc}")

        if y is None or len(y) == 0:
            raise AudioError("The audio file is empty.")

        if len(y) / sr > max_seconds:
            raise AudioError(
                f"Audio exceeds the {max_seconds}-second limit."
            )

        return y.astype(np.float32), sr

    def mel_db(self, y, sr):
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            fmax=sr // 2
        )

        return librosa.power_to_db(
            mel,
            ref=np.max
        )

    def draw(self, mel_db, sr):
        fig, ax = plt.subplots(figsize=(10, 3))

        librosa.display.specshow(
            mel_db,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            ax=ax
        )

        ax.set_title("Audio spectrogram")
        fig.tight_layout()

        buffer = io.BytesIO()

        fig.savefig(
            buffer,
            format="png",
            dpi=120
        )

        plt.close(fig)
        buffer.seek(0)

        return buffer.getvalue()


class KerasPredictor:

    def __init__(self, model_path, classes):
        try:
            from tensorflow.keras.models import load_model

            self.model = load_model(
                model_path,
                compile=False
            )
        except Exception as exc:
            raise AudioError(f"Could not load model: {exc}")

        self.classes = classes

        input_shape = self.model.input_shape
        output_shape = self.model.output_shape

        if isinstance(input_shape, list):
            raise AudioError("Model has multiple inputs.")

        if isinstance(output_shape, list):
            raise AudioError("Model has multiple outputs.")

        if tuple(input_shape[1:]) != bf.INPUT_SHAPE:
            raise AudioError(
                f"Expected model input {bf.INPUT_SHAPE}, "
                f"got {input_shape[1:]}."
            )

        if output_shape[-1] != len(classes):
            raise AudioError(
                f"Model outputs {output_shape[-1]} classes, "
                f"but classes.json contains {len(classes)}."
            )

    def predict(self, features):
        return np.asarray(
            self.model.predict(features, verbose=0)
        )


def make_windows(
    n_samples,
    sr=SAMPLE_RATE,
    window_sec=WINDOW_SEC,
    hop_sec=None
):
    if hop_sec is None:
        hop_sec = window_sec

    if n_samples <= 0:
        return []

    window_samples = int(round(window_sec * sr))
    hop_samples = int(round(hop_sec * sr))

    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("Window and hop lengths must be positive.")

    if n_samples <= window_samples:
        return [(0, window_samples)]

    spans = []
    start = 0

    while start + window_samples < n_samples:
        spans.append(
            (start, start + window_samples)
        )
        start += hop_samples

    last_start = n_samples - window_samples

    if not spans or spans[-1][0] < last_start:
        spans.append(
            (last_start, n_samples)
        )

    return list(dict.fromkeys(spans))


def merge_segments(windows, threshold):
    grouped = defaultdict(list)

    # Group by species so detections merge independently.
    for window in windows:
        if window["confidence"] >= threshold:
            grouped[window["species"]].append(window)

    segments = []

    for species, species_windows in grouped.items():
        species_windows.sort(
            key=lambda item: item["start"]
        )

        current = None

        for window in species_windows:
            if (
                current is not None
                and window["start"] <= current["end"] + 0.011
            ):
                current["end"] = max(
                    current["end"],
                    window["end"]
                )

                current["_confs"].append(
                    window["confidence"]
                )

            else:
                if current is not None:
                    segments.append(current)

                current = {
                    "species": species,
                    "start": window["start"],
                    "end": window["end"],
                    "_confs": [window["confidence"]]
                }

        if current is not None:
            segments.append(current)

    result = []

    for segment in segments:
        confidence_values = segment["_confs"]

        result.append({
            "species": segment["species"],
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "mean_confidence": round(
                float(np.mean(confidence_values)), 4
            ),
            "peak_confidence": round(
                float(max(confidence_values)), 4
            )
        })

    return sorted(
        result,
        key=lambda item: item["start"]
    )


def summarize(segments):
    grouped = defaultdict(list)

    for segment in segments:
        grouped[segment["species"]].append(segment)

    summary = []

    for species, items in grouped.items():
        intervals = sorted(
            (item["start"], item["end"])
            for item in items
        )

        total = 0.0

        if intervals:
            start, end = intervals[0]

            for next_start, next_end in intervals[1:]:
                if next_start <= end:
                    end = max(end, next_end)
                else:
                    total += end - start
                    start, end = next_start, next_end

            total += end - start

        summary.append({
            "species": species,
            "detections": len(items),
            "seconds": round(total, 2),
            "total_seconds": round(total, 2),
            "mean_confidence": round(
                float(np.mean([
                    item["mean_confidence"]
                    for item in items
                ])), 4
            ),
            "peak_confidence": round(
                max(
                    item.get(
                        "peak_confidence",
                        item["mean_confidence"]
                    )
                    for item in items
                ), 4
            )
        })

    return sorted(
        summary,
        key=lambda item: item["seconds"],
        reverse=True
    )


def waveform_peaks(y, bins=1000):
    y = np.asarray(y, dtype=np.float32)

    if len(y) == 0:
        return []

    peak = float(np.max(np.abs(y)))

    if peak > 0:
        y = y / peak

    bins = max(1, int(bins))
    chunk_size = max(1, math.ceil(len(y) / bins))

    result = []

    for start in range(0, len(y), chunk_size):
        chunk = y[start:start + chunk_size]

        if len(chunk) == 0:
            continue

        result.append([
            round(float(np.min(chunk)), 4),
            round(float(np.max(chunk)), 4)
        ])

    return result


def render_display_spectrogram(backend, y, sr):
    try:
        mel = backend.mel_db(y, sr)
        return backend.draw(mel, sr)
    except Exception:
        return None


class Analyzer:

    def __init__(self, backend, predictor, classes):
        self.backend = backend
        self.predictor = predictor
        self.classes = classes

    def analyze(
        self,
        path,
        hop_sec=None,
        threshold=0.6,
        max_seconds=MAX_SECONDS
    ):
        y, sr = self.backend.load(path, max_seconds)

        y = np.asarray(y, dtype=np.float32)

        if len(y) == 0:
            raise AudioError("The audio file is empty.")

        duration = len(y) / sr

        if duration > max_seconds:
            raise AudioError(
                f"Audio exceeds the {max_seconds}-second limit."
            )

        y = bf.normalize_recording(y)
        duration = len(y) / sr

        spans = make_windows(
            len(y),
            sr,
            WINDOW_SEC,
            hop_sec
        )

        features = []

        for start_sample, end_sample in spans:
            chunk = y[start_sample:min(end_sample, len(y))]

            if len(chunk) < int(WINDOW_SEC * sr):
                chunk = np.pad(
                    chunk,
                    (0, int(WINDOW_SEC * sr) - len(chunk))
                )

            features.append(
                bf.chunk_to_features(chunk)
            )

        if features:
            batch = np.stack(features, axis=0)
            predictions = self.predictor.predict(batch)
        else:
            predictions = np.empty(
                (0, len(self.classes))
            )

        predictions = np.asarray(predictions)

        if predictions.shape != (
            len(spans),
            len(self.classes)
        ):
            raise AudioError(
                "Prediction output does not match the "
                "number of windows and classes."
            )

        window_results = []
        detected_windows = defaultdict(list)

        for index, (start_sample, end_sample) in enumerate(spans):
            start = start_sample / sr
            end = min(end_sample, len(y)) / sr

            probs = {
                species: round(
                    float(predictions[index, class_index]), 4
                )
                for class_index, species in enumerate(self.classes)
            }

            window_results.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "probs": probs
            })

            # Independent sigmoid scores: multiple species
            # can be detected in the same audio window.
            for species, confidence in probs.items():
                if confidence >= threshold:
                    detected_windows[species].append({
                        "species": species,
                        "start": start,
                        "end": end,
                        "confidence": confidence
                    })

        segments = []

        for species_windows in detected_windows.values():
            segments.extend(
                merge_segments(species_windows, threshold)
            )

        segments.sort(
            key=lambda item: item["start"]
        )

        summary = summarize(segments)

        spectrogram_png = render_display_spectrogram(
            self.backend,
            y,
            sr
        )

        return {
            "duration": round(duration, 2),
            "sample_rate": sr,
            "classes": self.classes,
            "windows": window_results,
            "segments": segments,
            "summary": summary,
            "waveform": waveform_peaks(y),
            "spectrogram_png": spectrogram_png
        }


def build_default_analyzer(model_dir=None):
    if model_dir is None:
        model_dir = os.path.join(
            os.path.dirname(__file__),
            "model"
        )

    model_path = os.path.join(
        model_dir,
        "bird_model.h5"
    )

    classes_path = os.path.join(
        model_dir,
        "classes.json"
    )

    with open(classes_path, "r", encoding="utf-8") as file:
        classes_data = json.load(file)

    # Support both list-based and dictionary-based classes.json.
    if isinstance(classes_data, dict):
        classes = classes_data["classes"]
    else:
        classes = classes_data

    predictor = KerasPredictor(
        model_path,
        classes
    )

    return Analyzer(
        LibrosaBackend(),
        predictor,
        classes
    )
