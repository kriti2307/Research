
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from inference import (
    SAMPLE_RATE,
    Analyzer,
    AudioError,
    make_windows,
    merge_segments,
    summarize,
    waveform_peaks
)
from tests.fakes import FakeBackend, make_wav_bytes


class FakePredictor:
    is_stub = True

    def predict(self, features):
        outputs = np.zeros((len(features), 8), dtype=np.float32)

        # Two species detected in the same window
        outputs[:, 0] = 0.85
        outputs[:, 1] = 0.72

        return outputs


CLASSES = [
    "asbfly",
    "ashpri1",
    "barfly1",
    "barswa",
    "bcnher",
    "bkcbul1",
    "bkwsti",
    "brnhao1"
]


class WindowTests(unittest.TestCase):

    def test_short_clip_gives_one_padded_window(self):
        spans = make_windows(
            3 * SAMPLE_RATE,
            SAMPLE_RATE,
            5.0,
            2.5
        )
        self.assertEqual(spans, [(0, 5 * SAMPLE_RATE)])

    def test_exact_window_length(self):
        spans = make_windows(
            5 * SAMPLE_RATE,
            SAMPLE_RATE,
            5.0,
            2.5
        )
        self.assertEqual(len(spans), 1)

    def test_tail_is_covered(self):
        n = 12 * SAMPLE_RATE

        spans = make_windows(
            n,
            SAMPLE_RATE,
            5.0,
            2.5
        )

        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], n)
        self.assertEqual(len(spans), 4)

    def test_non_overlapping_hop(self):
        spans = make_windows(
            15 * SAMPLE_RATE,
            SAMPLE_RATE,
            5.0,
            5.0
        )

        self.assertEqual(
            [s for s, _ in spans],
            [0, 5 * SAMPLE_RATE, 10 * SAMPLE_RATE]
        )


class SegmentTests(unittest.TestCase):

    def test_multiple_species_merge_independently(self):
        windows = [
            {
                "start": 0.0,
                "end": 5.0,
                "species": "A",
                "confidence": 0.9
            },
            {
                "start": 0.0,
                "end": 5.0,
                "species": "B",
                "confidence": 0.8
            },
            {
                "start": 2.5,
                "end": 7.5,
                "species": "A",
                "confidence": 0.85
            }
        ]

        result = merge_segments(windows, 0.6)

        self.assertEqual(len(result), 2)

        by_species = {
            item["species"]: item
            for item in result
        }

        self.assertEqual(by_species["A"]["end"], 7.5)
        self.assertEqual(by_species["B"]["end"], 5.0)

    def test_below_threshold_is_not_detected(self):
        windows = [
            {
                "start": 0.0,
                "end": 5.0,
                "species": "A",
                "confidence": 0.4
            }
        ]

        self.assertEqual(
            merge_segments(windows, 0.6),
            []
        )

    def test_summary(self):
        segments = [
            {
                "species": "A",
                "start": 0.0,
                "end": 5.0,
                "mean_confidence": 0.8,
                "peak_confidence": 0.9
            },
            {
                "species": "B",
                "start": 0.0,
                "end": 5.0,
                "mean_confidence": 0.7,
                "peak_confidence": 0.8
            }
        ]

        result = summarize(segments)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["species"], "A")
        self.assertEqual(result[0]["seconds"], 5.0)


class WaveformTests(unittest.TestCase):

    def test_peaks_are_scaled_and_sized(self):
        y = (
            np.sin(np.linspace(0, 200, 50000))
            .astype(np.float32) * 0.2
        )

        peaks = waveform_peaks(y, bins=500)

        self.assertEqual(len(peaks), 500)

        flat = [v for p in peaks for v in p]

        self.assertAlmostEqual(
            max(abs(v) for v in flat),
            1.0,
            places=2
        )

    def test_tiny_input(self):
        result = waveform_peaks(
            np.array([0.1, -0.1], dtype=np.float32),
            bins=1200
        )

        self.assertEqual(len(result), 2)


class AnalyzerTests(unittest.TestCase):

    def test_analyzer_returns_multiple_species(self):
        analyzer = Analyzer(
            FakeBackend(),
            FakePredictor(),
            CLASSES
        )

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:
            f.write(make_wav_bytes(5))
            path = f.name

        try:
            fake_features = np.zeros(
                (128, 216, 3),
                dtype=np.float32
            )

            with patch(
                "inference.bf.chunk_to_features",
                return_value=fake_features
            ):
                result = analyzer.analyze(
                    path,
                    threshold=0.6
                )

            self.assertEqual(len(result["windows"]), 1)

            scores = result["windows"][0]["probs"]

            self.assertEqual(scores["asbfly"], 0.85)
            self.assertEqual(scores["ashpri1"], 0.72)

            detected = {
                s["species"]
                for s in result["segments"]
            }

            self.assertIn("asbfly", detected)
            self.assertIn("ashpri1", detected)

        finally:
            os.remove(path)

    def test_no_species_above_threshold_means_no_segments(self):
        class LowScorePredictor:
            is_stub = True

            def predict(self, features):
                return np.full(
                    (len(features), 8),
                    0.1,
                    dtype=np.float32
                )

        analyzer = Analyzer(
            FakeBackend(),
            LowScorePredictor(),
            CLASSES
        )

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as f:
            f.write(make_wav_bytes(5))
            path = f.name

        try:
            fake_features = np.zeros(
                (128, 216, 3),
                dtype=np.float32
            )

            with patch(
                "inference.bf.chunk_to_features",
                return_value=fake_features
            ):
                result = analyzer.analyze(
                    path,
                    threshold=0.6
                )

            self.assertEqual(result["segments"], [])
            self.assertEqual(result["summary"], [])

        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
