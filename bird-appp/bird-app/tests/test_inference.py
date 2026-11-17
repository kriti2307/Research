import json
import os
import unittest

import numpy as np

from inference import (SAMPLE_RATE, KerasPredictor, make_windows, merge_segments,
                       segments_from_probs, summarize, waveform_peaks, window_detections)

HERE = os.path.dirname(os.path.abspath(__file__))


class WindowTests(unittest.TestCase):
    def test_short_clip_gives_one_padded_window(self):
        spans = make_windows(3 * SAMPLE_RATE, SAMPLE_RATE, 5.0, 2.5)
        self.assertEqual(spans, [(0, 5 * SAMPLE_RATE)])

    def test_exact_window_length(self):
        spans = make_windows(5 * SAMPLE_RATE, SAMPLE_RATE, 5.0, 2.5)
        self.assertEqual(len(spans), 1)

    def test_tail_is_covered_by_an_end_anchored_window(self):
        n = 12 * SAMPLE_RATE
        spans = make_windows(n, SAMPLE_RATE, 5.0, 2.5)
        self.assertEqual(spans[0][0], 0)
        self.assertEqual(spans[-1][1], n)
        self.assertEqual(len(spans), 4)

    def test_non_overlapping_hop(self):
        spans = make_windows(15 * SAMPLE_RATE, SAMPLE_RATE, 5.0, 5.0)
        self.assertEqual([s for s, _ in spans], [0, 5 * SAMPLE_RATE, 10 * SAMPLE_RATE])


class SegmentTests(unittest.TestCase):
    def setUp(self):
        with open(os.path.join(HERE, "fixture_windows.json")) as f:
            self.fx = json.load(f)

    def test_merge_matches_hand_worked_fixture(self):
        got = merge_segments(self.fx["windows"], self.fx["threshold"])
        self.assertEqual(got, self.fx["segments"])

    def test_summary_matches_hand_worked_fixture(self):
        self.assertEqual(summarize(self.fx["segments"]), self.fx["summary"])

    def test_nothing_above_threshold(self):
        self.assertEqual(merge_segments(self.fx["windows"], 0.99), [])
        self.assertEqual(summarize([]), [])


class WaveformTests(unittest.TestCase):
    def test_peaks_are_scaled_and_sized(self):
        y = np.sin(np.linspace(0, 200, 50_000)).astype(np.float32) * 0.2
        peaks = waveform_peaks(y, bins=500)
        self.assertEqual(len(peaks), 500)
        flat = [v for p in peaks for v in p]
        self.assertAlmostEqual(max(abs(v) for v in flat), 1.0, places=2)

    def test_tiny_input(self):
        self.assertEqual(len(waveform_peaks(np.array([0.1, -0.1], dtype=np.float32), bins=1200)), 2)


class MultiLabelTests(unittest.TestCase):
    """Several birds in one recording, and "no bird" when nothing is confident."""

    def setUp(self):
        with open(os.path.join(HERE, "fixture_multilabel.json")) as f:
            self.fx = json.load(f)

    def test_each_window_lists_every_species_over_threshold(self):
        got = [window_detections(w, self.fx["classes"], self.fx["threshold"])
               for w in self.fx["windows"]]
        self.assertEqual(got, self.fx["detections"])

    def test_two_birds_in_one_window(self):
        both = window_detections(self.fx["windows"][1], self.fx["classes"], 0.5)
        self.assertEqual([d["species"] for d in both], ["A", "B"])

    def test_no_bird_is_allowed_and_there_is_no_best_guess_fallback(self):
        quiet = self.fx["windows"][4]
        self.assertEqual(window_detections(quiet, self.fx["classes"], 0.5), [])
        # even though one species is technically the "highest", it must not be reported
        self.assertEqual(segments_from_probs([quiet], self.fx["classes"], 0.5), [])

    def test_segments_of_different_species_can_overlap(self):
        got = segments_from_probs(self.fx["windows"], self.fx["classes"], self.fx["threshold"])
        self.assertEqual(got, self.fx["segments"])
        a, b = got[0], got[1]
        self.assertLess(b["start"], a["end"])  # B starts while A is still going

    def test_summary_matches_fixture(self):
        self.assertEqual(summarize(self.fx["segments"]), self.fx["summary"])

    def test_raising_the_threshold_only_removes_species(self):
        got = segments_from_probs(self.fx["windows"], self.fx["classes"], 0.78)
        self.assertEqual([s["species"] for s in got], ["A"])
        self.assertEqual(segments_from_probs(self.fx["windows"], self.fx["classes"], 0.99), [])


class ModelGuardTests(unittest.TestCase):
    """The app must refuse a softmax model, because softmax can only ever pick one bird."""

    @classmethod
    def setUpClass(cls):
        try:
            import tensorflow as tf  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("TensorFlow is not installed")

    def _save(self, activation, n_out=8):
        import tempfile

        import tensorflow as tf

        import birdfeatures as bf

        m = tf.keras.Sequential([
            tf.keras.layers.Input(bf.INPUT_SHAPE),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(n_out, activation=activation),
        ])
        path = os.path.join(tempfile.mkdtemp(), "m.h5")
        m.save(path)
        return path

    def test_softmax_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sigmoid"):
            KerasPredictor(self._save("softmax"), list("abcdefgh"))

    def test_sigmoid_model_is_accepted(self):
        KerasPredictor(self._save("sigmoid"), list("abcdefgh"))

    def test_class_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outputs"):
            KerasPredictor(self._save("sigmoid", n_out=8), list("abc"))


if __name__ == "__main__":
    unittest.main()
