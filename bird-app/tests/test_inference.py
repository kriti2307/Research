import json
import os
import unittest

import numpy as np

from inference import (SAMPLE_RATE, make_windows, merge_segments, render_model_image,
                       summarize, waveform_peaks)
from tests.fakes import FakeBackend

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


class ModelImageTests(unittest.TestCase):
    def test_image_has_the_shape_of_the_training_images(self):
        """Training PNGs are about 232 x 231 px RGB. This checks the matplotlib
        layout (3x3 in figure, axes off, tight bbox) gives the same size. The
        FakeBackend draws with pcolormesh instead of librosa's specshow, so this
        confirms the layout, not the colours or pixel values."""
        chunk = np.random.default_rng(1).standard_normal(5 * SAMPLE_RATE).astype(np.float32)
        img = render_model_image(FakeBackend(), chunk, SAMPLE_RATE)
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape[2], 3)
        self.assertTrue(228 <= img.shape[0] <= 234, img.shape)
        self.assertTrue(228 <= img.shape[1] <= 235, img.shape)


if __name__ == "__main__":
    unittest.main()
