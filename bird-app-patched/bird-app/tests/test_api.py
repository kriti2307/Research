import base64
import io
import unittest

from app import create_app
from inference import Analyzer
from tests.fakes import FakeBackend, StubPredictor, make_wav_bytes

CLASSES = ["Barfly", "asbfly", "ashpri", "bkcbul"]


def make_client(plan=None):
    analyzer = Analyzer(FakeBackend(), StubPredictor(len(CLASSES), plan), CLASSES,
                        {"asbfly": "Asian Brown Flycatcher"})
    app = create_app(analyzer)
    return app, app.test_client()


def upload(client, name="clip.wav", data=None, **form):
    data = make_wav_bytes(12) if data is None else data
    return client.post("/api/predict",
                       data={"file": (io.BytesIO(data), name), **{k: str(v) for k, v in form.items()}},
                       content_type="multipart/form-data")


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app, self.client = make_client()

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(body["classes"], CLASSES)
        self.assertTrue(body["demo"])

    def test_index_page_is_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn(b"<html", r.data)

    def test_predict_returns_windows_segments_and_media(self):
        r = upload(self.client)
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        d = r.get_json()
        self.assertTrue(d["demo"])
        self.assertAlmostEqual(d["duration"], 12.0, places=1)
        self.assertEqual(len(d["windows"]), 4)
        self.assertEqual(d["windows"][0]["start"], 0.0)
        self.assertAlmostEqual(d["windows"][-1]["end"], 12.0, places=1)
        self.assertEqual(set(d["windows"][0]["probs"]), set(CLASSES))
        self.assertEqual(d["labels"]["asbfly"], "Asian Brown Flycatcher")
        self.assertEqual(d["labels"]["ashpri"], "ashpri")
        self.assertEqual(len(d["waveform"]), 1200)
        self.assertTrue(base64.b64decode(d["spectrogram_png"]).startswith(b"\x89PNG"))
        self.assertEqual(d["filename"], "clip.wav")

    def test_scripted_predictions_become_the_expected_segments(self):
        # plan per window: Barfly .9, Barfly .8, asbfly .7, ashpri .4 (uncertain)
        d = upload(self.client).get_json()
        got = [(s["species"], s["start"], s["end"]) for s in d["segments"]]
        self.assertEqual(got, [("Barfly", 0.0, 7.5), ("asbfly", 5.0, 10.0)])
        self.assertEqual([row["species"] for row in d["summary"]], ["Barfly", "asbfly"])

    def test_threshold_parameter_changes_segments(self):
        d = upload(self.client, threshold=0.95).get_json()
        self.assertEqual(d["segments"], [])
        self.assertEqual(d["threshold"], 0.95)

    def test_missing_file(self):
        r = self.client.post("/api/predict", data={}, content_type="multipart/form-data")
        self.assertEqual(r.status_code, 400)
        self.assertIn("error", r.get_json())

    def test_unsupported_extension(self):
        r = upload(self.client, name="notes.txt", data=b"hello")
        self.assertEqual(r.status_code, 415)

    def test_undecodable_audio(self):
        r = upload(self.client, name="broken.wav", data=b"this is not audio")
        self.assertEqual(r.status_code, 422)
        self.assertIn("Could not read", r.get_json()["error"])

    def test_recording_too_long(self):
        self.app.config["MAX_SECONDS"] = 5
        r = upload(self.client)
        self.assertEqual(r.status_code, 422)
        self.assertIn("longer than 5 seconds", r.get_json()["error"])

    def test_bad_parameters(self):
        self.assertEqual(upload(self.client, hop=0.2).status_code, 400)
        self.assertEqual(upload(self.client, hop="abc").status_code, 400)
        self.assertEqual(upload(self.client, threshold=1.5).status_code, 400)

    def test_file_too_large(self):
        self.app.config["MAX_CONTENT_LENGTH"] = 2000
        r = upload(self.client)
        self.assertEqual(r.status_code, 413)
        self.assertIn("larger than", r.get_json()["error"])

    def test_unexpected_failure_is_a_clean_500(self):
        class Boom:
            is_stub = True

            def predict(self, images):
                raise RuntimeError("secret internal detail")

        analyzer = Analyzer(FakeBackend(), Boom(), CLASSES)
        client = create_app(analyzer).test_client()
        r = upload(client)
        self.assertEqual(r.status_code, 500)
        self.assertNotIn("secret", r.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
