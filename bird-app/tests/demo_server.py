"""Preview the web page WITHOUT the trained model.

    python -m tests.demo_server

Needs only flask, numpy, scipy, matplotlib and pillow. It uses scripted
predictions, so every result is flagged as demo mode on the page. Upload a
WAV file (or the sample this script writes to tests/sample_demo.wav).
"""
import os

from app import create_app
from inference import Analyzer
from tests.fakes import FakeBackend, StubPredictor, make_wav_bytes

CLASSES = ["Barfly", "asbfly", "ashpri", "bkcbul"]
HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    sample = os.path.join(HERE, "sample_demo.wav")
    if not os.path.exists(sample):
        with open(sample, "wb") as f:
            f.write(make_wav_bytes(20))
    analyzer = Analyzer(FakeBackend(), StubPredictor(len(CLASSES)), CLASSES)
    port = int(os.environ.get("PORT", 5000))
    print(f"Demo server on http://localhost:{port}  (sample clip: {sample})")
    create_app(analyzer).run(host="127.0.0.1", port=port)
