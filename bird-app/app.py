"""Flask app: serves the web page and the /api/predict endpoint.

Run locally:   python app.py
Production:    gunicorn wsgi:app   (see Dockerfile)
"""
from __future__ import annotations

import os
import tempfile

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from inference import MAX_SECONDS, AudioError

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".ogg", ".oga", ".opus", ".flac", ".m4a", ".aac", ".webm"}
MAX_UPLOAD_MB = 25


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def create_app(analyzer=None) -> Flask:
    """Build the app. Tests pass in their own analyzer; otherwise the real model is loaded."""
    app = Flask(__name__, static_folder="static", static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
    app.config["MAX_SECONDS"] = MAX_SECONDS

    if analyzer is None:
        from inference import build_default_analyzer

        analyzer = build_default_analyzer()
    app.config["ANALYZER"] = analyzer

    # Only needed if the web page is hosted on a different domain than this API.
    cors_origin = os.environ.get("CORS_ORIGIN")
    if cors_origin:

        @app.after_request
        def add_cors(response):
            response.headers["Access-Control-Allow-Origin"] = cors_origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return response

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "classes": analyzer.classes, "demo": analyzer.is_demo})

    @app.post("/api/predict")
    def predict():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            return _error("Choose an audio file to analyze.", 400)

        ext = os.path.splitext(upload.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return _error("This file type is not supported. Use WAV, MP3, OGG, FLAC or M4A.", 415)

        try:
            hop = float(request.form.get("hop", 2.5))
            threshold = float(request.form.get("threshold", 0.6))
        except ValueError:
            return _error("hop and threshold must be numbers.", 400)
        if not 1.0 <= hop <= 5.0:
            return _error("hop must be between 1 and 5 seconds.", 400)
        if not 0.0 <= threshold <= 1.0:
            return _error("threshold must be between 0 and 1.", 400)

        fd, tmp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        try:
            upload.save(tmp_path)
            result = analyzer.analyze(
                tmp_path, hop_sec=hop, threshold=threshold,
                max_seconds=app.config["MAX_SECONDS"],
            )
            result["filename"] = upload.filename
            return jsonify(result)
        except AudioError as exc:
            return _error(str(exc), 422)
        except Exception:  # keep the details in the server log, not in the response
            app.logger.exception("Analysis failed")
            return _error("The analysis failed on the server. Try again, or try a shorter file.", 500)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_exc):
        return _error(f"That file is larger than {MAX_UPLOAD_MB} MB.", 413)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    create_app().run(host="0.0.0.0", port=port, debug=False)
