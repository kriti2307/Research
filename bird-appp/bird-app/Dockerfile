FROM python:3.11-slim

# ffmpeg lets librosa read MP3/M4A uploads; libsndfile reads WAV/OGG/FLAC.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user id 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    MPLBACKEND=Agg \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860
WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
# One worker keeps a single copy of the model in memory; threads handle concurrent requests.
CMD ["sh", "-c", "gunicorn wsgi:app --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180"]
