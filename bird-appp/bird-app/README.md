---
title: Bird Call Detector
emoji: 🐦
sdk: docker
app_port: 7860
---

# Bird call detector

A small website for Workstream 2. You upload a bird recording, a backend runs
`bird_model.h5` on it, and the page shows which species it hears, with
confidence and timestamps, on a shared time axis with the waveform and spectrogram.

## What it does

1. The browser uploads the audio to `POST /api/predict`.
2. The server loads it at 22,050 Hz (same as `Dataset/dataset1.py`) and cuts it into
   5-second windows, one every 2.5 seconds.
3. Each window is drawn as a Mel spectrogram image with the same matplotlib/librosa
   calls as `dataset1.py`, resized to 128 x 128 the way `train.py` does
   (bilinear, divided by 255), and passed to the CNN.
4. Consecutive windows with the same species above the confidence threshold are merged
   into segments. The threshold slider on the page re-draws them instantly.

## Run it on your computer

```bash
cd bird-app
cp /path/to/Research/bird_model.h5 model/bird_model.h5   # your trained CNN
pip install -r requirements.txt
python app.py                                            # http://localhost:5000
```

Open the address, drop in a recording, press **Analyze recording**.

`model/classes.json` must list the classes in the same order the training folders
sort in (that is how `image_dataset_from_directory` numbers them). It currently holds
`Barfly, asbfly, ashpri, bkcbul`, copied from your `predict.py`. To show common names
on the page, fill in `display_names`, for example `{"asbfly": "Asian Brown Flycatcher"}`.

## Deploy so your professor can open a link

The whole site (page and API) is one container, so there is one thing to host.
The `Dockerfile` is ready for Hugging Face Spaces.

1. Create a Space at https://huggingface.co/new-space and choose **Docker** as the SDK.
   Check that Docker is selectable on your account before going further.
2. Put the contents of this folder in the Space's repository, including
   `model/bird_model.h5` (about 40 MB). The `README.md` header above is what tells
   Spaces to use Docker on port 7860.
3. Spaces builds the image and serves it at a public URL.

Notes from the hosting docs when this was written (October 2026):

- Hugging Face free CPU Spaces have 2 vCPU and 16 GB RAM and go to sleep when unused,
  so the first visit after a pause is slow. Open the link yourself shortly before the demo.
- Render's free web service has 512 MB RAM and spins down after 15 minutes without
  traffic. TensorFlow plus librosa will very likely not fit in 512 MB, so it is not a
  good fit for this app.
- If you split the page and API onto different domains, set `CORS_ORIGIN` on the
  server and `window.BIRD_API_BASE` in `static/index.html`.

## Tests

```bash
python -m unittest discover -s tests -t .       # needs scipy as well
node --test tests/segments.test.js
```

These use stand-ins for librosa and the model (`tests/fakes.py`). To see the page
without the trained model, run `python -m tests.demo_server`. It shows a yellow
banner and every detection is scripted, so never use it for real results.

## Known limits (worth stating in your report)

- The CNN has no "no bird" class. Every window gets one of the four species, which is
  why the page marks windows under the threshold as uncertain.
- Two species calling at once still produce a single best guess per window. That is
  Workstream 1's question.
- `Dataset/split.py` shuffles individual chunks, so chunks from one recording can end
  up in both train and test. The reported test accuracy is probably optimistic.
  Workstream 3 should split by recording.
