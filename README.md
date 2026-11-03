# Bird Species Detection from Audio (MFCC / Mel-Spectrogram + ML)

A project-based research study on identifying bird species from field recordings.
Audio is converted to Mel-spectrograms and classified with a CNN (4 species) or a
small neural network (2 species), then extended to overlapping calls, noise, and a
web app.

**Author:** Kriti Saraogi, B.Tech CSE, Manipal University Jaipur
**Data:** recordings from [xeno-canto](https://xeno-canto.org) (files are named by their `XC` id, e.g. `XC134896`)

---

## Species

| Folder code | Species | Used in |
|---|---|---|
| `asbfly` | Asian Brown Flycatcher | all experiments |
| `brnhao` | Brown Hawk-Owl | binary + multi-label experiments |
| `ashpri` | Ashy Prinia | 4-species CNN |
| `bkcbul` | Black-crested Bulbul | 4-species CNN |
| `Barfly` | (see note) | 4-species CNN |

> `Barfly` is a folder name from the original dataset; fill in the full species name here.

---

## Pipeline

```
Audio (.ogg) -> 22,050 Hz mono -> 5 s chunks -> Mel-spectrogram (dB)
             -> image (128x128) -> classifier -> species
```

---

## Experiments

The repo grew experiment by experiment, so the folders are mixed. This is the map of
what belongs to what.

### Exp 0 - Signal-processing trials
Learning the basics on a single recording: waveform, spectrogram, mel-spectrogram, MFCC.

| Path | What it is |
|---|---|
| `Trial/waveform.py`, `spectrogram.py`, `melspectrogram.py`, `MFCC.py` | one plot each |
| `Trial/Bird.ogg` | test bird recording (same file as `XC134896 (1).ogg`) |
| `Trial/Eminem - Stan ... .mp3` | non-bird test audio for the waveform/spectrogram plots |

### Exp 1 - Four-species CNN (`Barfly`, `asbfly`, `ashpri`, `bkcbul`)

| Step | File | Reads | Writes |
|---|---|---|---|
| 1. Spectrograms | `Dataset/dataset1.py` | `Bird_sound/` | `spectrograms/` (81 images) |
| 2. Train/val/test split (70/15/15) | `Dataset/split.py` | `spectrograms/` | `data_split/` |
| 3. Train CNN (3 conv layers, 10 epochs) | `train.py` | `data_split/` | `bird_model.h5` |
| 4. Predict one image | `predict.py` | `bird_model.h5`, `XC134896 (1)_chunk0.png` | prints class |

### Exp 2 - Binary: `asbfly` vs `brnhao`
Raw recordings: `Binary_Classification/binary_analysis/` (10 recordings per species, plus
`test/` with pre-cut clips).

| Sub-experiment | File | Reads | Writes |
|---|---|---|---|
| **2a** Hand-crafted features (spectral centroid, bandwidth, ZCR, RMS), t-tests, threshold, LDA | `Binary_Classification/binary_analysis/feature_analysis.ipynb` | `binary_analysis/` | plots only |
| **2b** Spectrogram -> MLP | `Dataset/Binary.py` then `model.py` | `binary_analysis/` -> `spectrograms_binary/` (155 images) | accuracy printed |
| **2c** Overlapping calls + synthetic noise | `Bird_Mixing_Experiment.ipynb` (first half) | `binary_analysis/`, `model.py` | `noisy_experiment/` |
| **2d** Multi-label (both species in one clip) | `Bird_Mixing_Experiment.ipynb` (second half) | `binary_analysis/` | `multilabel_dataset/`, `multilabel_train/`, `audio_test/` |

**Which script made which dataset**

| Dataset folder | Made by | How |
|---|---|---|
| `spectrograms/` | `Dataset/dataset1.py` | 5 s mel-spectrogram chunks of `Bird_sound/` (no axes) |
| `data_split/` | `Dataset/split.py` | random 70/15/15 copy of `spectrograms/` |
| `spectrograms_binary/` | `Dataset/Binary.py` | same as above for the binary recordings (with axes + colorbar) |
| `noisy_experiment/` | `Bird_Mixing_Experiment.ipynb` | 14 s mixture of both birds; second file adds Gaussian noise at 10 dB SNR |
| `multilabel_dataset/` (v1) | `Bird_Mixing_Experiment.ipynb` | 10 `both_*` mixes + 10 single-species clips per species, built from **all** recordings |
| `multilabel_train/` (v2) | `Bird_Mixing_Experiment.ipynb` | recordings split 70/30 first (seed 42); 7 `both_*` mixes + 7 singles per species from the **training** recordings only |
| `audio_test/` | `Bird_Mixing_Experiment.ipynb` | the 3 held-out recordings per species + `heldout_both_species.wav` |

Held-out recordings in `audio_test/`: asbfly `XC267681`, `XC164848`, `XC209219`; brnhao `XC197755`, `XC124210`, `XC171249`.

### Exp 3 - Web app (`bird-app/`)
Upload a recording, get species with confidence and timestamps. Runs `bird_model.h5` on
5 s windows every 2.5 s, merges consecutive windows into segments, and shows them over
the waveform and spectrogram. See `bird-app/README.md`.

```bash
cd bird-app
cp ../bird_model.h5 model/bird_model.h5
pip install -r requirements.txt
python app.py            # http://localhost:5000
```

Tests: `pytest bird-app/tests` (Python) and `bird-app/tests/segments.test.js` (JS).

### Other
| Path | What it is |
|---|---|
| `extra/` | offline Windows `.whl` packages (librosa, numpy, scipy, scikit-learn ...) |
| `bird-app/model/bird_model.h5` | copy of the root `bird_model.h5` |

---

## Results so far

| Experiment | Result | Note |
|---|---|---|
| 2a bandwidth threshold (3355.8 Hz) | 85% | threshold picked on the same data, so optimistic |
| 2a bandwidth threshold, leave-one-out | 80% | the honest number |
| 2a LDA (centroid + bandwidth), leave-one-out | 80% | |
| 2a t-tests | only bandwidth differs (p = 0.046) | centroid, ZCR, RMS do not separate the species |
| 2b spectrogram MLP | 90.3% (confusion matrix `[[13,1],[2,15]]`) | random split of chunks, see limitations |
| 2d v1 multi-label (leaky) | F1 0.83 per species | same recordings in train and test |
| 2d v2 multi-label (held-out recordings) | asbfly F1 1.00, brnhao F1 0.86; held-out mixture got both labels right | only 6 test clips |

Exp 1 prints its test accuracy when `train.py` runs; it is not stored in the repo.

---

## Known limitations

- **Chunk-level leakage.** Chunks from one recording land in both train and test in
  `split.py` and in 2b, so those scores are higher than they would be on new recordings.
  Split by recording first, as 2d v2 does.
- `binary_analysis/test/` contains clips from `XC158670` and `XC171249`, which are also
  full training recordings in `binary_analysis/brnhao/`.
- Very small dataset (3-10 recordings per species), so every number above has wide error bars.
- `Dataset/Binary.py` draws axes and a colorbar, while `dataset1.py` and the app do not,
  so a model trained on one is not directly comparable with the other.
- `predict.py` and `bird-app/model/classes.json` hardcode the class order
  `Barfly, asbfly, ashpri, bkcbul`; it must match the training folder sort order.
- Scripts use relative paths (`../Bird_sound`), so run them from inside their own folder.
- Notebooks must be run from the repo root as written (`Path.cwd()`), and `Bird_Mixing_Experiment.ipynb`
  expects a `binary_analysis/` folder at the root, whereas it currently lives in `Binary_Classification/`.

## Future work

Split by recording, add more species and recordings, try pre-trained models
(ResNet / EfficientNet), real field-noise augmentation, a multi-label CNN, and a
hosted version of the web app.
