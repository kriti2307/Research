# Bird Species Detection from Audio (Mel-Spectrogram + ML)

A project-based research study on identifying bird species from field recordings.
Audio is turned into Mel-spectrograms and classified with a CNN (4 species) or a small
neural network (2 species). The work is then extended to overlapping calls, noise, and
a web app.

**Author:** Kriti Saraogi, B.Tech CSE, Manipal University Jaipur
**Data:** recordings from [xeno-canto](https://xeno-canto.org). Files are named by their `XC` id, e.g. `XC134896`.

```
Audio (.ogg) -> 22,050 Hz mono -> 5 s chunks -> Mel-spectrogram (dB)
             -> image (128x128) -> classifier -> species
```

## Repository layout

```
Song_Experiment/             signal-processing basics (waveform, spectrogram, mel, MFCC)
multiclass_Classification/   4-species CNN
Binary_Classification/       asbfly vs brnhao (features, MLP)
Bird_Mixing/                 overlapping calls, noise, multi-label
bird-app/                    web app that runs the CNN on uploaded audio
extra/                       offline Windows .whl packages (librosa, numpy, scipy, ...)
```

## Species

| Code | Species | Used in |
|---|---|---|
| `asbfly` | Asian Brown Flycatcher | every experiment |
| `brnhao` | Brown Hawk-Owl | Binary_Classification, Bird_Mixing |
| `ashpri` | Ashy Prinia | multiclass_Classification |
| `bkcbul` | Black-crested Bulbul | multiclass_Classification |
| `Barfly` | _(add full name)_ | multiclass_Classification |

---

## 1. `Song_Experiment/` - signal-processing basics

One plot per script, to learn what each representation looks like.

| File | Output |
|---|---|
| `waveform.py` | waveform of the song (`Eminem - Stan ... .mp3`) |
| `spectrogram.py` | dB spectrogram of the song |
| `melspectrogram.py` | mel-spectrogram of `Bird.ogg` |
| `MFCC.py` | 13 MFCCs of `Bird.ogg` |

`Bird.ogg` is the same recording as `XC134896 (1).ogg` (asbfly).

## 2. `multiclass_Classification/` - four-species CNN

Species: `Barfly`, `asbfly`, `ashpri`, `bkcbul` (15 recordings in total).

| Step | File | Reads | Writes |
|---|---|---|---|
| 1. Spectrograms | `Dataset/dataset1.py` | `Bird_sound/` | `spectrograms/` (82 images) |
| 2. Split 70/15/15 | `Dataset/split.py` | `spectrograms/` | `data_split/{train,val,test}/` |
| 3. Train CNN (3 conv layers, 10 epochs) | `train.py` | `data_split/` | `bird_model.h5` |
| 4. Predict one image | `predict.py` | `bird_model.h5` + a spectrogram PNG | prints class and confidence |

`audio_test/` holds the held-out clips for the multi-label experiment (see section 4), not for this CNN.

## 3. `Binary_Classification/` - asbfly vs brnhao

`binary_analysis/` has the raw recordings (10 per species) and `test/` with 10 pre-cut clips per species.

| Sub-experiment | File | Reads | Writes |
|---|---|---|---|
| **3a** Hand-crafted features (spectral centroid, bandwidth, ZCR, RMS), t-tests, threshold, LDA | `binary_analysis/feature_analysis.ipynb` | `binary_analysis/` | plots only |
| **3b** Spectrogram images, then an MLP | `model.py` | `spectrograms_binary/` (155 images) | accuracy printed |

`spectrograms_binary/` is made from `binary_analysis/` by `multiclass_Classification/Dataset/Binary.py`
(5 s chunks, drawn with axes and a colorbar).

## 4. `Bird_Mixing/` - overlap, noise and multi-label

Everything is produced by `Bird_Mixing_Experiment.ipynb`.

| Part | Output folder | What it is |
|---|---|---|
| Overlap + noise | `noisy_experiment/` | 14 s mixture (asbfly 0-6 s, brnhao 3-9 s, asbfly 8-14 s) and a copy with Gaussian noise at 10 dB SNR |
| Multi-label **v1** | `multilabel_dataset/` | 10 `both_*` mixes + 10 single-species clips per species, built from **all** recordings, then split at random |
| Multi-label **v2** | `multilabel_train/` | recordings split 70/30 first (seed 42); 7 `both_*` mixes + 7 singles per species from the **training** recordings only |
| v2 test set | `../multiclass_Classification/audio_test/` | 3 held-out recordings per species + `heldout_both_species.wav` |

Held-out recordings: asbfly `XC267681`, `XC164848`, `XC209219`; brnhao `XC197755`, `XC124210`, `XC171249`.
Each folder has a `labels.csv` (`filename, asbfly, brnhao`).

## 5. `bird-app/` - web app

Upload a recording and get species, confidence and timestamps. It cuts the audio into 5 s windows every 2.5 s,
runs `bird_model.h5` on each, merges consecutive windows into segments, and shows them over the waveform and
spectrogram. Details in `bird-app/README.md`.

```bash
cd bird-app
pip install -r requirements.txt
python app.py            # http://localhost:5000
```

`bird-app/model/bird_model.h5` is a copy of `multiclass_Classification/bird_model.h5`.
Tests: `pytest bird-app/tests` and `bird-app/tests/segments.test.js`.

---

## Results so far

| Experiment | Result | Note |
|---|---|---|
| 3a bandwidth threshold (3355.8 Hz) | 85% | threshold chosen on the same data, so optimistic |
| 3a bandwidth threshold, leave-one-out | 80% | the honest number |
| 3a LDA (centroid + bandwidth), leave-one-out | 80% | |
| 3a t-tests | only bandwidth differs (p = 0.046) | centroid, ZCR and RMS do not separate the species |
| 3b spectrogram MLP | 90.3%, confusion matrix `[[13,1],[2,15]]` | random split of chunks, see limitations |
| 4 multi-label v1 (leaky) | F1 0.83 for each species | same recordings in train and test |
| 4 multi-label v2 (held-out recordings) | asbfly F1 1.00, brnhao F1 0.86 | only 6 test clips; the held-out mixture got both labels right |

The 4-species CNN prints its test accuracy when `train.py` runs; it is not stored in the repo.

## Known issues

**Paths broken by the new folder layout** (the scripts still use the old locations):

- `multiclass_Classification/Dataset/Binary.py` reads `../binary_analysis` and writes `../spectrograms_binary`.
  Those now live in `Binary_Classification/`, and the script itself belongs there.
- `multiclass_Classification/predict.py` loads `XC134896 (1)_chunk0.png` from its own folder; that file now only
  exists at `data_split/test/asbfly/`.
- `Bird_Mixing_Experiment.ipynb` looks for `binary_analysis/` and `model.py` next to itself, and writes the held-out
  test clips next to itself. They are now in `../Binary_Classification/` and `multiclass_Classification/audio_test/`.
- Run each script from its own folder (paths are relative), e.g. `cd multiclass_Classification/Dataset`.

**Modelling limitations**

- `split.py` and 3b split by chunk, so chunks of one recording land in both train and test. Scores are higher than they
  would be on new recordings. Split by recording first, as multi-label v2 does.
- `binary_analysis/test/` has clips from `XC158670` and `XC171249`, which are also full training recordings in `binary_analysis/brnhao/`.
- Tiny dataset (2-10 recordings per species), so every number has wide error bars.
- `Binary.py` draws axes and a colorbar while `dataset1.py` and the app do not, so models trained on one are not comparable with the other.
- The class order `Barfly, asbfly, ashpri, bkcbul` is hardcoded in `predict.py` and `bird-app/model/classes.json`; it must match the sorted training folders.

**Clean-up**

- Delete `.ipynb_checkpoints/` (root and `binary_analysis/`), the empty `Bird_Mixing/Bird_Mixing_Experiment.ipynb.txt`, and
  `bird-app/**/__pycache__/`, then add them to `.gitignore`.

## Future work

Split by recording, more species and recordings, pre-trained models (ResNet / EfficientNet), real field-noise
augmentation, a multi-label CNN, and a hosted version of the app.
