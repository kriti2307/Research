#!/usr/bin/env python3
"""Train the multi-label bird model (sigmoid outputs: any number of species, or none).

    python train_multilabel.py --data path/to/folder_with_species_folders --out bird_out

--data is the folder that contains one sub-folder per species (bkwsti/, asbfly/, ...), each
full of .ogg/.wav/.mp3/.flac recordings. Needs: tensorflow, librosa, numpy (scikit-learn optional).

What it does
  1. Splits RECORDINGS (not chunks) into train / val / test per species, so no recording is
     ever in two splits.
  2. Cuts each recording into 5 s chunks (max 10 per recording so long files don't dominate).
  3. Builds fixed training/val/test sets from those chunks, inside each split only:
        single species, 2-3 species mixed together, and "none" (noise / quiet background)
     with multi-hot labels. "None" is simply all-zero labels.
  4. Trains a small CNN with sigmoid outputs and binary cross-entropy.
  5. Reports per-species precision/recall/F1 on the held-out TEST recordings and saves
     bird_model.h5, classes.json, splits.csv, metrics.json.

Labels are weak: every chunk of a recording gets that recording's species, even if the bird
is silent for part of it. Quiet "none" chunks are taken from the quietest part of recordings,
which may still hold a faint call. Both are known limits, not bugs.
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

import birdfeatures as bf

AUDIO_EXT = {".ogg", ".wav", ".mp3", ".flac"}
MAX_CHUNKS = 10


# ---------------------------------------------------------------- data discovery + split
def find_recordings(data_dir):
    recs = {}
    for sp_dir in sorted(p for p in Path(data_dir).iterdir() if p.is_dir()):
        files = sorted(f for f in sp_dir.iterdir() if f.suffix.lower() in AUDIO_EXT)
        if files:
            recs[sp_dir.name] = files
    return recs


def split_recordings(recs, seed, val_frac=0.15, test_frac=0.15):
    """Per species, whole recordings go to exactly one of train / val / test."""
    rng = random.Random(seed)
    out = {}
    for sp, files in recs.items():
        files = list(files)
        rng.shuffle(files)
        n = len(files)
        n_test = max(1, round(test_frac * n)) if n >= 2 else 0
        n_val = max(1, round(val_frac * n)) if n >= 5 else 0
        out[sp] = {
            "test": files[:n_test],
            "val": files[n_test:n_test + n_val],
            "train": files[n_test + n_val:],
        }
    return out


# ---------------------------------------------------------------- chunking
def chunk_recording(y):
    """Normalised recording -> list of float16 chunks of exactly 5 s (max MAX_CHUNKS)."""
    n, win = len(y), bf.N_SAMPLES
    if n <= win:
        starts = [0]
    else:
        starts = list(range(0, n - win + 1, win))
        if n - (starts[-1] + win) >= win // 2:        # keep the tail if it is at least 2.5 s
            starts.append(n - win)
        if len(starts) > MAX_CHUNKS:
            idx = np.linspace(0, len(starts) - 1, MAX_CHUNKS).round().astype(int)
            starts = [starts[i] for i in idx]
    return [bf.fix_length(y[s:s + win]).astype(np.float16) for s in starts]


def load_chunks(path):
    import librosa

    y, _ = librosa.load(str(path), sr=bf.SR, mono=True)
    if len(y) == 0:
        return []
    return chunk_recording(bf.normalize_recording(y))


def build_pools(split_map, loader=load_chunks):
    """pools[split] = {"chunks": {species: [chunk, ...]}, "quiet": [chunk, ...]}"""
    pools = {s: {"chunks": {}, "quiet": []} for s in ("train", "val", "test")}
    rows = []
    for sp, parts in split_map.items():
        for split, files in parts.items():
            for f in files:
                chunks = loader(f)
                rows.append({"file": f"{sp}/{Path(f).name}", "species": sp,
                             "split": split, "n_chunks": len(chunks)})
                if not chunks:
                    continue
                pools[split]["chunks"].setdefault(sp, []).extend(chunks)
                if len(chunks) >= 4:                       # quietest quarter = background
                    rms = np.array([np.sqrt(np.mean(c.astype(np.float32) ** 2)) for c in chunks])
                    cut = np.percentile(rms, 25)
                    pools[split]["quiet"].extend(c for c, r in zip(chunks, rms) if r <= cut)
    return pools, rows


# ---------------------------------------------------------------- sample synthesis
def pink_noise(rng, amp):
    spec = np.fft.rfft(rng.standard_normal(bf.N_SAMPLES))
    f = np.arange(len(spec), dtype=np.float64)
    f[0] = 1.0
    spec = spec / f ** (rng.uniform(0.0, 0.75))            # 0 = white noise, 0.75 = pink-ish
    x = np.fft.irfft(spec, bf.N_SAMPLES)
    return (x / (np.std(x) + 1e-9) * amp).astype(np.float32)


def make_samples(pool, species, n, rng, to_features=bf.chunk_to_features,
                 probs=(0.4, 0.4, 0.2), log=None):
    """n fixed samples: single species / 2-3 species mixed / none. Returns X, Y, kinds."""
    present = [i for i, s in enumerate(species) if pool["chunks"].get(s)]
    if not present:
        raise ValueError("this split has no audio at all")
    X = np.zeros((n,) + bf.INPUT_SHAPE, dtype=np.float16)
    Y = np.zeros((n, len(species)), dtype=np.float32)
    kinds = []

    def draw(i):
        return pool["chunks"][species[i]][rng.integers(len(pool["chunks"][species[i]]))].astype(np.float32)

    for j in range(n):
        kind = rng.choice(["single", "mix", "none"], p=probs)
        if kind == "mix" and len(present) < 2:
            kind = "single"

        if kind == "single":
            i = present[rng.integers(len(present))]
            y = draw(i) * rng.uniform(0.6, 1.2)
            Y[j, i] = 1.0
        elif kind == "mix":
            k = 3 if (len(present) >= 3 and rng.random() < 0.3) else 2
            picks = rng.choice(present, size=k, replace=False)
            y = sum(draw(i) * rng.uniform(0.4, 1.2) for i in picks)
            Y[j, picks] = 1.0
        else:
            if pool["quiet"] and rng.random() < 0.5:
                y = pool["quiet"][rng.integers(len(pool["quiet"]))].astype(np.float32) * rng.uniform(0.5, 1.5)
            else:
                y = pink_noise(rng, rng.uniform(0.002, 0.05))

        if kind != "none" and rng.random() < 0.5:           # background noise augmentation
            y = y + pink_noise(rng, rng.uniform(0.002, 0.03))
        peak = float(np.max(np.abs(y)))
        if peak > 1.5:
            y = y * (1.5 / peak)

        X[j] = to_features(y).astype(np.float16)
        kinds.append(str(kind))
        if log and (j + 1) % 500 == 0:
            log(f"    {j + 1}/{n}")
    return X, Y, np.array(kinds)


# ---------------------------------------------------------------- evaluation
def eval_report(prob, y, kinds, classes, thr=0.5):
    pred = prob >= thr
    truth = y >= 0.5
    per_class = {}
    for c, name in enumerate(classes):
        tp = int(np.sum(pred[:, c] & truth[:, c]))
        fp = int(np.sum(pred[:, c] & ~truth[:, c]))
        fn = int(np.sum(~pred[:, c] & truth[:, c]))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class[name] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
                           "support": int(truth[:, c].sum())}
    out = {"threshold": thr, "per_class": per_class}
    none = kinds == "none"
    if none.any():
        out["none_correct"] = round(float((~pred[none]).all(axis=1).mean()), 3)
    mix = kinds == "mix"
    if mix.any():
        out["mix_all_species_found"] = round(float((pred[mix] | ~truth[mix]).all(axis=1).mean()), 3)
    single = kinds == "single"
    if single.any():
        out["single_exact_match"] = round(float((pred[single] == truth[single]).all(axis=1).mean()), 3)
    try:
        from sklearn.metrics import average_precision_score
        ok = [c for c in range(len(classes)) if truth[:, c].any()]
        out["mean_average_precision"] = round(float(np.mean(
            [average_precision_score(truth[:, c], prob[:, c]) for c in ok])), 3)
    except Exception:
        pass
    return out


# ---------------------------------------------------------------- model
def build_model(n_classes, x_adapt):
    import tensorflow as tf
    from tensorflow.keras import layers, models

    norm = layers.Normalization(axis=-1, name="channel_norm")
    norm.adapt(x_adapt.astype("float32"))                    # per-channel mean/std from TRAIN data only

    inp = layers.Input(shape=bf.INPUT_SHAPE)
    x = norm(inp)
    for f in (32, 64, 128, 128):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(2)(x)
    x = layers.Concatenate()([layers.GlobalAveragePooling2D()(x), layers.GlobalMaxPooling2D()(x)])
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_classes, activation="sigmoid", name="species")(x)   # independent probabilities

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc"),
                 tf.keras.metrics.BinaryAccuracy(name="bin_acc")],
    )
    return model


def to_dataset(x, y, batch, shuffle):
    import tensorflow as tf

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(len(x), seed=0)
    return ds.batch(batch).map(lambda a, b: (tf.cast(a, tf.float32), b)).prefetch(2)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with one sub-folder per species")
    ap.add_argument("--out", default="bird_out")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=500)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    recs = find_recordings(a.data)
    if not recs:
        raise SystemExit(f"No species folders with audio found in {a.data}")
    classes = sorted(recs)
    print("Species:", classes)

    split_map = split_recordings(recs, a.seed)
    print(f"\n{'species':12s} {'train':>6s} {'val':>5s} {'test':>5s}   (recordings)")
    for sp in classes:
        p = split_map[sp]
        flag = "   <-- very few recordings" if len(recs[sp]) < 5 else ""
        print(f"{sp:12s} {len(p['train']):6d} {len(p['val']):5d} {len(p['test']):5d}{flag}")

    print("\nLoading audio and cutting 5 s chunks ...")
    pools, rows = build_pools(split_map)
    with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "species", "split", "n_chunks"])
        w.writeheader()
        w.writerows(rows)
    for s in ("train", "val", "test"):
        n = sum(len(v) for v in pools[s]["chunks"].values())
        print(f"  {s}: {n} chunks")

    data = {}
    for split, n in (("train", a.n_train), ("val", a.n_eval), ("test", a.n_eval)):
        print(f"Building {split} set ({n} samples) ...")
        data[split] = make_samples(pools[split], classes, n, rng, log=print)

    import tensorflow as tf

    tf.keras.utils.set_random_seed(a.seed)
    xtr, ytr, _ = data["train"]
    model = build_model(len(classes), xtr[:2000])
    model.summary()

    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    model.fit(to_dataset(xtr, ytr, 32, True),
              validation_data=to_dataset(*data["val"][:2], 32, False),
              epochs=a.epochs, callbacks=cbs)

    xte, yte, kte = data["test"]
    prob = model.predict(to_dataset(xte, yte, 64, False), verbose=0)
    report = eval_report(prob, yte, kte, classes)
    print("\nTEST (held-out recordings):")
    print(json.dumps(report, indent=2))

    model.save(out / "bird_model.h5")
    with open(out / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": classes, "display_names": {}}, f, indent=2)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved bird_model.h5, classes.json, splits.csv, metrics.json in {out.resolve()}")


if __name__ == "__main__":
    main()