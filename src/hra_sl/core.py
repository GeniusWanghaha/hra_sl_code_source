import argparse
import copy
import json
import time
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLASSES = [
    "normal",
    "offset_drift",
    "scale_drift",
    "trend_drift",
    "noise_floor",
    "saturation",
    "dropout",
    "time_lag",
    "process_fault",
]

SENSOR_LOCAL_LABELS = {
    "offset_drift",
    "scale_drift",
    "trend_drift",
    "noise_floor",
    "saturation",
    "dropout",
    "time_lag",
}

GROUPS = ["normal", "sensor_local_fault", "process_fault"]
GROUP_BY_CLASS = {
    "normal": "normal",
    "offset_drift": "sensor_local_fault",
    "scale_drift": "sensor_local_fault",
    "trend_drift": "sensor_local_fault",
    "noise_floor": "sensor_local_fault",
    "saturation": "sensor_local_fault",
    "dropout": "sensor_local_fault",
    "time_lag": "sensor_local_fault",
    "process_fault": "process_fault",
}

UCI_HAR_URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
UCI_HAR_SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
]

AIR_QUALITY_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
AIR_QUALITY_COLUMNS = [
    "CO(GT)",
    "PT08.S1(CO)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]


def normalize_window(x):
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mean) / std


def class_to_group(y):
    return np.asarray([GROUPS.index(GROUP_BY_CLASS[CLASSES[int(label)]]) for label in y], dtype=int)


def gini(values):
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0 or values.sum() <= 0:
        return 0.0
    n = len(values)
    return float((2.0 * np.arange(1, n + 1) @ values) / (n * values.sum()) - (n + 1) / n)


def base_sensor_window(rng, length, channels):
    t = np.linspace(0.0, 1.0, length)
    latent_a = np.sin(2 * np.pi * rng.uniform(0.8, 2.4) * t + rng.uniform(0, 2 * np.pi))
    latent_b = 0.5 * np.sin(2 * np.pi * rng.uniform(2.5, 5.5) * t + rng.uniform(0, 2 * np.pi))
    latent_c = 0.25 * rng.normal(size=length).cumsum() / np.sqrt(length)
    latent = latent_a + latent_b + latent_c
    gains = rng.uniform(0.5, 1.8, size=channels)
    offsets = rng.uniform(-0.35, 0.35, size=channels)
    noise = rng.uniform(0.02, 0.10, size=channels)
    lags = rng.integers(0, 5, size=channels)
    x = np.zeros((length, channels), dtype=np.float32)
    for c in range(channels):
        x[:, c] = gains[c] * np.roll(latent, lags[c]) + offsets[c]
        x[:, c] += rng.normal(0, noise[c], size=length)
    return x


def inject_perturbation(rng, x, label, severity):
    x = x.copy()
    length, channels = x.shape
    start = int(length * rng.uniform(0.25, 0.55))
    end = length
    target = int(rng.integers(channels))
    if label == "normal":
        return x, {"target": -1, "start": -1, "severity": float(severity)}
    if label == "offset_drift":
        x[start:end, target] += rng.choice([-1, 1]) * 0.7 * severity
    elif label == "scale_drift":
        x[start:end, target] *= 1.0 + 0.55 * severity
    elif label == "trend_drift":
        x[start:end, target] += rng.choice([-1, 1]) * np.linspace(0, 1.1 * severity, end - start)
    elif label == "noise_floor":
        x[start:end, target] += rng.normal(0, 0.40 * severity, size=end - start)
    elif label == "saturation":
        hi = np.quantile(x[:, target], 0.65)
        lo = np.quantile(x[:, target], 0.35)
        if rng.random() < 0.5:
            x[start:end, target] = np.minimum(x[start:end, target], hi)
        else:
            x[start:end, target] = np.maximum(x[start:end, target], lo)
    elif label == "dropout":
        x[start:end, target] = 0.0
    elif label == "time_lag":
        x[start:end, target] = np.roll(x[start:end, target], int(3 + 8 * severity))
    elif label == "process_fault":
        affected = rng.choice(channels, size=max(2, channels // 2), replace=False)
        shock = rng.choice([-1, 1]) * severity * np.hanning(end - start)
        for c in affected:
            x[start:end, c] += shock * rng.uniform(0.7, 1.3)
    else:
        raise ValueError(label)
    return x, {"target": int(target), "start": int(start), "severity": float(severity)}


def statistical_features(x):
    z = normalize_window(x)
    diff = np.diff(z, axis=0)
    corr = np.nan_to_num(np.corrcoef(z.T), nan=0.0)
    upper = corr[np.triu_indices_from(corr, k=1)]
    feats = [
        z.mean(axis=0),
        z.std(axis=0),
        z.min(axis=0),
        z.max(axis=0),
        np.percentile(z, 10, axis=0),
        np.percentile(z, 90, axis=0),
        np.mean(np.abs(diff), axis=0),
        np.std(diff, axis=0),
        np.polyfit(np.arange(z.shape[0]), z, 1)[0],
        np.array([upper.mean(), upper.std(), upper.min(), upper.max()]),
        np.mean(np.isclose(x, 0.0, atol=1e-5), axis=0),
    ]
    return np.concatenate([np.ravel(f) for f in feats]).astype(np.float32)


def waveform_spectrum_features(x):
    z = normalize_window(x)
    wave = z.T.reshape(-1)
    spec = np.log1p(np.abs(np.fft.rfft(wave)))
    bins = np.array_split(spec, 48)
    pooled = np.array([b.mean() for b in bins] + [b.std() for b in bins], dtype=np.float32)
    envelope = np.array(
        [
            np.mean(np.abs(wave)),
            np.std(wave),
            np.max(np.abs(wave)),
            np.mean(np.abs(np.diff(wave))),
        ],
        dtype=np.float32,
    )
    return np.concatenate([pooled, envelope])


def spectrogram_features(x):
    z = normalize_window(x)
    all_feats = []
    for c in range(z.shape[1]):
        sig = z[:, c]
        frames = []
        frame_len = 32
        hop = 16
        for start in range(0, len(sig) - frame_len + 1, hop):
            frame = sig[start : start + frame_len] * np.hanning(frame_len)
            frames.append(np.log1p(np.abs(np.fft.rfft(frame))))
        stft = np.vstack(frames)
        freq_bins = np.array_split(stft, 8, axis=1)
        all_feats.extend([b.mean() for b in freq_bins])
        all_feats.extend([b.std() for b in freq_bins])
    return np.asarray(all_feats, dtype=np.float32)


def spectral_features_matrix(windows):
    wave = np.vstack([waveform_spectrum_features(x) for x in windows])
    spec = np.vstack([spectrogram_features(x) for x in windows])
    return hstack([wave, spec])


def ensure_uci_har(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "uci_har.zip"
    extract_dir = data_dir / "uci_har"
    if not zip_path.exists():
        urlretrieve(UCI_HAR_URL, zip_path)
    if (not extract_dir.exists()) or (not list(extract_dir.glob("**/Inertial Signals"))):
        if extract_dir.exists():
            import shutil

            shutil.rmtree(extract_dir)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        for nested_zip in extract_dir.glob("*.zip"):
            with zipfile.ZipFile(nested_zip) as zf:
                zf.extractall(extract_dir)
    roots = list(extract_dir.glob("**/Inertial Signals"))
    if not roots:
        raise FileNotFoundError("Could not find UCI HAR Inertial Signals directory")
    return roots[0].parent.parent


def load_uci_har_with_subjects(root, split, max_windows, seed):
    inertial = root / split / "Inertial Signals"
    arrays = []
    for sig in UCI_HAR_SIGNALS:
        arrays.append(np.loadtxt(inertial / f"{sig}_{split}.txt", dtype=np.float32))
    x = np.stack(arrays, axis=-1)
    subjects = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
    rng = np.random.default_rng(seed)
    if max_windows and len(x) > max_windows:
        idx = rng.choice(len(x), size=max_windows, replace=False)
        x = x[idx]
        subjects = subjects[idx]
    return x, subjects


def split_subjects(subjects, seed):
    unique = np.unique(subjects)
    if len(unique) < 3:
        raise ValueError("Need at least three subjects for subject-disjoint splitting")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique)
    first = max(1, int(0.35 * len(shuffled)))
    second = max(first + 1, int(0.70 * len(shuffled)))
    if second >= len(shuffled):
        second = len(shuffled) - 1
    return set(shuffled[:first]), set(shuffled[first:second]), set(shuffled[second:])


def ensure_air_quality(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "air_quality.zip"
    extract_dir = data_dir / "air_quality"
    if not zip_path.exists():
        urlretrieve(AIR_QUALITY_URL, zip_path)
    if not extract_dir.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
    csvs = list(extract_dir.glob("**/*.csv"))
    if not csvs:
        raise FileNotFoundError("Air Quality CSV not found after extraction")
    return csvs[0]


def clean_air_quality_block(block, fill_values=None):
    data = block.replace(-200, np.nan).astype(float)
    data = data.interpolate(method="linear", limit_direction="both")
    if fill_values is not None:
        data = data.fillna(fill_values)
    return data


def windows_from_values(values, length):
    starts = np.arange(0, len(values) - length + 1, length)
    if len(starts) == 0:
        raise ValueError("Not enough rows to create a window")
    return np.asarray([values[s : s + length] for s in starts], dtype=np.float32)


def make_injected_windows(rng, base, base_idx, n_per_class, severity):
    xs, labels, binary, targets, starts = [], [], [], [], []
    for label_idx, label in enumerate(CLASSES):
        chosen = rng.choice(base_idx, size=n_per_class, replace=True)
        for i in chosen:
            x, meta = inject_perturbation(rng, base[int(i)], label, severity)
            xs.append(x)
            labels.append(label_idx)
            binary.append(0 if label == "normal" else 1)
            targets.append(int(meta["target"]))
            starts.append(int(meta["start"]))
    return (
        xs,
        np.asarray(labels, dtype=int),
        np.asarray(binary, dtype=int),
        np.asarray(targets, dtype=int),
        np.asarray(starts, dtype=int),
    )


def build_synthetic(seed, n_train, n_test, length, channels, severity):
    rng = np.random.default_rng(seed)
    clean_train = [base_sensor_window(rng, length, channels) for _ in range(max(240, n_train * 2))]

    def make_split(n_per_class):
        xs, labels, binary, targets, starts = [], [], [], [], []
        for label_idx, label in enumerate(CLASSES):
            for _ in range(n_per_class):
                x = base_sensor_window(rng, length, channels)
                x, meta = inject_perturbation(rng, x, label, severity)
                xs.append(x)
                labels.append(label_idx)
                binary.append(0 if label == "normal" else 1)
                targets.append(int(meta["target"]))
                starts.append(int(meta["start"]))
        return (
            xs,
            np.asarray(labels, dtype=int),
            np.asarray(binary, dtype=int),
            np.asarray(targets, dtype=int),
            np.asarray(starts, dtype=int),
        )

    return clean_train, *make_split(n_train), *make_split(n_test), {"split": "independent synthetic generation"}


def build_uci_har(seed, n_train, n_test, severity, max_base_windows, data_dir):
    root = ensure_uci_har(data_dir)
    base, subjects = load_uci_har_with_subjects(root, "train", max_base_windows, seed)
    cal_subjects, train_subjects, test_subjects = split_subjects(subjects, seed)
    cal_idx = np.asarray([i for i, s in enumerate(subjects) if s in cal_subjects], dtype=int)
    train_idx = np.asarray([i for i, s in enumerate(subjects) if s in train_subjects], dtype=int)
    test_idx = np.asarray([i for i, s in enumerate(subjects) if s in test_subjects], dtype=int)
    rng = np.random.default_rng(seed)
    clean_train = [base[i] for i in cal_idx]
    train = make_injected_windows(rng, base, train_idx, n_train, severity)
    test = make_injected_windows(rng, base, test_idx, n_test, severity)
    info = {
        "split": "subject-disjoint partition within UCI HAR train split",
        "base_windows": int(len(base)),
        "calibration_windows": int(len(cal_idx)),
        "train_base_windows": int(len(train_idx)),
        "test_base_windows": int(len(test_idx)),
        "calibration_subjects": sorted(int(s) for s in cal_subjects),
        "train_subjects": sorted(int(s) for s in train_subjects),
        "test_subjects": sorted(int(s) for s in test_subjects),
    }
    return clean_train, *train, *test, info


def build_air_quality(seed, n_train, n_test, severity, data_dir, length, channels):
    csv_path = ensure_air_quality(data_dir)
    df = pd.read_csv(csv_path, sep=";", decimal=",", engine="python")
    df = df.dropna(axis=1, how="all")
    cols = [c for c in AIR_QUALITY_COLUMNS if c in df.columns]
    raw = df[cols].copy()
    if raw.shape[1] > channels:
        raw = raw.iloc[:, :channels]
    n = len(raw)
    first = int(0.35 * n)
    second = int(0.65 * n)
    raw_cal = raw.iloc[:first].copy()
    raw_train = raw.iloc[first:second].copy()
    raw_test = raw.iloc[second:].copy()

    cal_pre = clean_air_quality_block(raw_cal)
    train_pre = clean_air_quality_block(raw_train)
    fit_pre = pd.concat([cal_pre, train_pre], axis=0)
    fill_values = fit_pre.median(axis=0)
    cal_pre = cal_pre.fillna(fill_values)
    train_pre = train_pre.fillna(fill_values)
    test_pre = clean_air_quality_block(raw_test, fill_values=fill_values)
    mean = fit_pre.fillna(fill_values).mean(axis=0).to_numpy(dtype=np.float32, copy=True)
    std = fit_pre.fillna(fill_values).std(axis=0).replace(0, 1.0).to_numpy(dtype=np.float32, copy=True)

    def normalize_block(block):
        values = block.to_numpy(dtype=np.float32)
        return (values - mean.reshape(1, -1)) / (std.reshape(1, -1) + 1e-6)

    cal_base = windows_from_values(normalize_block(cal_pre), length)
    train_base = windows_from_values(normalize_block(train_pre), length)
    test_base = windows_from_values(normalize_block(test_pre), length)
    rng = np.random.default_rng(seed)
    clean_train = [x for x in cal_base]
    train = make_injected_windows(rng, train_base, np.arange(len(train_base)), n_train, severity)
    test = make_injected_windows(rng, test_base, np.arange(len(test_base)), n_test, severity)
    info = {
        "split": "chronological split with train-fit preprocessing and non-overlapping windows",
        "base_windows": int(len(cal_base) + len(train_base) + len(test_base)),
        "calibration_windows": int(len(cal_base)),
        "train_base_windows": int(len(train_base)),
        "test_base_windows": int(len(test_base)),
        "window_length": int(length),
        "step": int(length),
    }
    return clean_train, *train, *test, info


def build_dataset(dataset, seed, n_train, n_test, severity, max_base_windows, data_dir, length, channels):
    if dataset == "synthetic":
        return build_synthetic(seed, n_train, n_test, length, channels, severity)
    if dataset == "uci_har":
        return build_uci_har(seed, n_train, n_test, severity, max_base_windows, data_dir)
    if dataset == "air_quality":
        return build_air_quality(seed, n_train, n_test, severity, data_dir, length, channels)
    raise ValueError(dataset)


class HraCoreSignature:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.models = []
        self.channels = None

    def fit(self, calibration_windows):
        calibration_windows = [normalize_window(x) for x in calibration_windows]
        self.channels = calibration_windows[0].shape[1]
        self.models = []
        for target in range(self.channels):
            x_rows, y_rows = [], []
            for window in calibration_windows:
                x_rows.append(np.delete(window, target, axis=1))
                y_rows.append(window[:, target])
            model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
            model.fit(np.vstack(x_rows), np.concatenate(y_rows))
            self.models.append(model)
        return self

    def transform_one(self, x):
        z = normalize_window(x)
        length, channels = z.shape
        residual = np.zeros((length, channels), dtype=np.float32)
        pred = np.zeros_like(residual)
        for target, model in enumerate(self.models):
            others = np.delete(z, target, axis=1)
            pred[:, target] = model.predict(others)
            residual[:, target] = z[:, target] - pred[:, target]

        abs_res = np.abs(residual)
        mse = np.mean(residual**2, axis=0)
        mae = np.mean(abs_res, axis=0)
        max_abs = np.max(abs_res, axis=0)
        res_slope = np.polyfit(np.arange(length), residual, 1)[0]
        top = np.sort(mse)[::-1]
        concentration = np.array(
            [
                top[0] / (top.sum() + 1e-8),
                top[:2].sum() / (top.sum() + 1e-8),
                np.mean(mse > mse.mean() + mse.std()),
                np.std(mse) / (np.mean(mse) + 1e-8),
            ],
            dtype=np.float32,
        )
        diff = np.diff(z, axis=0)
        zero_ratio = np.mean(np.isclose(x, 0.0, atol=1e-5), axis=0)
        clipping_ratio = []
        lag_feats = []
        for c in range(channels):
            q_low, q_high = np.quantile(x[:, c], [0.05, 0.95])
            clipping_ratio.append(np.mean((x[:, c] <= q_low + 1e-6) | (x[:, c] >= q_high - 1e-6)))
            peer = np.mean(np.delete(z, c, axis=1), axis=1)
            lags = range(-8, 9)
            cors = []
            for lag in lags:
                if lag < 0:
                    a, b = z[-lag:, c], peer[: lag or None]
                elif lag > 0:
                    a, b = z[:-lag, c], peer[lag:]
                else:
                    a, b = z[:, c], peer
                if len(a) < 4 or np.std(a) < 1e-6 or np.std(b) < 1e-6:
                    cors.append(0.0)
                else:
                    cors.append(float(np.corrcoef(a, b)[0, 1]))
            lag_feats.append(float(lags[int(np.nanargmax(np.abs(cors)))]))
        corr = np.nan_to_num(np.corrcoef(z.T), nan=0.0)
        upper = corr[np.triu_indices_from(corr, k=1)]
        global_feats = np.array(
            [
                mse.mean(),
                mse.std(),
                mse.max(),
                mae.mean(),
                max_abs.max(),
                np.mean(np.abs(diff)),
                np.std(diff),
                upper.mean(),
                upper.std(),
                upper.min(),
                upper.max(),
            ],
            dtype=np.float32,
        )
        signature = self._change_signature(x, z, residual)
        return np.concatenate(
            [
                mse,
                mae,
                max_abs,
                res_slope,
                zero_ratio,
                np.asarray(clipping_ratio, dtype=np.float32),
                np.asarray(lag_feats, dtype=np.float32) / 8.0,
                concentration,
                global_feats,
                signature,
            ]
        ).astype(np.float32)

    def transform(self, windows):
        return np.vstack([self.transform_one(x) for x in windows])

    @staticmethod
    def _slope(a):
        if len(a) < 3:
            return np.zeros(a.shape[1], dtype=np.float32)
        return np.polyfit(np.arange(len(a)), a, 1)[0].astype(np.float32)

    @staticmethod
    def _clip_ratio(a):
        vals = []
        for c in range(a.shape[1]):
            q_low, q_high = np.quantile(a[:, c], [0.05, 0.95])
            vals.append(np.mean((a[:, c] <= q_low + 1e-6) | (a[:, c] >= q_high - 1e-6)))
        return np.asarray(vals, dtype=np.float32)

    def _change_signature(self, x, z, residual):
        length, _channels = z.shape
        candidates = np.unique(np.linspace(int(0.2 * length), int(0.8 * length), 17).astype(int))
        best_score = -np.inf
        best = None
        for split in candidates:
            if split < 8 or length - split < 8:
                continue
            before, after = z[:split], z[split:]
            rb, ra = residual[:split], residual[split:]
            mean_delta = after.mean(axis=0) - before.mean(axis=0)
            std_ratio = np.log((after.std(axis=0) + 1e-6) / (before.std(axis=0) + 1e-6))
            slope_delta = self._slope(after) - self._slope(before)
            res_ratio = np.log((np.mean(ra**2, axis=0) + 1e-6) / (np.mean(rb**2, axis=0) + 1e-6))
            score = (
                np.sort(np.abs(mean_delta))[-2:].sum()
                + np.sort(np.abs(std_ratio))[-2:].sum()
                + 0.5 * np.sort(np.abs(slope_delta))[-2:].sum()
                + 0.5 * np.sort(np.abs(res_ratio))[-2:].sum()
            )
            if score > best_score:
                best_score = float(score)
                best = (split, mean_delta, std_ratio, slope_delta, res_ratio)
        split, mean_delta, std_ratio, slope_delta, res_ratio = best
        before, after = z[:split], z[split:]
        xb, xa = x[:split], x[split:]
        zero_delta = np.mean(np.isclose(xa, 0.0, atol=1e-5), axis=0) - np.mean(
            np.isclose(xb, 0.0, atol=1e-5), axis=0
        )
        clip_delta = self._clip_ratio(xa) - self._clip_ratio(xb)
        corr_before = np.nan_to_num(np.corrcoef(before.T), nan=0.0)
        corr_after = np.nan_to_num(np.corrcoef(after.T), nan=0.0)
        upper_delta = (corr_after - corr_before)[np.triu_indices_from(corr_after, k=1)]

        def sorted_top(v, k=6):
            values = np.sort(np.abs(v))[::-1]
            if len(values) < k:
                values = np.pad(values, (0, k - len(values)))
            return values[:k].astype(np.float32)

        stacked = np.vstack(
            [
                np.abs(mean_delta),
                np.abs(std_ratio),
                np.abs(slope_delta),
                np.abs(res_ratio),
                np.abs(zero_delta),
                np.abs(clip_delta),
            ]
        )
        affected_counts = np.array(
            [
                np.mean(np.abs(mean_delta) > 0.35),
                np.mean(np.abs(std_ratio) > 0.35),
                np.mean(np.abs(slope_delta) > 0.02),
                np.mean(np.abs(res_ratio) > 0.5),
                np.mean(zero_delta > 0.25),
                np.mean(clip_delta > 0.15),
            ],
            dtype=np.float32,
        )
        summary = np.array(
            [
                split / length,
                best_score,
                np.mean(np.abs(mean_delta)),
                np.mean(np.abs(std_ratio)),
                np.mean(np.abs(slope_delta)),
                np.mean(np.abs(res_ratio)),
                np.mean(np.abs(upper_delta)),
                np.max(np.abs(upper_delta)) if len(upper_delta) else 0.0,
                np.max(zero_delta),
                np.max(clip_delta),
                stacked.max(),
                stacked.std(),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                sorted_top(mean_delta),
                sorted_top(std_ratio),
                sorted_top(slope_delta),
                sorted_top(res_ratio),
                sorted_top(zero_delta),
                sorted_top(clip_delta),
                sorted_top(upper_delta),
                affected_counts,
                summary,
            ]
        ).astype(np.float32)


def _slope(a):
    if len(a) < 3:
        return np.zeros(a.shape[1], dtype=np.float32)
    return np.polyfit(np.arange(len(a)), a, 1)[0].astype(np.float32)


def channel_change_scores(x, residual, mode="scan"):
    z = normalize_window(x)
    length, _channels = z.shape
    residual_norm = np.mean(np.abs(residual), axis=0)
    if mode == "global":
        return (residual_norm / (np.mean(residual_norm) + 1e-9)).astype(np.float32)
    if mode == "coarse":
        candidates = np.unique(np.linspace(int(0.2 * length), int(0.8 * length), 5).astype(int))
    elif mode == "scan":
        candidates = np.unique(np.linspace(int(0.2 * length), int(0.8 * length), 17).astype(int))
    else:
        raise ValueError(mode)

    best_score = -np.inf
    best_components = None
    for split in candidates:
        if split < 8 or length - split < 8:
            continue
        before, after = z[:split], z[split:]
        xb, xa = x[:split], x[split:]
        rb, ra = residual[:split], residual[split:]
        pre_std = before.std(axis=0) + 1e-6
        mean_delta = np.abs((after.mean(axis=0) - before.mean(axis=0)) / pre_std)
        std_ratio = np.abs(np.log((after.std(axis=0) + 1e-6) / pre_std))
        slope_delta = np.abs((_slope(after) - _slope(before)) * 10.0)
        diff_before = np.diff(before, axis=0)
        diff_after = np.diff(after, axis=0)
        noise_ratio = np.abs(np.log((diff_after.std(axis=0) + 1e-6) / (diff_before.std(axis=0) + 1e-6)))
        res_ratio = np.abs(np.log((np.mean(ra**2, axis=0) + 1e-6) / (np.mean(rb**2, axis=0) + 1e-6)))
        zero_delta = np.maximum(
            0.0,
            np.mean(np.isclose(xa, 0.0, atol=1e-5), axis=0)
            - np.mean(np.isclose(xb, 0.0, atol=1e-5), axis=0),
        )
        clip_scores = []
        for c in range(x.shape[1]):
            span_b = np.quantile(xb[:, c], 0.95) - np.quantile(xb[:, c], 0.05) + 1e-6
            span_a = np.quantile(xa[:, c], 0.95) - np.quantile(xa[:, c], 0.05) + 1e-6
            clip_scores.append(max(0.0, (span_b - span_a) / span_b))
        clip_scores = np.asarray(clip_scores, dtype=np.float32)
        components = (
            0.95 * mean_delta
            + 0.8 * std_ratio
            + 0.6 * slope_delta
            + 0.65 * noise_ratio
            + 0.45 * res_ratio
            + 0.8 * zero_delta
            + 0.55 * clip_scores
        )
        score = float(np.sort(components)[-2:].sum() + 0.2 * components.max())
        if score > best_score:
            best_score = score
            best_components = components
    if best_components is None:
        best_components = residual_norm
    return (best_components + 0.35 * residual_norm).astype(np.float32)


def residual_blocks(residual, attribution_scores):
    abs_res = np.abs(residual)
    energy = attribution_scores.astype(np.float32)
    total = energy.sum() + 1e-9
    sorted_energy = np.sort(energy)[::-1]
    probs = energy / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-9)) / np.log(len(energy)))
    concentration = np.array(
        [
            float(sorted_energy[0] / total),
            float(sorted_energy[:2].sum() / total),
            gini(energy),
            entropy,
            float(np.std(energy) / (np.mean(energy) + 1e-9)),
        ],
        dtype=np.float32,
    )
    corr = np.nan_to_num(np.corrcoef(residual.T), nan=0.0)
    upper = corr[np.triu_indices_from(corr, k=1)]
    coherence = np.array(
        [
            float(np.mean(np.abs(upper))) if len(upper) else 0.0,
            float(np.max(np.abs(upper))) if len(upper) else 0.0,
            float(np.mean(upper)) if len(upper) else 0.0,
            float(np.std(upper)) if len(upper) else 0.0,
        ],
        dtype=np.float32,
    )
    summaries = np.concatenate(
        [
            np.mean(abs_res, axis=0),
            np.max(abs_res, axis=0),
            np.polyfit(np.arange(residual.shape[0]), residual, 1)[0],
            np.asarray(
                [
                    float(np.mean(energy)),
                    float(np.max(energy)),
                    float(np.mean(abs_res)),
                    float(np.max(abs_res)),
                    float(np.std(np.diff(residual, axis=0))),
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    indices = np.concatenate([concentration, coherence]).astype(np.float32)
    index_values = {
        "rci_top1": float(concentration[0]),
        "rci_top2": float(concentration[1]),
        "residual_gini": float(concentration[2]),
        "residual_entropy": float(concentration[3]),
        "residual_spread": float(concentration[4]),
        "rcs_mean_abs_corr": float(coherence[0]),
        "rcs_max_abs_corr": float(coherence[1]),
        "rcs_mean_corr": float(coherence[2]),
        "rcs_std_corr": float(coherence[3]),
    }
    return energy.astype(np.float32), summaries, indices, index_values


class ResidualAttributionExtractor:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.models = []
        self.channels = None

    def fit(self, calibration_windows):
        calibration_windows = [normalize_window(x) for x in calibration_windows]
        self.channels = calibration_windows[0].shape[1]
        self.models = []
        for target in range(self.channels):
            xs, ys = [], []
            for window in calibration_windows:
                xs.append(np.delete(window, target, axis=1))
                ys.append(window[:, target])
            model = make_pipeline(StandardScaler(), Ridge(alpha=self.alpha))
            model.fit(np.vstack(xs), np.concatenate(ys))
            self.models.append(model)
        return self

    def residual_one(self, x):
        z = normalize_window(x)
        residual = np.zeros_like(z, dtype=np.float32)
        for target, model in enumerate(self.models):
            residual[:, target] = z[:, target] - model.predict(np.delete(z, target, axis=1))
        return residual

    def transform(self, windows, mode="scan"):
        attrs, summaries, indices, energies, index_rows = [], [], [], [], []
        for x in windows:
            residual = self.residual_one(x)
            attribution = channel_change_scores(x, residual, mode=mode)
            energy, summary, index_vec, index_values = residual_blocks(residual, attribution)
            attrs.append(energy)
            summaries.append(summary)
            indices.append(index_vec)
            energies.append(energy)
            index_rows.append(index_values)
        return {
            "attribution": np.vstack(attrs).astype(np.float32),
            "summaries": np.vstack(summaries).astype(np.float32),
            "indices": np.vstack(indices).astype(np.float32),
            "energy": np.vstack(energies).astype(np.float32),
            "index_rows": index_rows,
        }


class ConvAutoencoder(nn.Module):
    def __init__(self, channels, hidden=24):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, kernel_size=5, padding=2),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class MlpAutoencoder(nn.Module):
    def __init__(self, dim, hidden=128, latent=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class GruAutoencoder(nn.Module):
    def __init__(self, channels, hidden=48):
        super().__init__()
        self.encoder = nn.GRU(channels, hidden, batch_first=True)
        self.decoder = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, channels)

    def forward(self, x):
        _, h = self.encoder(x)
        repeated = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        decoded, _ = self.decoder(repeated)
        return self.out(decoded)


def ae_data(clean, kind):
    if kind == "tcn":
        return np.transpose(clean, (0, 2, 1))
    if kind == "usad":
        return clean.reshape(clean.shape[0], -1)
    if kind == "gru":
        return clean
    raise ValueError(kind)


def make_ae_model(kind, length, channels, config):
    if kind == "tcn":
        return ConvAutoencoder(channels, hidden=int(config["hidden"]))
    if kind == "usad":
        return MlpAutoencoder(length * channels, hidden=int(config["hidden"]), latent=int(config["latent"]))
    if kind == "gru":
        return GruAutoencoder(channels, hidden=int(config["hidden"]))
    raise ValueError(kind)


def ae_grid(kind, full_grid=True):
    lrs = [1e-3, 5e-4] if full_grid else [1e-3]
    if kind == "tcn":
        cfgs = [{"hidden": 24}, {"hidden": 48}] if full_grid else [{"hidden": 16}]
    elif kind == "usad":
        cfgs = [{"hidden": 128, "latent": 32}, {"hidden": 256, "latent": 64}] if full_grid else [{"hidden": 64, "latent": 16}]
    elif kind == "gru":
        cfgs = [{"hidden": 32}, {"hidden": 64}] if full_grid else [{"hidden": 16}]
    else:
        raise ValueError(kind)
    out = []
    for cfg in cfgs:
        for lr in lrs:
            row = dict(cfg)
            row["lr"] = lr
            out.append(row)
    return out


def train_tuned_autoencoder(clean_windows, kind, seed, budgets, batch_size, full_grid=True, patience=12, min_delta=1e-5):
    torch.manual_seed(seed)
    np.random.seed(seed)
    clean = np.asarray([normalize_window(x) for x in clean_windows], dtype=np.float32)
    length, channels = clean.shape[1], clean.shape[2]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(clean))
    val_n = max(1, int(0.2 * len(order)))
    val_idx = order[:val_n]
    train_idx = order[val_n:] if len(order[val_n:]) else val_idx
    train_np = ae_data(clean[train_idx], kind)
    val_np = ae_data(clean[val_idx], kind)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_budget = int(max(budgets))
    best = {"val_loss": np.inf, "state": None, "config": None, "best_epoch": None, "selected_budget": None, "train_seconds": None}
    logs = []
    start_all = time.perf_counter()
    loss_fn = nn.MSELoss()
    for cfg_id, cfg in enumerate(ae_grid(kind, full_grid=full_grid)):
        torch.manual_seed(seed + cfg_id * 1009)
        model = make_ae_model(kind, length, channels, cfg).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
        train_loader = DataLoader(TensorDataset(torch.from_numpy(train_np)), batch_size=batch_size, shuffle=True)
        val_tensor = torch.from_numpy(val_np).to(device)
        cfg_best = np.inf
        cfg_best_state = None
        cfg_best_epoch = 0
        stale = 0
        cfg_start = time.perf_counter()
        for epoch in range(1, max_budget + 1):
            model.train()
            train_losses = []
            for (batch,) in train_loader:
                batch = batch.to(device)
                opt.zero_grad()
                loss = loss_fn(model(batch), batch)
                loss.backward()
                opt.step()
                train_losses.append(float(loss.detach().cpu().item()))
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(val_tensor), val_tensor).detach().cpu().item())
            if val_loss + min_delta < cfg_best:
                cfg_best = val_loss
                cfg_best_epoch = epoch
                cfg_best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            budget = min([b for b in budgets if epoch <= b], default=max_budget)
            logs.append(
                {
                    "ae_kind": kind,
                    "config_id": cfg_id,
                    "epoch": epoch,
                    "epoch_budget": int(budget),
                    "train_loss": float(np.mean(train_losses)) if train_losses else np.nan,
                    "val_loss": val_loss,
                    "best_val_loss_so_far": float(cfg_best),
                    "lr": float(cfg["lr"]),
                    "hidden": int(cfg.get("hidden", -1)),
                    "latent": int(cfg.get("latent", -1)),
                    "seconds_elapsed_config": time.perf_counter() - cfg_start,
                }
            )
            if epoch >= min(10, max_budget) and stale >= patience:
                break
        selected_budget = min([b for b in budgets if cfg_best_epoch <= b], default=max_budget)
        if cfg_best < best["val_loss"]:
            best.update(
                {
                    "val_loss": float(cfg_best),
                    "state": cfg_best_state,
                    "config": dict(cfg),
                    "best_epoch": int(cfg_best_epoch),
                    "selected_budget": int(selected_budget),
                    "train_seconds": time.perf_counter() - start_all,
                }
            )
    final_model = make_ae_model(kind, length, channels, best["config"]).to(device)
    final_model.load_state_dict(best["state"])
    final_model.eval()
    metadata = {
        "ae_kind": kind,
        "selected_lr": float(best["config"]["lr"]),
        "selected_hidden": int(best["config"].get("hidden", -1)),
        "selected_latent": int(best["config"].get("latent", -1)),
        "best_epoch": int(best["best_epoch"]),
        "selected_epoch_budget": int(best["selected_budget"]),
        "best_val_loss": float(best["val_loss"]),
        "ae_train_seconds": float(best["train_seconds"]),
        "device": str(device),
        "n_calibration_train": int(len(train_idx)),
        "n_calibration_val": int(len(val_idx)),
    }
    return (final_model, device, kind), metadata, logs


def reconstruction_features(model_bundle, windows, batch_size=256):
    model, device, kind = model_bundle
    clean = np.asarray([normalize_window(x) for x in windows], dtype=np.float32)
    model.eval()
    outs = []
    with torch.no_grad():
        data_np = ae_data(clean, kind)
        loader = DataLoader(TensorDataset(torch.from_numpy(data_np)), batch_size=batch_size, shuffle=False)
        for (batch,) in loader:
            outs.append(model(batch.to(device)).detach().cpu().numpy())
    recon_raw = np.concatenate(outs, axis=0)
    if kind == "tcn":
        recon = np.transpose(recon_raw, (0, 2, 1))
    elif kind == "usad":
        recon = recon_raw.reshape(clean.shape)
    elif kind == "gru":
        recon = recon_raw
    else:
        raise ValueError(kind)
    channel_err = ((recon - clean) ** 2).mean(axis=1)
    sorted_err = -np.sort(-channel_err, axis=1)
    total = channel_err.sum(axis=1, keepdims=True) + 1e-9
    concentration = np.column_stack(
        [
            sorted_err[:, 0] / total[:, 0],
            sorted_err[:, :2].sum(axis=1) / total[:, 0],
            channel_err.std(axis=1) / (channel_err.mean(axis=1) + 1e-9),
        ]
    )
    return np.hstack([channel_err, concentration]).astype(np.float32), channel_err.astype(np.float32)


def train_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]) if idx < len(thresholds) else 0.5


def proba_column(model, x, label):
    proba = model.predict_proba(x)
    classes = list(model.classes_)
    if label not in classes:
        return np.zeros(len(x), dtype=float)
    return proba[:, classes.index(label)]


def hierarchical_predict(train_features, test_features, y_group_train, yb_train, seed=0):
    start = time.perf_counter()
    stage1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1600, class_weight="balanced", random_state=seed))
    stage1.fit(train_features, yb_train)
    train_score = proba_column(stage1, train_features, 1)
    test_score = proba_column(stage1, test_features, 1)
    abnormal_train = y_group_train != GROUPS.index("normal")
    stage2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1600, class_weight="balanced", random_state=seed + 17))
    stage2.fit(
        train_features[abnormal_train],
        (y_group_train[abnormal_train] == GROUPS.index("process_fault")).astype(int),
    )
    stage2_train = stage2.predict(train_features)
    stage2_test = stage2.predict(test_features)
    candidates = np.unique(np.quantile(train_score, np.linspace(0.05, 0.95, 31)))
    candidates = np.unique(np.concatenate([[train_threshold(yb_train, train_score)], candidates]))
    best_threshold = float(candidates[0])
    best_f1 = -1.0
    for threshold_candidate in candidates:
        candidate_abnormal = (train_score >= threshold_candidate).astype(int)
        train_pred = np.full(len(y_group_train), GROUPS.index("normal"), dtype=int)
        train_pred[(candidate_abnormal == 1) & (stage2_train == 0)] = GROUPS.index("sensor_local_fault")
        train_pred[(candidate_abnormal == 1) & (stage2_train == 1)] = GROUPS.index("process_fault")
        score = f1_score(y_group_train, train_pred, average="macro")
        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold_candidate)
    abnormal = (test_score >= best_threshold).astype(int)
    pred = np.full(len(test_features), GROUPS.index("normal"), dtype=int)
    pred[(abnormal == 1) & (stage2_test == 0)] = GROUPS.index("sensor_local_fault")
    pred[(abnormal == 1) & (stage2_test == 1)] = GROUPS.index("process_fault")
    seconds = time.perf_counter() - start
    return pred, test_score, abnormal, best_threshold, seconds


def flat_predict(train_features, test_features, y_group_train, yb_train, seed=0):
    start = time.perf_counter()
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1600, class_weight="balanced", random_state=seed))
    model.fit(train_features, y_group_train)
    pred = model.predict(test_features)
    normal_label = GROUPS.index("normal")
    train_score = 1.0 - proba_column(model, train_features, normal_label)
    test_score = 1.0 - proba_column(model, test_features, normal_label)
    threshold = train_threshold(yb_train, train_score)
    yb_pred = (test_score >= threshold).astype(int)
    seconds = time.perf_counter() - start
    return pred, test_score, yb_pred, threshold, seconds


def localization_metrics(energy, y_labels, targets):
    mask = np.asarray([CLASSES[int(y)] in SENSOR_LOCAL_LABELS for y in y_labels], dtype=bool)
    if mask.sum() == 0:
        return {
            "localization_top1": np.nan,
            "localization_top2": np.nan,
            "localization_mrr": np.nan,
            "localization_mean_rank": np.nan,
        }
    ranks = []
    for row, target in zip(energy[mask], targets[mask]):
        order = np.argsort(-row)
        ranks.append(int(np.where(order == int(target))[0][0]) + 1)
    ranks = np.asarray(ranks)
    return {
        "localization_top1": float(np.mean(ranks <= 1)),
        "localization_top2": float(np.mean(ranks <= 2)),
        "localization_mrr": float(np.mean(1.0 / ranks)),
        "localization_mean_rank": float(np.mean(ranks)),
    }


def safe_auc(y_true, scores, fn):
    try:
        return float(fn(y_true, scores))
    except ValueError:
        return np.nan


@dataclass
class SystemSpec:
    name: str
    train_features: np.ndarray
    test_features: np.ndarray
    energy: np.ndarray
    probe: str = "hier"
    family: str = "feature"
    ae_metadata: dict | None = None


def evaluate_system(spec, y_group_train, y_group_test, y_train, y_test, yb_train, yb_test, targets_test, seed):
    if spec.probe == "hier":
        pred, score, yb_pred, threshold, seconds = hierarchical_predict(
            spec.train_features, spec.test_features, y_group_train, yb_train, seed=seed
        )
    elif spec.probe == "flat":
        pred, score, yb_pred, threshold, seconds = flat_predict(
            spec.train_features, spec.test_features, y_group_train, yb_train, seed=seed
        )
    else:
        raise ValueError(spec.probe)
    abnormal = y_group_test != GROUPS.index("normal")
    fault_source_f1 = f1_score(
        (y_group_test[abnormal] == GROUPS.index("process_fault")).astype(int),
        (pred[abnormal] == GROUPS.index("process_fault")).astype(int),
        average="macro",
        zero_division=0,
    )
    per_class = f1_score(y_group_test, pred, labels=np.arange(len(GROUPS)), average=None, zero_division=0)
    row = {
        "system": spec.name,
        "family": spec.family,
        "probe": spec.probe,
        "source_group_macro_f1": f1_score(y_group_test, pred, average="macro", zero_division=0),
        "source_group_weighted_f1": f1_score(y_group_test, pred, average="weighted", zero_division=0),
        "normal_f1": float(per_class[GROUPS.index("normal")]),
        "sensor_local_f1": float(per_class[GROUPS.index("sensor_local_fault")]),
        "process_f1": float(per_class[GROUPS.index("process_fault")]),
        "fault_source_macro_f1": fault_source_f1,
        "binary_roc_auc": safe_auc(yb_test, score, roc_auc_score),
        "binary_pr_auc": safe_auc(yb_test, score, average_precision_score),
        "binary_f1_train_threshold": f1_score(yb_test, yb_pred, zero_division=0),
        "false_alarm_rate_train_threshold": float(np.mean(yb_pred[yb_test == 0])),
        "probe_train_seconds": float(seconds),
        "selected_threshold": float(threshold),
    }
    if spec.ae_metadata:
        row.update(spec.ae_metadata)
    row.update(localization_metrics(spec.energy, y_test, targets_test))
    pred_rows = []
    for i in range(len(y_test)):
        pred_rows.append(
            {
                "system": spec.name,
                "window_id": i,
                "true_class": CLASSES[int(y_test[i])],
                "true_group": GROUPS[int(y_group_test[i])],
                "pred_group": GROUPS[int(pred[i])],
                "binary_label": int(yb_test[i]),
                "binary_score": float(score[i]),
                "binary_pred": int(yb_pred[i]),
                "target_channel": int(targets_test[i]),
            }
        )
    cm = confusion_matrix(y_group_test, pred, labels=np.arange(len(GROUPS)))
    return row, pred_rows, cm


def hstack(parts):
    return np.hstack([p for p in parts if p is not None and p.shape[1] > 0]).astype(np.float32)


def make_zero_energy(n, channels):
    return np.zeros((n, channels), dtype=np.float32)


def build_feature_systems(clean_train, x_train, x_test, include_flat=False):
    signature_model = HraCoreSignature().fit(clean_train)
    signature_train = signature_model.transform(x_train)
    signature_test = signature_model.transform(x_test)
    stat_train = np.vstack([statistical_features(x) for x in x_train])
    stat_test = np.vstack([statistical_features(x) for x in x_test])
    spec_train = spectral_features_matrix(x_train)
    spec_test = spectral_features_matrix(x_test)
    residual = ResidualAttributionExtractor().fit(clean_train)
    blocks_train = residual.transform(x_train, mode="scan")
    blocks_test = residual.transform(x_test, mode="scan")
    channels = blocks_test["energy"].shape[1]

    def hra_features(blocks, sig):
        return hstack([sig, blocks["attribution"], blocks["summaries"], blocks["indices"]])

    systems = [
        SystemSpec("statistical_hier", stat_train, stat_test, stat_test[:, :channels], probe="hier", family="baseline"),
        SystemSpec("spectral_hier", spec_train, spec_test, make_zero_energy(len(x_test), channels), probe="hier", family="baseline"),
        SystemSpec(
            "stat_spectral_hier",
            hstack([stat_train, spec_train]),
            hstack([stat_test, spec_test]),
            stat_test[:, :channels],
            probe="hier",
            family="baseline",
        ),
        SystemSpec(
            "hra_core_signature_hier",
            signature_train,
            signature_test,
            blocks_test["energy"],
            probe="hier",
            family="feature_block_ablation",
        ),
        SystemSpec(
            "residual_summaries_only_hier",
            blocks_train["summaries"],
            blocks_test["summaries"],
            blocks_test["energy"],
            probe="hier",
            family="feature_block_ablation",
        ),
        SystemSpec(
            "attribution_scores_only_hier",
            blocks_train["attribution"],
            blocks_test["attribution"],
            blocks_test["energy"],
            probe="hier",
            family="feature_block_ablation",
        ),
        SystemSpec(
            "indices_only_hier",
            blocks_train["indices"],
            blocks_test["indices"],
            blocks_test["energy"],
            probe="hier",
            family="feature_block_ablation",
        ),
        SystemSpec(
            "hra_plus_spectral_hier",
            hstack([hra_features(blocks_train, signature_train), spec_train]),
            hstack([hra_features(blocks_test, signature_test), spec_test]),
            blocks_test["energy"],
            probe="hier",
            family="hra_main",
        ),
    ]
    if include_flat:
        systems.extend(
            [
                SystemSpec("statistical_flat", stat_train, stat_test, stat_test[:, :channels], probe="flat", family="hierarchy_vs_flat"),
                SystemSpec("spectral_flat", spec_train, spec_test, make_zero_energy(len(x_test), channels), probe="flat", family="hierarchy_vs_flat"),
                SystemSpec(
                    "hra_plus_spectral_flat",
                    hstack([hra_features(blocks_train, signature_train), spec_train]),
                    hstack([hra_features(blocks_test, signature_test), spec_test]),
                    blocks_test["energy"],
                    probe="flat",
                    family="hierarchy_vs_flat",
                ),
            ]
        )
    return systems, blocks_test["index_rows"]


def add_ae_systems(systems, clean_train, x_train, x_test, args, dataset, seed, ae_cache):
    all_logs = []
    ae_kinds = ["tcn", "usad", "gru"]
    if args.no_gru:
        ae_kinds = ["tcn", "usad"]
    budgets = [int(v) for v in args.ae_budgets]
    for kind in ae_kinds:
        cache_key = (
            dataset,
            int(seed),
            kind,
            int(args.length),
            int(args.channels),
            int(args.max_base_windows),
            int(args.n_train_per_class),
            tuple(budgets),
            bool(args.quick_ae_grid),
        )
        if cache_key in ae_cache:
            model, cached_meta = ae_cache[cache_key]
            meta = dict(cached_meta)
            meta["ae_training_reused"] = True
        else:
            model_seed = seed + {"tcn": 101, "usad": 1001, "gru": 2001}[kind]
            model, meta, logs = train_tuned_autoencoder(
                clean_train,
                kind,
                model_seed,
                budgets=budgets,
                batch_size=args.batch_size,
                full_grid=not args.quick_ae_grid,
                patience=args.ae_patience,
            )
            for row in logs:
                row.update({"dataset": dataset, "seed": seed})
            all_logs.extend(logs)
            meta["ae_training_reused"] = False
            ae_cache[cache_key] = (model, dict(meta))
        start = time.perf_counter()
        train_features, _ = reconstruction_features(model, x_train, batch_size=args.batch_size)
        test_features, energy = reconstruction_features(model, x_test, batch_size=args.batch_size)
        infer_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(x_train) + len(x_test))
        name = {"tcn": "tuned_tcn_ae_hier", "usad": "tuned_usad_ae_hier", "gru": "gru_ae_hier"}[kind]
        meta = dict(meta)
        meta["ae_feature_infer_ms_per_window"] = float(infer_ms)
        systems.append(
            SystemSpec(
                name,
                train_features,
                test_features,
                energy,
                probe="hier",
                family="strengthened_reconstruction",
                ae_metadata=meta,
            )
        )
    return all_logs


def run_one(dataset, seed, severity, args, ae_cache):
    payload = build_dataset(
        dataset,
        seed,
        args.n_train_per_class,
        args.n_test_per_class,
        severity,
        args.max_base_windows,
        args.data_dir,
        args.length,
        args.channels,
    )
    (
        clean_train,
        x_train,
        y_train,
        yb_train,
        _target_train,
        _start_train,
        x_test,
        y_test,
        yb_test,
        target_test,
        start_test,
        split_info,
    ) = payload
    y_group_train = class_to_group(y_train)
    y_group_test = class_to_group(y_test)
    systems, residual_index_rows = build_feature_systems(clean_train, x_train, x_test, include_flat=args.include_flat)
    ae_logs = []
    if not args.skip_ae:
        ae_logs = add_ae_systems(systems, clean_train, x_train, x_test, args, dataset, seed, ae_cache)

    rows, pred_rows, confusions = [], [], {}
    for spec in systems:
        row, preds, cm = evaluate_system(
            spec,
            y_group_train,
            y_group_test,
            y_train,
            y_test,
            yb_train,
            yb_test,
            target_test,
            seed,
        )
        row.update(
            {
                "dataset": dataset,
                "seed": seed,
                "severity": severity,
                "n_train_per_class": args.n_train_per_class,
                "n_test_per_class": args.n_test_per_class,
                "split_protocol": split_info["split"],
            }
        )
        rows.append(row)
        for pred_row in preds:
            pred_row.update({"dataset": dataset, "seed": seed, "severity": severity})
        pred_rows.extend(preds)
        confusions[spec.name] = cm.astype(int).tolist()

    index_rows = []
    for i, values in enumerate(residual_index_rows):
        row = {
            "dataset": dataset,
            "seed": seed,
            "severity": severity,
            "window_id": i,
            "class_name": CLASSES[int(y_test[i])],
            "source_group": GROUPS[int(y_group_test[i])],
            "target_channel": int(target_test[i]),
            "injected_start": int(start_test[i]),
        }
        row.update(values)
        index_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(pred_rows), pd.DataFrame(index_rows), ae_logs, confusions, split_info


def write_partial(out_dir, metric_rows, pred_rows, index_rows, ae_log_rows, confusions, split_infos):
    out_dir.mkdir(parents=True, exist_ok=True)
    if metric_rows:
        pd.concat(metric_rows, ignore_index=True).to_csv(out_dir / "per_run_metrics_partial.csv", index=False)
    if pred_rows:
        pd.concat(pred_rows, ignore_index=True).to_csv(out_dir / "per_window_predictions_partial.csv", index=False)
    if index_rows:
        pd.concat(index_rows, ignore_index=True).to_csv(out_dir / "residual_indices_partial.csv", index=False)
    if ae_log_rows:
        pd.DataFrame(ae_log_rows).to_csv(out_dir / "ae_training_logs_partial.csv", index=False)
    (out_dir / "confusion_matrices_partial.json").write_text(json.dumps(confusions, indent=2), encoding="utf-8")
    (out_dir / "split_info_partial.json").write_text(json.dumps(split_infos, indent=2), encoding="utf-8")


def summarize_metrics(metrics):
    return (
        metrics.groupby(["dataset", "system"])
        .agg(
            cells=("source_group_macro_f1", "count"),
            mean_macro_f1=("source_group_macro_f1", "mean"),
            mean_fault_source_f1=("fault_source_macro_f1", "mean"),
            mean_far=("false_alarm_rate_train_threshold", "mean"),
            mean_top1=("localization_top1", "mean"),
            mean_mrr=("localization_mrr", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "mean_macro_f1"], ascending=[True, False])
    )


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["synthetic"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--severities", nargs="+", type=float, default=[0.75, 1.0, 1.25, 1.5, 1.75])
    parser.add_argument("--n-train-per-class", type=int, default=50)
    parser.add_argument("--n-test-per-class", type=int, default=50)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--max-base-windows", type=int, default=1400)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/source_diagnosis"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ae-budgets", nargs="+", type=int, default=[10, 30, 50, 100])
    parser.add_argument("--ae-patience", type=int, default=12)
    parser.add_argument("--quick-ae-grid", action="store_true")
    parser.add_argument("--skip-ae", action="store_true")
    parser.add_argument("--no-gru", action="store_true")
    parser.add_argument("--include-flat", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    run_info = vars(args).copy()
    run_info["command"] = "python experiments/source_diagnosis/run.py"
    run_info["torch_cuda_available"] = bool(torch.cuda.is_available())
    run_info["torch_version"] = torch.__version__
    (args.out / "run_info.json").write_text(json.dumps(run_info, default=str, indent=2), encoding="utf-8")

    metric_rows, pred_rows, index_rows, ae_log_rows, split_infos, confusions = [], [], [], [], {}, {}
    ae_cache = {}
    for dataset in args.datasets:
        for seed in args.seeds:
            for severity in args.severities:
                print(f"Running source diagnosis dataset={dataset} seed={seed} severity={severity}")
                df, preds, indices, ae_logs, cms, split_info = run_one(dataset, seed, float(severity), args, ae_cache)
                metric_rows.append(df)
                pred_rows.append(preds)
                index_rows.append(indices)
                ae_log_rows.extend(ae_logs)
                split_infos[f"{dataset}/seed={seed}/severity={severity}"] = split_info
                for system, cm in cms.items():
                    confusions[f"{dataset}/seed={seed}/severity={severity}/{system}"] = cm
                print(df.sort_values("source_group_macro_f1", ascending=False).head(8).to_string(index=False))
                write_partial(args.out, metric_rows, pred_rows, index_rows, ae_log_rows, confusions, split_infos)

    metrics = pd.concat(metric_rows, ignore_index=True)
    metrics.to_csv(args.out / "per_run_metrics.csv", index=False)
    pd.concat(pred_rows, ignore_index=True).to_csv(args.out / "per_window_predictions.csv", index=False)
    pd.concat(index_rows, ignore_index=True).to_csv(args.out / "residual_indices.csv", index=False)
    if ae_log_rows:
        pd.DataFrame(ae_log_rows).to_csv(args.out / "ae_training_logs.csv", index=False)
    (args.out / "confusion_matrices.json").write_text(json.dumps(confusions, indent=2), encoding="utf-8")
    (args.out / "split_info.json").write_text(json.dumps(split_infos, indent=2), encoding="utf-8")
    summary = summarize_metrics(metrics)
    summary.to_csv(args.out / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
