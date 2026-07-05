import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .core import (
    CLASSES,
    SENSOR_LOCAL_LABELS,
    ResidualAttributionExtractor,
    build_dataset,
    reconstruction_features,
    train_tuned_autoencoder,
)


def row_zscore(scores):
    return (scores - scores.mean(axis=1, keepdims=True)) / (scores.std(axis=1, keepdims=True) + 1e-6)


def row_ratio(scores):
    return scores / (scores.sum(axis=1, keepdims=True) + 1e-6)


def row_rank(scores):
    order = np.argsort(-scores, axis=1)
    ranks = np.empty_like(order, dtype=np.float32)
    for i in range(scores.shape[0]):
        ranks[i, order[i]] = np.arange(scores.shape[1], dtype=np.float32)
    denom = max(1, scores.shape[1] - 1)
    return 1.0 - ranks / denom


def channel_feature_tensor(
    windows,
    residual_blocks=None,
    residual_indices=None,
    tcn_energy=None,
    usad_energy=None,
    rich=False,
):
    energy_mats = []
    if residual_blocks is not None:
        for name in ["scan", "global", "coarse"]:
            energy_mats.append((f"hra_{name}", np.asarray(residual_blocks[name], dtype=np.float32)))
    if tcn_energy is not None:
        energy_mats.append(("tcn", np.asarray(tcn_energy, dtype=np.float32)))
    if usad_energy is not None:
        energy_mats.append(("usad", np.asarray(usad_energy, dtype=np.float32)))

    energy_context = {}
    if rich:
        for name, matrix in energy_mats:
            energy_context[name] = {
                "z": row_zscore(matrix),
                "ratio": row_ratio(matrix),
                "rank": row_rank(matrix),
                "max": matrix.max(axis=1, keepdims=True),
                "mean": matrix.mean(axis=1, keepdims=True),
            }

    rows = []
    for i, x in enumerate(windows):
        z = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-6)
        diff = np.diff(z, axis=0)
        channels = x.shape[1]
        feats = []
        for c in range(channels):
            series = z[:, c]
            ch = [
                float(series.mean()),
                float(series.std()),
                float(series.min()),
                float(series.max()),
                float(np.percentile(series, 5)),
                float(np.percentile(series, 25)),
                float(np.percentile(series, 50)),
                float(np.percentile(series, 75)),
                float(np.percentile(series, 95)),
                float(np.mean(np.abs(diff[:, c]))),
                float(np.std(diff[:, c])),
                float(np.polyfit(np.arange(len(series)), series, 1)[0]),
                float(np.mean(np.isclose(x[:, c], 0.0, atol=1e-5))),
                float(len(np.unique(np.round(x[:, c], 4))) / max(1, len(x[:, c]))),
            ]
            if residual_blocks is not None:
                for name in ["scan", "global", "coarse"]:
                    ch.append(float(residual_blocks[name][i, c]))
            if tcn_energy is not None:
                ch.append(float(tcn_energy[i, c]))
            if usad_energy is not None:
                ch.append(float(usad_energy[i, c]))
            if rich:
                ch.append(float(c / max(1, channels - 1)))
                for name, matrix in energy_mats:
                    raw = float(matrix[i, c])
                    ctx = energy_context[name]
                    ch.extend(
                        [
                            raw,
                            float(ctx["z"][i, c]),
                            float(ctx["ratio"][i, c]),
                            float(ctx["rank"][i, c]),
                            float(raw / (ctx["mean"][i, 0] + 1e-6)),
                            float(raw / (ctx["max"][i, 0] + 1e-6)),
                        ]
                    )
                if residual_blocks is not None and tcn_energy is not None and usad_energy is not None:
                    h = float(residual_blocks["scan"][i, c])
                    hg = float(residual_blocks["global"][i, c])
                    hc = float(residual_blocks["coarse"][i, c])
                    t = float(tcn_energy[i, c])
                    u = float(usad_energy[i, c])
                    ch.extend(
                        [
                            t * h,
                            u * h,
                            t * hg,
                            u * hg,
                            max(t, u) * max(h, hg, hc),
                            abs(t - u),
                            abs(h - hg),
                            abs(h - hc),
                            t / (h + 1e-6),
                            u / (h + 1e-6),
                        ]
                    )
                if residual_indices is not None:
                    ch.extend(np.asarray(residual_indices[i], dtype=np.float32).ravel().tolist())
            feats.append(ch)
        rows.append(feats)
    return np.asarray(rows, dtype=np.float32)


def local_windows_only(x, y, targets):
    return np.asarray([CLASSES[int(label)] in SENSOR_LOCAL_LABELS and int(t) >= 0 for label, t in zip(y, targets)])


def make_pairwise_dataset(tensor, targets):
    n, channels, dim = tensor.shape
    x = tensor.reshape(n * channels, dim)
    y = np.zeros(n * channels, dtype=int)
    for i, target in enumerate(targets):
        y[i * channels + int(target)] = 1
    return x, y


def fit_channel_model(x, y, model_name, seed):
    if model_name == "hgb":
        model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=seed)
    elif model_name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed),
        )
    else:
        raise ValueError(model_name)
    model.fit(x, y)
    return model


def predict_scores(model, tensor):
    n, channels, dim = tensor.shape
    flat = tensor.reshape(n * channels, dim)
    proba = model.predict_proba(flat)
    classes = list(model.classes_)
    scores = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(flat), dtype=float)
    return scores.reshape(n, channels)


def localization_metrics(scores, targets):
    ranks = []
    for row, target in zip(scores, targets):
        order = np.argsort(-row)
        ranks.append(int(np.where(order == int(target))[0][0]) + 1)
    ranks = np.asarray(ranks, dtype=np.float32)
    return {
        "n": int(len(ranks)),
        "top1": float(np.mean(ranks <= 1)),
        "top2": float(np.mean(ranks <= 2)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
    }


def train_ae_energy(clean_train, x_train, x_test, kind, seed, args, cache):
    key = (kind, seed, args.dataset, args.max_base_windows, args.n_train_per_class, tuple(args.ae_budgets))
    if key in cache:
        model = cache[key]
    else:
        model_seed = seed + {"tcn": 101, "usad": 1001}[kind]
        model, _meta, _logs = train_tuned_autoencoder(
            clean_train,
            kind,
            model_seed,
            budgets=[int(v) for v in args.ae_budgets],
            batch_size=args.batch_size,
            full_grid=not args.quick_ae_grid,
            patience=args.ae_patience,
        )
        cache[key] = model
    _, train_energy = reconstruction_features(model, x_train, batch_size=args.batch_size)
    _, test_energy = reconstruction_features(model, x_test, batch_size=args.batch_size)
    return train_energy, test_energy


def evaluate_cell(dataset, seed, severity, args, ae_cache):
    payload = build_dataset(
        dataset,
        seed,
        args.n_train_per_class,
        args.n_test_per_class,
        float(severity),
        args.max_base_windows,
        args.data_dir,
        args.length,
        args.channels,
    )
    (
        clean_train,
        x_train,
        y_train,
        _yb_train,
        target_train,
        _start_train,
        x_test,
        y_test,
        _yb_test,
        target_test,
        _start_test,
        _split_info,
    ) = payload

    train_mask = local_windows_only(x_train, y_train, target_train)
    test_mask = local_windows_only(x_test, y_test, target_test)
    x_train_local = [x for x, keep in zip(x_train, train_mask) if keep]
    x_test_local = [x for x, keep in zip(x_test, test_mask) if keep]
    target_train_local = target_train[train_mask]
    target_test_local = target_test[test_mask]

    residual = ResidualAttributionExtractor().fit(clean_train)
    train_scan = residual.transform(x_train_local, mode="scan")
    train_global = residual.transform(x_train_local, mode="global")
    train_coarse = residual.transform(x_train_local, mode="coarse")
    test_scan = residual.transform(x_test_local, mode="scan")
    test_global = residual.transform(x_test_local, mode="global")
    test_coarse = residual.transform(x_test_local, mode="coarse")
    train_res = {"scan": train_scan["energy"], "global": train_global["energy"], "coarse": train_coarse["energy"]}
    test_res = {"scan": test_scan["energy"], "global": test_global["energy"], "coarse": test_coarse["energy"]}

    tcn_train, tcn_test = train_ae_energy(clean_train, x_train_local, x_test_local, "tcn", seed, args, ae_cache)
    usad_train, usad_test = train_ae_energy(clean_train, x_train_local, x_test_local, "usad", seed, args, ae_cache)

    variants = {
        "hra_sl_hgb": (
            channel_feature_tensor(
                x_train_local,
                residual_blocks=train_res,
                residual_indices=train_scan["indices"],
                tcn_energy=tcn_train,
                usad_energy=usad_train,
                rich=True,
            ),
            channel_feature_tensor(
                x_test_local,
                residual_blocks=test_res,
                residual_indices=test_scan["indices"],
                tcn_energy=tcn_test,
                usad_energy=usad_test,
                rich=True,
            ),
            "hgb",
        ),
        "hra_sl_logistic": (
            channel_feature_tensor(x_train_local, residual_blocks=train_res, tcn_energy=tcn_train, usad_energy=usad_train),
            channel_feature_tensor(x_test_local, residual_blocks=test_res, tcn_energy=tcn_test, usad_energy=usad_test),
            "logistic",
        ),
        "ae_only_hgb": (
            channel_feature_tensor(x_train_local, tcn_energy=tcn_train, usad_energy=usad_train),
            channel_feature_tensor(x_test_local, tcn_energy=tcn_test, usad_energy=usad_test),
            "hgb",
        ),
        "hra_attribution_ranking": (None, test_res["scan"], None),
        "tuned_tcn_ae_ranking": (None, tcn_test, None),
        "tuned_usad_ae_ranking": (None, usad_test, None),
    }

    rows = []
    for method, (train_tensor, test_tensor_or_scores, model_name) in variants.items():
        if train_tensor is None:
            scores = test_tensor_or_scores
        else:
            train_x, train_y = make_pairwise_dataset(train_tensor, target_train_local)
            model = fit_channel_model(train_x, train_y, model_name, seed)
            scores = predict_scores(model, test_tensor_or_scores)
        row = {"dataset": dataset, "seed": int(seed), "severity": float(severity), "method": method}
        row.update(localization_metrics(scores, target_test_local))
        rows.append(row)
    return rows


def summarize(df):
    return (
        df.groupby(["dataset", "method"])
        .agg(
            cells=("top1", "count"),
            n=("n", "sum"),
            mean_top1=("top1", "mean"),
            mean_top2=("top2", "mean"),
            mean_mrr=("mrr", "mean"),
            mean_rank=("mean_rank", "mean"),
        )
        .reset_index()
        .sort_values(["dataset", "mean_top1"], ascending=[True, False])
    )


def main():
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
    parser.add_argument("--out-dir", type=Path, default=Path("runs/channel_shortlisting"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ae-budgets", nargs="+", type=int, default=[100])
    parser.add_argument("--ae-patience", type=int, default=12)
    parser.add_argument("--quick-ae-grid", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_info.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    rows = []
    ae_cache = {}
    for dataset in args.datasets:
        args.dataset = dataset
        for seed in args.seeds:
            for severity in args.severities:
                print(f"Running channel shortlisting dataset={dataset} seed={seed} severity={severity}")
                rows.extend(evaluate_cell(dataset, seed, float(severity), args, ae_cache))
                pd.DataFrame(rows).to_csv(args.out_dir / "shortlisting_metrics_partial.csv", index=False)
    raw = pd.DataFrame(rows)
    raw.to_csv(args.out_dir / "shortlisting_metrics.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(args.out_dir / "shortlisting_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
