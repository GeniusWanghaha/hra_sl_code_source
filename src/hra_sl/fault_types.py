import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .core import CLASSES, SENSOR_LOCAL_LABELS, ResidualAttributionExtractor, build_dataset


def ranks_from_energy(energy, y_test, targets):
    rows = []
    for i, (scores, label_idx, target) in enumerate(zip(energy, y_test, targets)):
        class_name = CLASSES[int(label_idx)]
        if class_name not in SENSOR_LOCAL_LABELS or int(target) < 0:
            continue
        order = np.argsort(-scores)
        rank = int(np.where(order == int(target))[0][0]) + 1
        rows.append(
            {
                "window_id": i,
                "fault_type": class_name,
                "target_channel": int(target),
                "localization_rank": rank,
                "localization_top1": float(rank <= 1),
                "localization_top2": float(rank <= 2),
                "localization_mrr": float(1.0 / rank),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["synthetic", "uci_har", "air_quality"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--severities", nargs="+", type=float, default=[0.75, 1.0, 1.25, 1.5, 1.75])
    parser.add_argument("--n-train-per-class", type=int, default=50)
    parser.add_argument("--n-test-per-class", type=int, default=50)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--max-base-windows", type=int, default=1400)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("runs/fault_type_breakdown/fault_type_localization.csv"))
    parser.add_argument("--raw-out", type=Path, default=Path("runs/fault_type_breakdown/fault_type_windows.csv"))
    args = parser.parse_args()

    all_rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            for severity in args.severities:
                print(f"Computing fault-type localization dataset={dataset} seed={seed} severity={severity}")
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
                    _x_train,
                    _y_train,
                    _yb_train,
                    _target_train,
                    _start_train,
                    x_test,
                    y_test,
                    _yb_test,
                    target_test,
                    _start_test,
                    _split_info,
                ) = payload
                residual = ResidualAttributionExtractor().fit(clean_train)
                blocks = residual.transform(x_test, mode="scan")
                rows = ranks_from_energy(blocks["energy"], y_test, target_test)
                for row in rows:
                    row.update({"dataset": dataset, "seed": int(seed), "severity": float(severity)})
                all_rows.extend(rows)

    raw = pd.DataFrame(all_rows)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.raw_out, index=False)
    summary = (
        raw.groupby(["dataset", "fault_type"])
        .agg(
            n=("localization_rank", "count"),
            localization_top1=("localization_top1", "mean"),
            localization_top2=("localization_top2", "mean"),
            localization_mrr=("localization_mrr", "mean"),
            localization_mean_rank=("localization_rank", "mean"),
        )
        .reset_index()
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
