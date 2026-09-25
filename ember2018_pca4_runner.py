#!/usr/bin/env python3
"""Run Sean's EMBER2018 PCA-4 exact quantum-kernel experiment safely."""

from __future__ import annotations

import argparse
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from cic_pca4_runner import (
    N_QFEATURES,
    evaluate_quantum_kernel,
    prepare_split,
    save_results,
)


DEFAULT_DATA_PATH = Path(
    "data/ember2018/train_ember_2018_v2_features.parquet"
)
TARGET_COLUMN = "Label"


def _is_numeric(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_boolean(data_type)
        or pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_decimal(data_type)
    )


def load_balanced_sample(
    data_path: Path = DEFAULT_DATA_PATH,
    *,
    samples_per_class: int = 100,
    random_state: int = 42,
    label_batch_size: int = 65_536,
    feature_batch_size: int = 1_024,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Reproduce Sean's samples while keeping the 2 GB parquet out of RAM."""

    if not data_path.is_file():
        raise FileNotFoundError(f"Missing EMBER2018 parquet: {data_path}")
    parquet_file = pq.ParquetFile(data_path)
    schema = parquet_file.schema_arrow
    if TARGET_COLUMN not in schema.names:
        raise ValueError(f"Missing required column {TARGET_COLUMN!r}")

    feature_columns = [
        field.name
        for field in schema
        if field.name != TARGET_COLUMN and _is_numeric(field.type)
    ]
    non_numeric = [
        field.name
        for field in schema
        if field.name != TARGET_COLUMN and not _is_numeric(field.type)
    ]
    if non_numeric:
        raise ValueError(f"Unexpected non-numeric EMBER features: {non_numeric[:10]}")

    print(
        "Dataset schema: "
        f"{parquet_file.metadata.num_rows:,} rows, "
        f"{len(feature_columns):,} numeric features",
        flush=True,
    )
    print("Scanning labels...", flush=True)

    benign_chunks: list[np.ndarray] = []
    malware_chunks: list[np.ndarray] = []
    family_counts: Counter[float] = Counter()
    row_offset = 0
    for batch in parquet_file.iter_batches(
        columns=[TARGET_COLUMN],
        batch_size=label_batch_size,
        use_threads=True,
    ):
        labels = batch.column(0).to_numpy(zero_copy_only=False)
        valid = ~pd.isna(labels)
        family_counts.update(float(label) for label in labels[valid])
        benign_chunks.append(
            np.flatnonzero(valid & (labels == 0)).astype(np.int64) + row_offset
        )
        malware_chunks.append(
            np.flatnonzero(valid & (labels == 1)).astype(np.int64) + row_offset
        )
        row_offset += batch.num_rows

    benign_indices = np.concatenate(benign_chunks)
    malware_indices = np.concatenate(malware_chunks)
    if len(benign_indices) < samples_per_class or len(malware_indices) < samples_per_class:
        raise ValueError(
            "Not enough labeled samples: "
            f"benign={len(benign_indices)}, malware={len(malware_indices)}"
        )

    # These independent generators match Sean's two
    # DataFrame.sample(n=100, random_state=42) calls.
    benign_sample = benign_indices[
        np.random.RandomState(random_state).choice(
            len(benign_indices), size=samples_per_class, replace=False
        )
    ]
    malware_sample = malware_indices[
        np.random.RandomState(random_state).choice(
            len(malware_indices), size=samples_per_class, replace=False
        )
    ]
    sample_order = np.concatenate([benign_sample, malware_sample])
    selected_indices = np.sort(sample_order)

    print(
        f"Loading {len(sample_order)} sampled rows in bounded feature batches...",
        flush=True,
    )
    selected_frames: list[pd.DataFrame] = []
    row_offset = 0
    read_columns = [*feature_columns, TARGET_COLUMN]
    for batch_number, batch in enumerate(
        parquet_file.iter_batches(
            columns=read_columns,
            batch_size=feature_batch_size,
            use_threads=True,
        ),
        start=1,
    ):
        batch_end = row_offset + batch.num_rows
        left = np.searchsorted(selected_indices, row_offset, side="left")
        right = np.searchsorted(selected_indices, batch_end, side="left")
        if right > left:
            global_indices = selected_indices[left:right]
            frame = batch.take(pa.array(global_indices - row_offset)).to_pandas()
            frame.index = global_indices
            selected_frames.append(frame)
        row_offset = batch_end
        if batch_number % 200 == 0:
            print(
                f"  scanned {row_offset:,}/{parquet_file.metadata.num_rows:,} rows",
                flush=True,
            )

    sample = pd.concat(selected_frames, axis=0).loc[sample_order].copy()
    sample.index.name = "source_row_index"
    if len(sample) != 2 * samples_per_class:
        raise RuntimeError(f"Expected {2 * samples_per_class} rows, loaded {len(sample)}")

    summary = {
        "path": str(data_path),
        "bytes": data_path.stat().st_size,
        "rows": parquet_file.metadata.num_rows,
        "columns": parquet_file.metadata.num_columns,
        "row_groups": parquet_file.metadata.num_row_groups,
        "numeric_feature_count": len(feature_columns),
        "label_counts": {
            str(int(label)): int(count)
            for label, count in sorted(family_counts.items())
        },
        "sample": {
            "random_state": random_state,
            "samples_per_class": samples_per_class,
            "rows": len(sample),
            "source_indices": sample.index.to_numpy(dtype=np.int64),
        },
    }
    return sample, feature_columns, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--feature-batch-size", type=int, default=1_024)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/ember2018_pca4_exact_seed42"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample, feature_columns, dataset_summary = load_balanced_sample(
        args.data_path,
        samples_per_class=args.samples_per_class,
        random_state=args.random_state,
        feature_batch_size=args.feature_batch_size,
    )
    prepared, preprocessing_summary = prepare_split(
        sample,
        feature_columns,
        test_size=args.test_size,
        random_state=args.random_state,
        target_column=TARGET_COLUMN,
        benign_value=0,
    )
    print(
        "Prepared split: "
        f"{len(prepared['y_train'])} train / {len(prepared['y_test'])} test, "
        f"{prepared['x_train'].shape[1]} PCA features",
        flush=True,
    )
    kernel_outputs, metrics = evaluate_quantum_kernel(prepared)

    summary = {
        "experiment": "EMBER2018 PCA-4 exact fidelity quantum kernel",
        "labels": {"0": "Benign", "1": "Malware", "-1": "Unlabeled/excluded"},
        "data": dataset_summary,
        "preprocessing_and_split": preprocessing_summary,
        "model": {
            "feature_map": "ZZFeatureMap",
            "feature_dimension": N_QFEATURES,
            "reps": 2,
            "entanglement": "linear",
            "kernel": "FidelityQuantumKernel (exact)",
            "classifier": "SVC(kernel='precomputed', C=1.0)",
        },
        "metrics": metrics,
        "software": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "pandas": version("pandas"),
            "pyarrow": version("pyarrow"),
            "scikit-learn": version("scikit-learn"),
            "qiskit": version("qiskit"),
            "qiskit-machine-learning": version("qiskit-machine-learning"),
        },
    }
    save_results(args.output_dir, prepared, kernel_outputs, summary)

    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.6f}")
    print(f"ROC AUC: {metrics['roc_auc']:.6f}")
    print("Confusion matrix [Benign, Malware]:")
    print(metrics["confusion_matrix"])
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
