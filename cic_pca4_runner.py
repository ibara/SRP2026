#!/usr/bin/env python3
"""Run Sean's repaired CIC-YNU-IoT SAR PCA-4 quantum-kernel test.

The source parquet files contain more than two million rows.  Sean's quantum
notebook ultimately uses only 100 benign and 100 malware rows, so this runner
scans the files in bounded batches instead of materializing the full combined
dataframe.  Sampling matches the notebook's two calls to
``DataFrame.sample(random_state=42)`` over the concatenated architecture files.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC


TARGET_COLUMN = "MalwareFamily"
BENIGN_FAMILY = "Benign"
ARCH_COLUMN = "Arch"
N_QFEATURES = 4

DEFAULT_DATA_PATHS = {
    "ARM": Path("data/cic_ynu/arm_sar.parquet"),
    "MIPS": Path("data/cic_ynu/mips_sar.parquet"),
    "MIPSEL": Path("data/cic_ynu/mipsel_sar.parquet"),
    "x86": Path("data/cic_ynu/x86_sar.parquet"),
}


def _is_numeric(data_type: pa.DataType) -> bool:
    return (
        pa.types.is_boolean(data_type)
        or pa.types.is_integer(data_type)
        or pa.types.is_floating(data_type)
        or pa.types.is_decimal(data_type)
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def inspect_parquets(
    data_paths: Mapping[str, Path],
) -> tuple[dict[str, pq.ParquetFile], list[str], list[str], dict[str, Any]]:
    """Validate schemas and find deterministic shared/numeric columns."""

    missing = [str(path) for path in data_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing CIC parquet file(s): " + ", ".join(missing))

    parquet_files = {
        architecture: pq.ParquetFile(path)
        for architecture, path in data_paths.items()
    }
    schemas = {
        architecture: parquet_file.schema_arrow
        for architecture, parquet_file in parquet_files.items()
    }

    first_schema = next(iter(schemas.values()))
    shared_columns = [
        column
        for column in first_schema.names
        if all(column in schema.names for schema in schemas.values())
    ]
    numeric_columns = [
        column
        for column in shared_columns
        if all(_is_numeric(schema.field(column).type) for schema in schemas.values())
    ]

    for required in (TARGET_COLUMN, ARCH_COLUMN):
        if required not in shared_columns:
            raise ValueError(f"Required column {required!r} is not shared by all files")
    if len(numeric_columns) < N_QFEATURES:
        raise ValueError(
            f"Only {len(numeric_columns)} shared numeric columns; need {N_QFEATURES}"
        )

    schema_types = {
        architecture: {
            field.name: str(field.type)
            for field in schema
            if field.name in shared_columns
        }
        for architecture, schema in schemas.items()
    }
    type_conflicts = {
        column: {
            architecture: types[column]
            for architecture, types in schema_types.items()
        }
        for column in shared_columns
        if len({types[column] for types in schema_types.values()}) > 1
    }

    summary = {
        "combined_rows": sum(
            parquet_file.metadata.num_rows
            for parquet_file in parquet_files.values()
        ),
        "shared_column_count": len(shared_columns),
        "numeric_feature_count": len(numeric_columns),
        "union_column_count": len(
            set().union(*(set(schema.names) for schema in schemas.values()))
        ),
        "shared_columns": shared_columns,
        "numeric_feature_columns": numeric_columns,
        "shared_column_type_conflicts": type_conflicts,
        "files": {
            architecture: {
                "path": str(data_paths[architecture]),
                "rows": parquet_file.metadata.num_rows,
                "columns": parquet_file.metadata.num_columns,
                "row_groups": parquet_file.metadata.num_row_groups,
                "bytes": data_paths[architecture].stat().st_size,
            }
            for architecture, parquet_file in parquet_files.items()
        },
    }
    return parquet_files, shared_columns, numeric_columns, summary


def load_balanced_sample(
    data_paths: Mapping[str, Path] = DEFAULT_DATA_PATHS,
    *,
    samples_per_class: int = 100,
    random_state: int = 42,
    batch_size: int = 8_192,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Load a reproducible benign/malware sample without loading all rows."""

    parquet_files, _, numeric_columns, summary = inspect_parquets(data_paths)

    print(
        "Dataset schema: "
        f"{summary['combined_rows']:,} rows, "
        f"{summary['shared_column_count']} shared columns, "
        f"{summary['numeric_feature_count']} numeric features",
        flush=True,
    )
    print("Scanning class labels...", flush=True)

    benign_index_chunks: list[np.ndarray] = []
    malware_index_chunks: list[np.ndarray] = []
    total_family_counts: Counter[str] = Counter()
    global_offset = 0

    for architecture, parquet_file in parquet_files.items():
        file_family_counts: Counter[str] = Counter()
        batch_offset = 0
        for batch in parquet_file.iter_batches(
            columns=[TARGET_COLUMN], batch_size=batch_size, use_threads=True
        ):
            labels = batch.column(0).to_numpy(zero_copy_only=False)
            valid = ~pd.isna(labels)
            benign = valid & (labels == BENIGN_FAMILY)
            malware = valid & ~benign

            benign_index_chunks.append(
                np.flatnonzero(benign).astype(np.int64)
                + global_offset
                + batch_offset
            )
            malware_index_chunks.append(
                np.flatnonzero(malware).astype(np.int64)
                + global_offset
                + batch_offset
            )
            file_family_counts.update(str(label) for label in labels[valid])
            batch_offset += batch.num_rows

        summary["files"][architecture]["family_counts"] = dict(
            sorted(file_family_counts.items())
        )
        total_family_counts.update(file_family_counts)
        global_offset += parquet_file.metadata.num_rows

    benign_indices = np.concatenate(benign_index_chunks)
    malware_indices = np.concatenate(malware_index_chunks)
    if len(benign_indices) < samples_per_class or len(malware_indices) < samples_per_class:
        raise ValueError(
            "Not enough rows for the requested balanced sample: "
            f"benign={len(benign_indices)}, malware={len(malware_indices)}"
        )

    # Sean used two independent DataFrame.sample(..., random_state=42) calls.
    # Reinitializing RandomState for each class reproduces those sample positions.
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
        f"Loading {samples_per_class} benign and {samples_per_class} malware rows...",
        flush=True,
    )
    selected_frames: list[pd.DataFrame] = []
    read_columns = [*numeric_columns, TARGET_COLUMN, ARCH_COLUMN]
    global_offset = 0

    for parquet_file in parquet_files.values():
        batch_offset = 0
        for batch in parquet_file.iter_batches(
            columns=read_columns, batch_size=batch_size, use_threads=True
        ):
            batch_start = global_offset + batch_offset
            batch_end = batch_start + batch.num_rows
            left = np.searchsorted(selected_indices, batch_start, side="left")
            right = np.searchsorted(selected_indices, batch_end, side="left")
            if right > left:
                global_indices = selected_indices[left:right]
                local_indices = pa.array(global_indices - batch_start)
                frame = batch.take(local_indices).to_pandas()
                frame.index = global_indices
                selected_frames.append(frame)
            batch_offset += batch.num_rows
        global_offset += parquet_file.metadata.num_rows

    sample = pd.concat(selected_frames, axis=0).loc[sample_order].copy()
    sample.index.name = "combined_source_index"
    if len(sample) != 2 * samples_per_class:
        raise RuntimeError(f"Expected {2 * samples_per_class} rows, loaded {len(sample)}")

    summary["family_counts"] = dict(sorted(total_family_counts.items()))
    summary["benign_rows"] = int(len(benign_indices))
    summary["malware_rows"] = int(len(malware_indices))
    summary["sample"] = {
        "random_state": random_state,
        "samples_per_class": samples_per_class,
        "rows": len(sample),
        "source_indices": sample.index.to_numpy(dtype=np.int64),
        "architecture_counts": sample[ARCH_COLUMN].value_counts().to_dict(),
    }
    return sample, numeric_columns, summary


def prepare_split(
    sample: pd.DataFrame,
    numeric_columns: Sequence[str],
    *,
    test_size: float = 0.30,
    random_state: int = 42,
    target_column: str = TARGET_COLUMN,
    benign_value: Any = BENIGN_FAMILY,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Split first, then fit each preprocessing step on training data only."""

    raw_features = sample.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    labels = np.where(sample[target_column].eq(benign_value), 0, 1)
    row_positions = np.arange(len(sample))
    train_positions, test_positions = train_test_split(
        row_positions,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    raw_train = raw_features.iloc[train_positions].to_numpy(dtype=float)
    raw_test = raw_features.iloc[test_positions].to_numpy(dtype=float)
    y_train = labels[train_positions]
    y_test = labels[test_positions]

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = imputer.fit_transform(raw_train)
    test_imputed = imputer.transform(raw_test)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_imputed)
    test_scaled = scaler.transform(test_imputed)

    pca = PCA(n_components=N_QFEATURES, random_state=random_state)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    range_scaler = MinMaxScaler(feature_range=(-1, 1))
    x_train = range_scaler.fit_transform(train_pca)
    x_test = range_scaler.transform(test_pca)

    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Preprocessing produced a non-finite quantum feature")

    cross_split_matches = np.all(
        test_imputed[:, np.newaxis, :] == train_imputed[np.newaxis, :, :], axis=2
    )
    prepared = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_positions": train_positions,
        "test_positions": test_positions,
        "train_source_indices": sample.index.to_numpy(dtype=np.int64)[train_positions],
        "test_source_indices": sample.index.to_numpy(dtype=np.int64)[test_positions],
    }
    summary = {
        "test_size": test_size,
        "random_state": random_state,
        "train_rows": len(train_positions),
        "test_rows": len(test_positions),
        "train_class_counts": dict(zip(*np.unique(y_train, return_counts=True))),
        "test_class_counts": dict(zip(*np.unique(y_test, return_counts=True))),
        "imputation": "median fitted on training data only",
        "all_missing_training_feature_count": int(
            np.isnan(raw_train).all(axis=0).sum()
        ),
        "standard_scaling": "fitted on training data only",
        "pca_components": N_QFEATURES,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_,
        "pca_explained_variance_ratio_sum": pca.explained_variance_ratio_.sum(),
        "quantum_feature_range": [-1, 1],
        "test_rows_matching_a_training_vector_after_imputation": int(
            cross_split_matches.any(axis=1).sum()
        ),
    }
    return prepared, summary


def evaluate_quantum_kernel(
    prepared: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit and evaluate Sean's exact four-qubit fidelity-kernel SVC."""

    feature_map = ZZFeatureMap(
        feature_dimension=N_QFEATURES,
        reps=2,
        entanglement="linear",
    )
    kernel = FidelityQuantumKernel(feature_map=feature_map)
    model = SVC(kernel="precomputed", C=1.0)

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    print(
        "Evaluating the "
        f"{len(prepared['y_train'])} x {len(prepared['y_train'])} training kernel...",
        flush=True,
    )
    k_train = np.asarray(
        kernel.evaluate(prepared["x_train"], prepared["x_train"]), dtype=float
    )
    k_train = np.nan_to_num(k_train, nan=0.0, posinf=1.0, neginf=0.0)
    model.fit(k_train, prepared["y_train"])

    print(
        "Evaluating the "
        f"{len(prepared['y_test'])} x {len(prepared['y_train'])} test kernel...",
        flush=True,
    )
    k_test = np.asarray(
        kernel.evaluate(prepared["x_test"], prepared["x_train"]), dtype=float
    )
    k_test = np.nan_to_num(k_test, nan=0.0, posinf=1.0, neginf=0.0)
    predictions = model.predict(k_test)
    decision_scores = model.decision_function(k_test)
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu

    y_test = prepared["y_test"]
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, decision_scores),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malware"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
    }
    outputs = {
        "k_train": k_train,
        "k_test": k_test,
        "predictions": predictions,
        "decision_scores": decision_scores,
    }
    return outputs, metrics


def save_results(
    output_dir: Path,
    prepared: Mapping[str, np.ndarray],
    kernel_outputs: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "results.npz", **prepared, **kernel_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Sean's quantum notebook uses 0.30; his full classical test uses 0.20.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cic_pca4_exact_seed42"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")

    sample, numeric_columns, dataset_summary = load_balanced_sample(
        samples_per_class=args.samples_per_class,
        random_state=args.random_state,
        batch_size=args.batch_size,
    )
    prepared, preprocessing_summary = prepare_split(
        sample,
        numeric_columns,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(
        "Prepared split: "
        f"{len(prepared['y_train'])} train / {len(prepared['y_test'])} test, "
        f"{prepared['x_train'].shape[1]} PCA features",
        flush=True,
    )
    kernel_outputs, metrics = evaluate_quantum_kernel(prepared)

    summary = {
        "experiment": "CIC-YNU-IoT SAR PCA-4 exact fidelity quantum kernel",
        "labels": {"0": "Benign", "1": "Malware"},
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
    print(f"Wall time: {metrics['wall_seconds']:.3f} seconds")
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
