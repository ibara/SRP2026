#!/usr/bin/env python3
"""Run the preregistered publication quantum-kernel hardware experiment.

The runner is deliberately separate from contributor notebooks and from the
completed pilot runner.  It prepares deterministic duplicate-safe folds, runs
local baselines, submits resumable IBM Runtime batches, retrieves kernels, and
generates publication tables and figures.

Real-QPU commands require the literal ``--authorize-qpu`` flag.  Merely running
``plan``, ``prepare``, tests, or importing this module cannot submit a job.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import t as student_t
from qiskit import qpy
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

from cic_pca4_runner import DEFAULT_DATA_PATHS as CIC_PATHS
from cic_pca4_runner import inspect_parquets as inspect_cic_parquets
from ember2018_pca4_runner import DEFAULT_DATA_PATH as EMBER_PATH


SCHEMA_VERSION = 1
MASTER_SEED = 20_260_825
# Seed 202 (and the next candidate, 203) caused a Fabiana test vector to
# collide with a training vector only after required range clipping.  Seed 204
# is the first ascending candidate that passes the frozen transformed-overlap
# guard across every dataset/representation; no model metric was consulted.
TRAIN_SEEDS = (101, 204, 303)
N_FOLDS = 3
TRAIN_ROWS = 80
TEST_ROWS = 80
ROWS_PER_CLASS = 40
SHOTS = 512
REPS = 2
ENTANGLEMENT = "linear"
SVC_C = 1.0
TRANSPILER_SEED = 42
OPTIMIZATION_LEVEL = 3
MAX_EXECUTIONS_PER_JOB = 10_000_000
MAX_EXECUTION_TIME_SECONDS = 7_200
CONDITIONS_PER_BATCH = 12
MAX_ATTEMPTS = 2
BOOTSTRAP_REPLICATES = 10_000
EXTRA_RUNS_PER_ANCHOR = 1
AER_PARAMETER_CHUNK_SIZE = 128

ACCOUNT_NAME = "fishing"
INSTANCE_NAME = "General-dedicated"
BACKEND_NAME = "ibm_rensselaer"

RESULT_ROOT = Path("hardware_results/publication_3split_80x80_final_v3")
CAMPAIGN_PATH = RESULT_ROOT / "campaign.json"
SOURCE_POOL_DIR = RESULT_ROOT / "prepared" / "source_pools"
PREPARED_DIR = RESULT_ROOT / "prepared" / "conditions"
CIRCUIT_DIR = RESULT_ROOT / "circuits"
SIMULATION_DIR = RESULT_ROOT / "simulations"
HARDWARE_DIR = RESULT_ROOT / "hardware"
ANALYSIS_DIR = RESULT_ROOT / "analysis"
LATEX_TABLE_DIR = ANALYSIS_DIR / "tables"

CLAMP_PATH = Path("ClaMP_Raw-5184.csv")
FABIANA_PATH = Path("data/fabiana/Malware_and_benign_recognition.csv")

CLAMP_FEATURES = (
    "NumberOfSections",
    "Characteristics",
    "Magic",
    "MajorLinkerVersion",
    "SizeOfCode",
    "AddressOfEntryPoint",
    "ImageBase",
    "SizeOfImage",
    "DllCharacteristics",
    "SizeOfStackReserve",
)

FABIANA_FEATURES = (
    "SizeOfCode",
    "SizeOfInitializedData",
    "SizeOfUninitializedData",
    "AddressOfEntryPoint",
    "BaseOfData",
    "ImageBase",
    "NumberOfSections",
    "DllCharacteristics",
)

CIC_ARCHITECTURES = ("ARM", "MIPS", "MIPSEL", "x86")
CIC_CANDIDATES_PER_STRATUM = 160
EMBER_CANDIDATES_PER_CLASS = 600


@dataclass(frozen=True)
class Configuration:
    name: str
    dataset: str
    representation: str
    qubits: int


CONFIGURATIONS = (
    Configuration("clamp_native10", "clamp", "native", 10),
    Configuration("clamp_pca4", "clamp", "pca", 4),
    Configuration("fabiana_native8", "fabiana", "native", 8),
    Configuration("fabiana_pca4", "fabiana", "pca", 4),
    Configuration("cic_pca4", "cic", "pca", 4),
    Configuration("ember2018_pca4", "ember2018", "pca", 4),
)
CONFIG_BY_NAME = {item.name: item for item in CONFIGURATIONS}
DATASET_CONFIGS: dict[str, tuple[Configuration, ...]] = {
    dataset: tuple(item for item in CONFIGURATIONS if item.dataset == dataset)
    for dataset in {item.dataset for item in CONFIGURATIONS}
}

ANCHORS = (
    ("clamp_pca4", 0),
    ("fabiana_native8", 0),
    ("clamp_native10", 0),
)
EXECUTION_MODES = ("exact", "ideal_aer", "noisy_aer", "hardware")
TERMINAL_JOB_STATES = {"DONE", "ERROR", "CANCELLED"}
CONFIGURATION_LABELS = {
    "clamp_native10": "CLaMP native-10",
    "clamp_pca4": "CLaMP PCA-4",
    "fabiana_native8": "Fabiana native-8",
    "fabiana_pca4": "Fabiana PCA-4",
    "cic_pca4": "CIC PCA-4",
    "ember2018_pca4": "EMBER2018 PCA-4",
}
CONFIGURATION_SHORT_LABELS = {
    "clamp_native10": "Native-10",
    "clamp_pca4": "PCA-4",
    "fabiana_native8": "Native-8",
    "fabiana_pca4": "PCA-4",
    "cic_pca4": "PCA-4",
    "ember2018_pca4": "PCA-4",
}
MODE_LABELS = {
    "exact": "Exact",
    "ideal_aer": "Ideal Aer",
    "noisy_aer": "Noisy Aer",
    "hardware": "Hardware",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(dict(value)), indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if not rows:
        temporary.write_text("", encoding="utf-8")
        temporary.replace(path)
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_jsonable(value))
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else value
                    for key, value in row.items()
                }
            )
    temporary.replace(path)


def atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def package_versions() -> dict[str, str]:
    packages = (
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "qiskit",
        "qiskit-aer",
        "qiskit-machine-learning",
        "qiskit-ibm-runtime",
        "matplotlib",
        "seaborn",
        "scipy",
    )
    values = {"python": platform.python_version()}
    for package in packages:
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = "not installed"
    return values


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(name: str, base: int = MASTER_SEED) -> int:
    digest = hashlib.sha256(f"{base}:{name}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def canonical_row_bytes(row: np.ndarray) -> bytes:
    values = np.asarray(row, dtype="<f8").copy()
    values[np.isnan(values)] = np.nan
    values[values == 0.0] = 0.0
    return values.tobytes(order="C")


def row_hashes(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [hashlib.sha256(canonical_row_bytes(row)).hexdigest() for row in values],
        dtype="U64",
    )


def evaluation_count(n_train: int, n_test: int) -> int:
    return n_train * (n_train - 1) // 2 + n_train * n_test


def train_test_pairs(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    train_pairs = np.asarray(
        [(i, j) for i in range(n_train) for j in range(i + 1, n_train)],
        dtype=np.int32,
    )
    test_pairs = np.asarray(
        [(i, j) for i in range(n_test) for j in range(n_train)],
        dtype=np.int32,
    )
    return train_pairs, test_pairs


def planned_primary_conditions() -> list[dict[str, Any]]:
    return [
        {
            "condition_id": f"{config.name}__fold{fold}",
            "configuration": config.name,
            "dataset": config.dataset,
            "representation": config.representation,
            "qubits": config.qubits,
            "fold": fold,
            "repeat": 0,
            "anchor": (config.name, fold) in ANCHORS,
        }
        for config in CONFIGURATIONS
        for fold in range(N_FOLDS)
    ]


def planned_hardware_schedule() -> list[dict[str, Any]]:
    primary = planned_primary_conditions()
    anchor_primary = [row for row in primary if row["anchor"]]
    others = [row for row in primary if not row["anchor"]]
    rng = np.random.default_rng(stable_seed("hardware-schedule"))
    rng.shuffle(others)
    def repeats(repeat: int) -> list[dict[str, Any]]:
        return [
            {
                **row,
                "condition_id": f"{row['configuration']}__fold{row['fold']}__repeat{repeat}",
                "repeat": repeat,
            }
            for row in anchor_primary
        ]

    # Measure each anchor once near the beginning and once near the end.  This
    # separates the three data splits from the deliberately repeated hardware
    # conditions and exposes coarse device drift without repeating the entire
    # campaign.
    schedule = [*anchor_primary, *others]
    for repeat in range(1, EXTRA_RUNS_PER_ANCHOR + 1):
        schedule.extend(repeats(repeat))
    for order, row in enumerate(schedule):
        row["order"] = order
    return schedule


def campaign_template() -> dict[str, Any]:
    evaluations = evaluation_count(TRAIN_ROWS, TEST_ROWS)
    schedule = planned_hardware_schedule()
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "state": "planned",
        "protocol": {
            "master_seed": MASTER_SEED,
            "training_seeds": TRAIN_SEEDS,
            "folds": N_FOLDS,
            "train_rows": TRAIN_ROWS,
            "test_rows": TEST_ROWS,
            "rows_per_class": ROWS_PER_CLASS,
            "shots": SHOTS,
            "feature_map": {
                "name": "zz_feature_map",
                "reps": REPS,
                "entanglement": ENTANGLEMENT,
                "input_range": [0, 1],
            },
            "classifier": {"name": "SVC", "C": SVC_C},
            "execution_modes": EXECUTION_MODES,
            "mitigation": {
                "dynamical_decoupling": False,
                "gate_twirling": False,
                "measurement_twirling": False,
            },
            "optimization_level": OPTIMIZATION_LEVEL,
            "transpiler_seed": TRANSPILER_SEED,
        },
        "configurations": [asdict(item) for item in CONFIGURATIONS],
        "workload": {
            "train_kernel_evaluations": TRAIN_ROWS * (TRAIN_ROWS - 1) // 2,
            "test_kernel_evaluations": TRAIN_ROWS * TEST_ROWS,
            "evaluations_per_condition": evaluations,
            "executions_per_condition": evaluations * SHOTS,
            "primary_hardware_conditions": len(planned_primary_conditions()),
            "repeat_anchor_conditions": len(schedule) - len(planned_primary_conditions()),
            "hardware_condition_runs": len(schedule),
            "total_hardware_evaluations": len(schedule) * evaluations,
            "total_hardware_executions": len(schedule) * evaluations * SHOTS,
            "max_executions_per_job": MAX_EXECUTIONS_PER_JOB,
        },
        "backend": {
            "account_name": ACCOUNT_NAME,
            "instance_name": INSTANCE_NAME,
            "backend_name": BACKEND_NAME,
        },
        "software": package_versions(),
        "source_pools": {},
        "prepared_conditions": {},
        "simulations": {},
        "circuits": {},
        "hardware_schedule": schedule,
        "batches": [],
    }


def validate_campaign_protocol(campaign: Mapping[str, Any]) -> None:
    expected = _jsonable(campaign_template())
    normalized = _jsonable(campaign)
    for key in ("protocol", "configurations", "workload", "backend"):
        if normalized.get(key) != expected[key]:
            raise RuntimeError(
                f"Existing campaign {CAMPAIGN_PATH} does not match the frozen "
                f"{key}; preserve it and use a deliberately versioned result directory"
            )
    schedule_keys = (
        "condition_id",
        "configuration",
        "dataset",
        "representation",
        "qubits",
        "fold",
        "repeat",
        "anchor",
        "order",
    )
    observed_schedule = normalized.get("hardware_schedule", [])
    if len(observed_schedule) != len(expected["hardware_schedule"]):
        raise RuntimeError("Existing campaign has a different hardware schedule length")
    for observed, planned in zip(observed_schedule, expected["hardware_schedule"]):
        if any(observed.get(key) != planned.get(key) for key in schedule_keys):
            raise RuntimeError(
                "Existing campaign hardware schedule does not match the frozen protocol"
            )


def load_or_create_campaign() -> dict[str, Any]:
    if CAMPAIGN_PATH.exists():
        campaign = read_json(CAMPAIGN_PATH)
        if campaign.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(
                f"Campaign schema {campaign.get('schema_version')} is not {SCHEMA_VERSION}"
            )
        validate_campaign_protocol(campaign)
        return campaign
    campaign = campaign_template()
    atomic_write_json(CAMPAIGN_PATH, campaign)
    return campaign


def save_campaign(campaign: dict[str, Any]) -> None:
    campaign["updated_at_utc"] = utc_now()
    atomic_write_json(CAMPAIGN_PATH, campaign)


def validate_protocol_constants() -> None:
    if TRAIN_ROWS != 2 * ROWS_PER_CLASS or TEST_ROWS != 2 * ROWS_PER_CLASS:
        raise AssertionError("Train/test sizes must each be twice ROWS_PER_CLASS")
    if len(TRAIN_SEEDS) != N_FOLDS:
        raise AssertionError("Need exactly one training seed per fold")
    per_condition = evaluation_count(TRAIN_ROWS, TEST_ROWS) * SHOTS
    if per_condition >= MAX_EXECUTIONS_PER_JOB:
        raise AssertionError(
            f"Condition has {per_condition:,} executions; limit is "
            f"{MAX_EXECUTIONS_PER_JOB:,}"
        )
    schedule = planned_hardware_schedule()
    expected_runs = (
        len(CONFIGURATIONS) * N_FOLDS
        + EXTRA_RUNS_PER_ANCHOR * len(ANCHORS)
    )
    if len(schedule) != expected_runs or len(
        {row["condition_id"] for row in schedule}
    ) != expected_runs:
        raise AssertionError(f"Expected {expected_runs} unique hardware condition runs")


def _deduplicate_pool(
    x: np.ndarray,
    y: np.ndarray,
    source_ids: np.ndarray,
    architectures: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=np.int8)
    source_ids = np.asarray(source_ids, dtype=str)
    architectures = (
        np.full(len(y), "", dtype="U1")
        if architectures is None
        else np.asarray(architectures, dtype=str)
    )
    hashes = row_hashes(x)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, digest in enumerate(hashes):
        groups[str(digest)].append(index)

    keep: list[int] = []
    conflict_groups = 0
    conflict_rows = 0
    duplicate_rows_removed = 0
    for digest in sorted(groups):
        indices = groups[digest]
        labels = np.unique(y[indices])
        if len(labels) != 1:
            conflict_groups += 1
            conflict_rows += len(indices)
            continue
        representative = min(indices, key=lambda idx: source_ids[idx])
        keep.append(representative)
        duplicate_rows_removed += len(indices) - 1

    keep_array = np.asarray(keep, dtype=int)
    order = np.argsort(source_ids[keep_array], kind="stable")
    keep_array = keep_array[order]
    pool = {
        "x": x[keep_array],
        "y": y[keep_array],
        "source_ids": source_ids[keep_array],
        "row_hashes": hashes[keep_array],
        "architectures": architectures[keep_array],
    }
    summary = {
        "candidate_rows": len(y),
        "unique_clean_groups": len(keep_array),
        "duplicate_rows_removed": duplicate_rows_removed,
        "conflicting_groups_excluded": conflict_groups,
        "conflicting_rows_excluded": conflict_rows,
        "class_counts": {
            str(int(label)): int(count)
            for label, count in zip(*np.unique(pool["y"], return_counts=True))
        },
    }
    return pool, summary


def _small_dataset_pool(dataset: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if dataset == "clamp":
        path = CLAMP_PATH
        features = list(CLAMP_FEATURES)
        label = "class"
    elif dataset == "fabiana":
        path = FABIANA_PATH
        features = list(FABIANA_FEATURES)
        label = "Malicious"
    else:
        raise ValueError(dataset)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = [name for name in [*features, label] if name not in frame]
    if missing:
        raise ValueError(f"{dataset} is missing columns: {missing}")
    x = frame.loc[:, features].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    y = frame[label].to_numpy(dtype=np.int8)
    source_ids = np.asarray([f"{dataset}:{index}" for index in frame.index], dtype=str)
    pool, summary = _deduplicate_pool(x, y, source_ids, None)
    summary.update(
        {
            "dataset": dataset,
            "path": str(path),
            "sha256": sha256_file(path),
            "source_rows": len(frame),
            "feature_names": features,
        }
    )
    pool["feature_names"] = np.asarray(features, dtype=str)
    return pool, summary


def _sample_indices(
    indices: np.ndarray,
    size: int,
    seed_name: str,
) -> np.ndarray:
    if len(indices) < size:
        raise ValueError(f"Need {size} candidates for {seed_name}; found {len(indices)}")
    rng = np.random.default_rng(stable_seed(seed_name))
    return rng.choice(indices, size=size, replace=False)


def _load_selected_cic_rows(
    parquet_files: Mapping[str, pq.ParquetFile],
    numeric_columns: Sequence[str],
    selections: Mapping[str, np.ndarray],
    batch_size: int = 2_048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    labels: list[int] = []
    source_ids: list[str] = []
    architectures: list[str] = []
    for architecture in CIC_ARCHITECTURES:
        selected = np.sort(np.asarray(selections[architecture], dtype=np.int64))
        found = 0
        offset = 0
        parquet_file = parquet_files[architecture]
        for batch in parquet_file.iter_batches(
            columns=[*numeric_columns, "MalwareFamily"],
            batch_size=batch_size,
            use_threads=True,
        ):
            end = offset + batch.num_rows
            left = np.searchsorted(selected, offset, side="left")
            right = np.searchsorted(selected, end, side="left")
            if right > left:
                local = selected[left:right] - offset
                frame = batch.take(pa.array(local)).to_pandas()
                feature_values = frame.loc[:, numeric_columns].replace(
                    [np.inf, -np.inf], np.nan
                ).to_numpy(float)
                rows.extend(feature_values)
                for local_index, family in zip(selected[left:right], frame["MalwareFamily"]):
                    labels.append(0 if family == "Benign" else 1)
                    source_ids.append(f"{architecture}:{int(local_index)}")
                    architectures.append(architecture)
                found += right - left
            offset = end
        if found != len(selected):
            raise RuntimeError(
                f"Loaded {found}/{len(selected)} selected {architecture} rows"
            )
    return (
        np.asarray(rows, dtype=float),
        np.asarray(labels, dtype=np.int8),
        np.asarray(source_ids, dtype=str),
        np.asarray(architectures, dtype=str),
    )


def _cic_pool() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    parquet_files, _, numeric_columns, inspection = inspect_cic_parquets(CIC_PATHS)
    selections: dict[str, np.ndarray] = {}
    family_counts: dict[str, dict[str, int]] = {}
    for architecture in CIC_ARCHITECTURES:
        parquet_file = parquet_files[architecture]
        benign_chunks: list[np.ndarray] = []
        malware_chunks: list[np.ndarray] = []
        counts: Counter[str] = Counter()
        offset = 0
        for batch in parquet_file.iter_batches(
            columns=["MalwareFamily"], batch_size=65_536, use_threads=True
        ):
            values = batch.column(0).to_numpy(zero_copy_only=False)
            valid = ~pd.isna(values)
            benign = valid & (values == "Benign")
            malware = valid & ~benign
            benign_chunks.append(np.flatnonzero(benign).astype(np.int64) + offset)
            malware_chunks.append(np.flatnonzero(malware).astype(np.int64) + offset)
            counts.update(str(value) for value in values[valid])
            offset += batch.num_rows
        benign_indices = np.concatenate(benign_chunks)
        malware_indices = np.concatenate(malware_chunks)
        chosen_benign = _sample_indices(
            benign_indices,
            CIC_CANDIDATES_PER_STRATUM,
            f"cic:{architecture}:benign-candidates",
        )
        chosen_malware = _sample_indices(
            malware_indices,
            CIC_CANDIDATES_PER_STRATUM,
            f"cic:{architecture}:malware-candidates",
        )
        selections[architecture] = np.concatenate([chosen_benign, chosen_malware])
        family_counts[architecture] = dict(sorted(counts.items()))

    x, y, source_ids, architectures = _load_selected_cic_rows(
        parquet_files, numeric_columns, selections
    )
    pool, summary = _deduplicate_pool(x, y, source_ids, architectures)
    pool["feature_names"] = np.asarray(numeric_columns, dtype=str)
    summary.update(
        {
            "dataset": "cic",
            "paths": {name: str(path) for name, path in CIC_PATHS.items()},
            "sha256": {name: sha256_file(path) for name, path in CIC_PATHS.items()},
            "source_rows": inspection["combined_rows"],
            "feature_names": numeric_columns,
            "family_counts_by_architecture": family_counts,
            "architecture_class_counts": {
                architecture: {
                    str(label): int(
                        np.sum(
                            (pool["architectures"] == architecture)
                            & (pool["y"] == label)
                        )
                    )
                    for label in (0, 1)
                }
                for architecture in CIC_ARCHITECTURES
            },
        }
    )
    return pool, summary


def _load_selected_ember_rows(
    parquet_file: pq.ParquetFile,
    feature_names: Sequence[str],
    selected_indices: np.ndarray,
    feature_chunk_size: int = 32,
    row_batch_size: int = 65_536,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load sparse rows from a wide, single-row-group parquet within bounded RAM.

    EMBER's 2,381 features live in one 2.35 GB uncompressed row group. Reading
    all columns in even a small row batch causes Arrow/Pandas allocations to
    accumulate. Instead, scan a small column block at a time and retain only
    the requested rows in the final dense sample matrix.
    """

    selected = np.sort(np.asarray(selected_indices, dtype=np.int64))
    if len(np.unique(selected)) != len(selected):
        raise ValueError("EMBER selected row indices are not unique")
    if len(selected) == 0:
        raise ValueError("No EMBER rows selected")
    if selected[0] < 0 or selected[-1] >= parquet_file.metadata.num_rows:
        raise IndexError("EMBER selected row index is outside the parquet")

    labels = np.empty(len(selected), dtype=np.int8)
    offset = 0
    for batch in parquet_file.iter_batches(
        columns=["Label"],
        batch_size=row_batch_size,
        use_threads=True,
    ):
        end = offset + batch.num_rows
        left = np.searchsorted(selected, offset, side="left")
        right = np.searchsorted(selected, end, side="left")
        if right > left:
            local = selected[left:right] - offset
            labels[left:right] = (
                batch.take(pa.array(local)).column(0).to_numpy(zero_copy_only=False)
            )
        offset = end
    if offset != parquet_file.metadata.num_rows:
        raise RuntimeError("EMBER label scan ended before the final row")

    x = np.empty((len(selected), len(feature_names)), dtype=np.float64)
    for column_start in range(0, len(feature_names), feature_chunk_size):
        column_end = min(column_start + feature_chunk_size, len(feature_names))
        columns = list(feature_names[column_start:column_end])
        offset = 0
        for batch in parquet_file.iter_batches(
            columns=columns,
            batch_size=row_batch_size,
            use_threads=True,
        ):
            end = offset + batch.num_rows
            left = np.searchsorted(selected, offset, side="left")
            right = np.searchsorted(selected, end, side="left")
            if right > left:
                local = selected[left:right] - offset
                frame = batch.take(pa.array(local)).to_pandas()
                x[left:right, column_start:column_end] = (
                    frame.loc[:, columns]
                    .replace([np.inf, -np.inf], np.nan)
                    .to_numpy(dtype=float)
                )
            offset = end
        if offset != parquet_file.metadata.num_rows:
            raise RuntimeError(
                f"EMBER feature scan ended early for columns {column_start}:{column_end}"
            )
        del batch
        gc.collect()
        pa.default_memory_pool().release_unused()
        if column_end == len(feature_names) or column_end % 256 == 0:
            print(
                f"  loaded {column_end:,}/{len(feature_names):,} EMBER features",
                flush=True,
            )

    if np.isinf(x).any():
        raise ValueError("EMBER sampled matrix still contains infinity")
    return (
        x,
        labels,
        np.asarray([f"ember2018:{int(index)}" for index in selected], dtype=str),
    )


def _ember_pool() -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not EMBER_PATH.is_file():
        raise FileNotFoundError(EMBER_PATH)
    parquet_file = pq.ParquetFile(EMBER_PATH)
    schema = parquet_file.schema_arrow
    feature_names = [
        field.name
        for field in schema
        if field.name != "Label"
        and (
            pa.types.is_boolean(field.type)
            or pa.types.is_integer(field.type)
            or pa.types.is_floating(field.type)
            or pa.types.is_decimal(field.type)
        )
    ]
    benign_chunks: list[np.ndarray] = []
    malware_chunks: list[np.ndarray] = []
    label_counts: Counter[int] = Counter()
    offset = 0
    for batch in parquet_file.iter_batches(
        columns=["Label"], batch_size=65_536, use_threads=True
    ):
        labels = batch.column(0).to_numpy(zero_copy_only=False)
        valid = ~pd.isna(labels)
        benign_chunks.append(
            np.flatnonzero(valid & (labels == 0)).astype(np.int64) + offset
        )
        malware_chunks.append(
            np.flatnonzero(valid & (labels == 1)).astype(np.int64) + offset
        )
        label_counts.update(int(value) for value in labels[valid])
        offset += batch.num_rows
    benign = np.concatenate(benign_chunks)
    malware = np.concatenate(malware_chunks)
    selected = np.concatenate(
        [
            _sample_indices(
                benign,
                EMBER_CANDIDATES_PER_CLASS,
                "ember2018:benign-candidates",
            ),
            _sample_indices(
                malware,
                EMBER_CANDIDATES_PER_CLASS,
                "ember2018:malware-candidates",
            ),
        ]
    )
    x, y, source_ids = _load_selected_ember_rows(
        parquet_file, feature_names, selected
    )
    pool, summary = _deduplicate_pool(x, y, source_ids, None)
    pool["feature_names"] = np.asarray(feature_names, dtype=str)
    summary.update(
        {
            "dataset": "ember2018",
            "path": str(EMBER_PATH),
            "sha256": sha256_file(EMBER_PATH),
            "source_rows": parquet_file.metadata.num_rows,
            "feature_names": feature_names,
            "source_label_counts": dict(sorted(label_counts.items())),
        }
    )
    return pool, summary


def source_pool_path(dataset: str) -> Path:
    return SOURCE_POOL_DIR / f"{dataset}.npz"


def source_pool_metadata_path(dataset: str) -> Path:
    return SOURCE_POOL_DIR / f"{dataset}.json"


def save_source_pool(
    dataset: str,
    pool: Mapping[str, np.ndarray],
    summary: Mapping[str, Any],
) -> None:
    SOURCE_POOL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(source_pool_path(dataset), **pool)
    atomic_write_json(source_pool_metadata_path(dataset), summary)


def load_source_pool(dataset: str) -> dict[str, np.ndarray]:
    path = source_pool_path(dataset)
    if not path.is_file():
        raise FileNotFoundError(f"Source pool not prepared: {path}")
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def ensure_source_pool(
    dataset: str,
    campaign: dict[str, Any],
) -> dict[str, np.ndarray]:
    if source_pool_path(dataset).is_file() and source_pool_metadata_path(dataset).is_file():
        campaign["source_pools"][dataset] = read_json(
            source_pool_metadata_path(dataset)
        )
        return load_source_pool(dataset)
    if dataset in {"clamp", "fabiana"}:
        pool, summary = _small_dataset_pool(dataset)
    elif dataset == "cic":
        pool, summary = _cic_pool()
    elif dataset == "ember2018":
        pool, summary = _ember_pool()
    else:
        raise ValueError(dataset)
    save_source_pool(dataset, pool, summary)
    campaign["source_pools"][dataset] = summary
    save_campaign(campaign)
    return pool


def cic_arch_counts(fold: int) -> dict[str, int]:
    base, remainder = divmod(ROWS_PER_CLASS, len(CIC_ARCHITECTURES))
    counts = {architecture: base for architecture in CIC_ARCHITECTURES}
    for offset in range(remainder):
        architecture = CIC_ARCHITECTURES[(fold + 2 * offset) % len(CIC_ARCHITECTURES)]
        counts[architecture] += 1
    if sum(counts.values()) != ROWS_PER_CLASS:
        raise AssertionError(counts)
    return counts


def build_fold_indices(
    dataset: str,
    pool: Mapping[str, np.ndarray],
) -> list[dict[str, np.ndarray]]:
    y = np.asarray(pool["y"], dtype=int)
    architectures = np.asarray(pool["architectures"], dtype=str)
    folds = [
        {"train": np.asarray([], dtype=int), "test": np.asarray([], dtype=int)}
        for _ in range(N_FOLDS)
    ]

    if dataset != "cic":
        for label in (0, 1):
            indices = np.flatnonzero(y == label)
            rng = np.random.default_rng(stable_seed(f"{dataset}:test:{label}"))
            indices = indices[rng.permutation(len(indices))]
            needed = N_FOLDS * ROWS_PER_CLASS + ROWS_PER_CLASS
            if len(indices) < needed:
                raise ValueError(
                    f"{dataset} class {label} has {len(indices)} unique groups; "
                    f"need at least {needed}"
                )
            reserved_test = indices[: N_FOLDS * ROWS_PER_CLASS]
            eligible_train = indices[N_FOLDS * ROWS_PER_CLASS :]
            for fold in range(N_FOLDS):
                test = reserved_test[
                    fold * ROWS_PER_CLASS : (fold + 1) * ROWS_PER_CLASS
                ]
                train_rng = np.random.default_rng(
                    stable_seed(f"{dataset}:train:{label}", TRAIN_SEEDS[fold])
                )
                train = train_rng.choice(
                    eligible_train, size=ROWS_PER_CLASS, replace=False
                )
                folds[fold]["test"] = np.concatenate([folds[fold]["test"], test])
                folds[fold]["train"] = np.concatenate([folds[fold]["train"], train])
    else:
        reserved: dict[tuple[int, str], dict[int, np.ndarray]] = {}
        eligible: dict[tuple[int, str], np.ndarray] = {}
        for label in (0, 1):
            for architecture in CIC_ARCHITECTURES:
                indices = np.flatnonzero(
                    (y == label) & (architectures == architecture)
                )
                rng = np.random.default_rng(
                    stable_seed(f"cic:test:{label}:{architecture}")
                )
                indices = indices[rng.permutation(len(indices))]
                offset = 0
                per_fold: dict[int, np.ndarray] = {}
                for fold in range(N_FOLDS):
                    count = cic_arch_counts(fold)[architecture]
                    per_fold[fold] = indices[offset : offset + count]
                    offset += count
                if len(indices) - offset < 13:
                    raise ValueError(
                        f"CIC {architecture} class {label} lacks train candidates"
                    )
                reserved[(label, architecture)] = per_fold
                eligible[(label, architecture)] = indices[offset:]
        for fold in range(N_FOLDS):
            for label in (0, 1):
                for architecture, count in cic_arch_counts(fold).items():
                    test = reserved[(label, architecture)][fold]
                    rng = np.random.default_rng(
                        stable_seed(
                            f"cic:train:{label}:{architecture}",
                            TRAIN_SEEDS[fold],
                        )
                    )
                    train = rng.choice(
                        eligible[(label, architecture)], size=count, replace=False
                    )
                    folds[fold]["test"] = np.concatenate(
                        [folds[fold]["test"], test]
                    )
                    folds[fold]["train"] = np.concatenate(
                        [folds[fold]["train"], train]
                    )

    all_test = np.concatenate([fold["test"] for fold in folds])
    if len(np.unique(all_test)) != N_FOLDS * TEST_ROWS:
        raise AssertionError(f"{dataset} test folds overlap")
    all_test_set = set(all_test.tolist())
    for fold_number, fold in enumerate(folds):
        fold["train"] = np.asarray(fold["train"], dtype=int)
        fold["test"] = np.asarray(fold["test"], dtype=int)
        if len(fold["train"]) != TRAIN_ROWS or len(fold["test"]) != TEST_ROWS:
            raise AssertionError(
                f"{dataset} fold {fold_number} has "
                f"{len(fold['train'])}/{len(fold['test'])} rows"
            )
        if set(fold["train"].tolist()) & all_test_set:
            raise AssertionError(f"{dataset} fold {fold_number} trains on reserved test")
        for kind in ("train", "test"):
            counts = np.bincount(y[fold[kind]], minlength=2)
            if not np.array_equal(counts, [ROWS_PER_CLASS, ROWS_PER_CLASS]):
                raise AssertionError(
                    f"{dataset} fold {fold_number} {kind} counts={counts}"
                )
        if dataset == "cic":
            for kind in ("train", "test"):
                for architecture in CIC_ARCHITECTURES:
                    benign = np.sum(
                        (architectures[fold[kind]] == architecture)
                        & (y[fold[kind]] == 0)
                    )
                    malware = np.sum(
                        (architectures[fold[kind]] == architecture)
                        & (y[fold[kind]] == 1)
                    )
                    if benign != malware:
                        raise AssertionError(
                            f"CIC fold {fold_number} {kind} architecture-label mismatch"
                        )
    return folds


def _transform_fold(
    raw_train: np.ndarray,
    raw_test: np.ndarray,
    representation: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    raw_train = np.asarray(raw_train, dtype=float)
    raw_test = np.asarray(raw_test, dtype=float)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    train_imputed = imputer.fit_transform(raw_train)
    test_imputed = imputer.transform(raw_test)
    transform_arrays: dict[str, np.ndarray] = {
        "imputer_statistics": imputer.statistics_,
    }
    metadata: dict[str, Any] = {
        "imputation": "median fitted on training data only",
        "all_missing_training_features": int(np.isnan(raw_train).all(axis=0).sum()),
        "representation": representation,
    }
    if representation == "pca":
        standard = StandardScaler()
        train_standard = standard.fit_transform(train_imputed)
        test_standard = standard.transform(test_imputed)
        pca = PCA(n_components=4, svd_solver="full")
        train_representation = pca.fit_transform(train_standard)
        test_representation = pca.transform(test_standard)
        transform_arrays.update(
            {
                "standard_mean": standard.mean_,
                "standard_scale": standard.scale_,
                "pca_components": pca.components_,
                "pca_mean": pca.mean_,
                "pca_explained_variance_ratio": pca.explained_variance_ratio_,
            }
        )
        metadata.update(
            {
                "standard_scaling": "fitted on training data only",
                "pca_components": 4,
                "pca_explained_variance_ratio": pca.explained_variance_ratio_,
                "pca_explained_variance_ratio_sum": float(
                    pca.explained_variance_ratio_.sum()
                ),
            }
        )
    elif representation == "native":
        train_representation = train_imputed
        test_representation = test_imputed
    else:
        raise ValueError(representation)

    range_scaler = MinMaxScaler(feature_range=(0, 1), clip=True)
    x_train = range_scaler.fit_transform(train_representation)
    unclipped_test = (
        (test_representation - range_scaler.data_min_) * range_scaler.scale_
        + range_scaler.feature_range[0]
    )
    x_test = range_scaler.transform(test_representation)
    transform_arrays.update(
        {
            "range_min": range_scaler.min_,
            "range_scale": range_scaler.scale_,
            "range_data_min": range_scaler.data_min_,
            "range_data_max": range_scaler.data_max_,
        }
    )
    metadata.update(
        {
            "quantum_feature_range": [0, 1],
            "range_scaler_clip": True,
            "test_values_clipped": int(
                np.sum((unclipped_test < 0) | (unclipped_test > 1))
            ),
        }
    )
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("Preprocessing produced non-finite quantum inputs")
    matches = np.all(
        x_test[:, np.newaxis, :] == x_train[np.newaxis, :, :], axis=2
    )
    metadata["test_rows_matching_transformed_train_vector"] = int(
        matches.any(axis=1).sum()
    )
    if metadata["test_rows_matching_transformed_train_vector"]:
        raise ValueError("Transformed train/test feature-vector overlap detected")
    return x_train, x_test, metadata, transform_arrays


def prepared_condition_path(config_name: str, fold: int) -> Path:
    return PREPARED_DIR / config_name / f"fold_{fold}.npz"


def prepared_metadata_path(config_name: str, fold: int) -> Path:
    return PREPARED_DIR / config_name / f"fold_{fold}.json"


def prepare_condition(
    config: Configuration,
    fold_number: int,
    pool: Mapping[str, np.ndarray],
    fold: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    archive_path = prepared_condition_path(config.name, fold_number)
    metadata_path = prepared_metadata_path(config.name, fold_number)
    if archive_path.is_file() and metadata_path.is_file():
        return read_json(metadata_path)
    train_indices = np.asarray(fold["train"], dtype=int)
    test_indices = np.asarray(fold["test"], dtype=int)
    x_train, x_test, transform_metadata, transform_arrays = _transform_fold(
        pool["x"][train_indices],
        pool["x"][test_indices],
        config.representation,
    )
    if x_train.shape != (TRAIN_ROWS, config.qubits):
        raise AssertionError(
            f"{config.name} fold {fold_number} shape {x_train.shape}"
        )
    train_pairs, test_pairs = train_test_pairs(TRAIN_ROWS, TEST_ROWS)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        archive_path,
        x_train=x_train,
        x_test=x_test,
        y_train=pool["y"][train_indices].astype(np.int8),
        y_test=pool["y"][test_indices].astype(np.int8),
        train_pool_indices=train_indices,
        test_pool_indices=test_indices,
        train_source_ids=pool["source_ids"][train_indices],
        test_source_ids=pool["source_ids"][test_indices],
        train_row_hashes=pool["row_hashes"][train_indices],
        test_row_hashes=pool["row_hashes"][test_indices],
        train_architectures=pool["architectures"][train_indices],
        test_architectures=pool["architectures"][test_indices],
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        **transform_arrays,
    )
    metadata = {
        "configuration": asdict(config),
        "fold": fold_number,
        "training_seed": TRAIN_SEEDS[fold_number],
        "train_rows": TRAIN_ROWS,
        "test_rows": TEST_ROWS,
        "train_class_counts": [ROWS_PER_CLASS, ROWS_PER_CLASS],
        "test_class_counts": [ROWS_PER_CLASS, ROWS_PER_CLASS],
        "feature_names": pool["feature_names"].tolist(),
        "source_pool": str(source_pool_path(config.dataset)),
        "archive": str(archive_path),
        "transform": transform_metadata,
        "created_at_utc": utc_now(),
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def prepare_all() -> None:
    validate_protocol_constants()
    campaign = load_or_create_campaign()
    pools: dict[str, dict[str, np.ndarray]] = {}
    folds_by_dataset: dict[str, list[dict[str, np.ndarray]]] = {}
    for dataset in ("clamp", "fabiana", "cic", "ember2018"):
        print(f"Preparing duplicate-safe {dataset} source pool...", flush=True)
        pools[dataset] = ensure_source_pool(dataset, campaign)
        folds_by_dataset[dataset] = build_fold_indices(dataset, pools[dataset])

    for config in CONFIGURATIONS:
        for fold in range(N_FOLDS):
            print(f"Preparing {config.name} fold {fold}...", flush=True)
            metadata = prepare_condition(
                config,
                fold,
                pools[config.dataset],
                folds_by_dataset[config.dataset][fold],
            )
            campaign["prepared_conditions"][f"{config.name}__fold{fold}"] = metadata

    for dataset in ("clamp", "fabiana"):
        configs = DATASET_CONFIGS[dataset]
        if len(configs) != 2:
            raise AssertionError(f"Expected native/PCA pair for {dataset}")
        for fold in range(N_FOLDS):
            with np.load(prepared_condition_path(configs[0].name, fold)) as left:
                with np.load(prepared_condition_path(configs[1].name, fold)) as right:
                    for key in ("train_source_ids", "test_source_ids", "y_train", "y_test"):
                        if not np.array_equal(left[key], right[key]):
                            raise AssertionError(
                                f"{dataset} native/PCA fold {fold} differs at {key}"
                            )
    campaign["state"] = "prepared"
    save_campaign(campaign)
    primary_conditions = len(CONFIGURATIONS) * N_FOLDS
    print(
        f"Prepared all {primary_conditions} primary conditions under {PREPARED_DIR}",
        flush=True,
    )


def _project_psd(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    symmetric = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) / 2
    np.fill_diagonal(symmetric, 1.0)
    values, vectors = np.linalg.eigh(symmetric)
    clipped = np.clip(values, 0, None)
    projected = (vectors * clipped) @ vectors.T
    diagonal = np.sqrt(np.clip(np.diag(projected), 1e-15, None))
    projected = projected / np.outer(diagonal, diagonal)
    projected = (projected + projected.T) / 2
    np.fill_diagonal(projected, 1.0)
    diagnostics = {
        "raw_minimum_eigenvalue": float(values.min()),
        "negative_eigenvalue_count": int(np.sum(values < -1e-12)),
        "negative_eigenvalue_mass": float(np.abs(values[values < 0]).sum()),
        "projected_minimum_eigenvalue": float(np.linalg.eigvalsh(projected).min()),
    }
    return projected, diagnostics


def _classification_metrics(
    k_train: np.ndarray,
    k_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model = SVC(kernel="precomputed", C=SVC_C)
    model.fit(k_train, y_train)
    predictions = model.predict(k_test)
    scores = model.decision_function(k_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]),
        "benign_recall": float(np.mean(predictions[y_test == 0] == 0)),
        "malware_recall": float(np.mean(predictions[y_test == 1] == 1)),
    }
    return metrics, predictions.astype(np.int8), scores.astype(float)


def _kernel_comparison(observed: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=float)
    exact = np.asarray(exact, dtype=float)
    delta = observed - exact
    denominator = float(np.linalg.norm(exact))
    if np.std(observed) == 0 or np.std(exact) == 0:
        correlation = float("nan")
    else:
        correlation = float(np.corrcoef(observed.ravel(), exact.ravel())[0, 1])
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "pearson_correlation": correlation,
        "relative_frobenius_error": (
            float(np.linalg.norm(delta) / denominator) if denominator else float("nan")
        ),
    }


def _classical_metrics(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for kernel in ("linear", "rbf"):
        model = SVC(kernel=kernel, C=SVC_C, gamma="scale")
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        scores = model.decision_function(x_test)
        results[kernel] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(y_test, predictions)
            ),
            "roc_auc": float(roc_auc_score(y_test, scores)),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ),
            "predictions": predictions,
            "decision_scores": scores,
        }
    return results


def simulation_condition_dir(config_name: str, fold: int) -> Path:
    return SIMULATION_DIR / config_name / f"fold_{fold}"


def _save_mode_result(
    config: Configuration,
    fold: int,
    mode: str,
    raw_train: np.ndarray,
    psd_train: np.ndarray,
    test_kernel: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    metrics: Mapping[str, Any],
    psd_diagnostics: Mapping[str, Any],
    exact_train: np.ndarray | None,
    exact_test: np.ndarray | None,
    elapsed_seconds: float,
    execution_details: Mapping[str, Any] | None = None,
) -> None:
    directory = simulation_condition_dir(config.name, fold)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / f"{mode}.npz",
        raw_train_kernel=raw_train,
        psd_train_kernel=psd_train,
        test_kernel=test_kernel,
        predictions=predictions,
        decision_scores=scores,
        y_train=y_train,
        y_test=y_test,
    )
    comparison = None
    if exact_train is not None and exact_test is not None:
        mask = ~np.eye(len(exact_train), dtype=bool)
        comparison = {
            "train_off_diagonal": _kernel_comparison(
                raw_train[mask], exact_train[mask]
            ),
            "test": _kernel_comparison(test_kernel, exact_test),
        }
    atomic_write_json(
        directory / f"{mode}.json",
        {
            "configuration": asdict(config),
            "fold": fold,
            "mode": mode,
            "shots": None if mode == "exact" else SHOTS,
            "metrics": metrics,
            "psd": psd_diagnostics,
            "kernel_vs_exact": comparison,
            "elapsed_seconds": elapsed_seconds,
            "execution_details": execution_details,
            "created_at_utc": utc_now(),
        },
    )


def simulate_exact_condition(config: Configuration, fold: int) -> None:
    directory = simulation_condition_dir(config.name, fold)
    if (directory / "exact.json").is_file() and (directory / "exact.npz").is_file():
        return
    with np.load(prepared_condition_path(config.name, fold)) as prepared:
        x_train = np.asarray(prepared["x_train"], dtype=float)
        x_test = np.asarray(prepared["x_test"], dtype=float)
        y_train = np.asarray(prepared["y_train"], dtype=int)
        y_test = np.asarray(prepared["y_test"], dtype=int)
    started = time.perf_counter()
    kernel = FidelityStatevectorKernel(
        feature_map=zz_feature_map(
            feature_dimension=config.qubits,
            reps=REPS,
            entanglement=ENTANGLEMENT,
        ),
        auto_clear_cache=False,
        enforce_psd=False,
    )
    raw_train = np.asarray(kernel.evaluate(x_train), dtype=float)
    test_kernel = np.asarray(kernel.evaluate(x_test, x_train), dtype=float)
    psd_train, psd_diagnostics = _project_psd(raw_train)
    metrics, predictions, scores = _classification_metrics(
        psd_train, test_kernel, y_train, y_test
    )
    classical = _classical_metrics(x_train, x_test, y_train, y_test)
    _save_mode_result(
        config,
        fold,
        "exact",
        raw_train,
        psd_train,
        test_kernel,
        y_train,
        y_test,
        predictions,
        scores,
        metrics,
        psd_diagnostics,
        None,
        None,
        time.perf_counter() - started,
    )
    atomic_write_json(
        directory / "classical.json",
        {
            name: {
                key: value
                for key, value in result.items()
                if key not in {"predictions", "decision_scores"}
            }
            for name, result in classical.items()
        },
    )
    np.savez_compressed(
        directory / "classical.npz",
        linear_predictions=classical["linear"]["predictions"],
        linear_decision_scores=classical["linear"]["decision_scores"],
        rbf_predictions=classical["rbf"]["predictions"],
        rbf_decision_scores=classical["rbf"]["decision_scores"],
        y_test=y_test,
    )


def write_pre_hardware_summary() -> None:
    campaign = load_or_create_campaign()
    lines = [
        "SRP2026 FROZEN PUBLICATION EXPERIMENT: PRE-HARDWARE RESULTS",
        "=" * 62,
        f"Generated (UTC): {utc_now()}",
        "",
        "This file contains duplicate-safe preparation audits plus exact quantum and",
        "matched classical baselines. It contains no ideal-Aer, noisy-Aer, or hardware",
        "claim. Final results will be written to analysis/publication_results.txt.",
        "",
        "PROTOCOL",
        "--------",
        f"Six configurations x {N_FOLDS} folds; {TRAIN_ROWS} train / {TEST_ROWS} test",
        f"zz_feature_map reps={REPS}, {ENTANGLEMENT}; SVC C={SVC_C:g}",
        f"Planned hardware: {campaign['workload']['hardware_condition_runs']} runs, "
        f"{campaign['workload']['total_hardware_executions']:,} circuit executions",
        "",
        "SOURCE-POOL AUDIT",
        "-----------------",
        "dataset | selected features | candidates | clean unique groups | duplicates removed | conflicts excluded | class 0 / class 1",
    ]
    for dataset in ("clamp", "fabiana", "cic", "ember2018"):
        summary = campaign["source_pools"][dataset]
        counts = summary["class_counts"]
        lines.append(
            f"{dataset} | {len(summary['feature_names'])} | "
            f"{summary['candidate_rows']} | {summary['unique_clean_groups']} | "
            f"{summary['duplicate_rows_removed']} | "
            f"{summary['conflicting_groups_excluded']} | "
            f"{counts.get('0', 0)} / {counts.get('1', 0)}"
        )

    lines.extend(
        [
            "",
            f"{N_FOLDS}-SPLIT BASELINES (mean +/- sample SD)",
            "------------------------------------------",
            "configuration | exact balanced accuracy | exact ROC AUC | linear balanced accuracy | RBF balanced accuracy",
        ]
    )
    exact_by_configuration: dict[str, np.ndarray] = {}
    for config in CONFIGURATIONS:
        exact_balanced = []
        exact_auc = []
        linear_balanced = []
        rbf_balanced = []
        for fold in range(N_FOLDS):
            directory = simulation_condition_dir(config.name, fold)
            exact_path = directory / "exact.json"
            classical_path = directory / "classical.json"
            if not exact_path.is_file() or not classical_path.is_file():
                raise RuntimeError(f"Missing pre-hardware result for {config.name} fold {fold}")
            exact = read_json(exact_path)["metrics"]
            classical = read_json(classical_path)
            exact_balanced.append(float(exact["balanced_accuracy"]))
            exact_auc.append(float(exact["roc_auc"]))
            linear_balanced.append(float(classical["linear"]["balanced_accuracy"]))
            rbf_balanced.append(float(classical["rbf"]["balanced_accuracy"]))
        exact_by_configuration[config.name] = np.asarray(exact_balanced)

        def mean_sd(values: Sequence[float]) -> str:
            return f"{np.mean(values):.4f} +/- {np.std(values, ddof=1):.4f}"

        lines.append(
            f"{config.name} | {mean_sd(exact_balanced)} | {mean_sd(exact_auc)} | "
            f"{mean_sd(linear_balanced)} | {mean_sd(rbf_balanced)}"
        )
        lines.append(
            "  exact balanced accuracy by fold: "
            + ", ".join(f"{value:.4f}" for value in exact_balanced)
        )

    lines.extend(
        [
            "",
            "EXACT COMPRESSION EFFECTS (PCA-4 minus native, paired by fold)",
            "--------------------------------------------------------------",
        ]
    )
    for dataset, native, pca in (
        ("clamp", "clamp_native10", "clamp_pca4"),
        ("fabiana", "fabiana_native8", "fabiana_pca4"),
    ):
        differences = exact_by_configuration[pca] - exact_by_configuration[native]
        lines.append(
            f"{dataset}: mean={np.mean(differences):.4f}, "
            f"SD={np.std(differences, ddof=1):.4f}, folds="
            + ", ".join(f"{value:.4f}" for value in differences)
        )

    submitted_attempts = sum(
        len(row.get("attempts", [])) for row in campaign["hardware_schedule"]
    )
    lines.extend(
        [
            "",
            "EXECUTION STATUS",
            "----------------",
            "Prepared primary conditions: "
            f"{len(campaign['prepared_conditions'])} / {len(CONFIGURATIONS) * N_FOLDS}",
            "Completed exact conditions: "
            f"{sum(key.endswith(':exact') and value == 'done' for key, value in campaign['simulations'].items())} / {len(CONFIGURATIONS) * N_FOLDS}",
            f"Hardware attempts currently recorded: {submitted_attempts}",
            "No hardware interpretation should be made from this file.",
        ]
    )
    atomic_write_text(RESULT_ROOT / "pre_hardware_results.txt", "\n".join(lines))


def _service() -> Any:
    from qiskit_ibm_runtime import QiskitRuntimeService

    account_service = QiskitRuntimeService(name=ACCOUNT_NAME)
    matches = [
        item
        for item in account_service.instances()
        if item.get("name") == INSTANCE_NAME and item.get("plan") == "on-prem"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one on-prem instance named {INSTANCE_NAME!r}; found {len(matches)}"
        )
    return QiskitRuntimeService(name=ACCOUNT_NAME, instance=matches[0]["crn"])


def _fidelity_circuit(width: int) -> tuple[QuantumCircuit, ParameterVector, ParameterVector]:
    left_parameters = ParameterVector("left", width)
    right_parameters = ParameterVector("right", width)
    left = zz_feature_map(
        feature_dimension=width, reps=REPS, entanglement=ENTANGLEMENT
    ).assign_parameters(left_parameters)
    right = zz_feature_map(
        feature_dimension=width, reps=REPS, entanglement=ENTANGLEMENT
    ).assign_parameters(right_parameters)
    circuit = left.compose(right.inverse())
    circuit.measure_all()
    return circuit, left_parameters, right_parameters


def circuit_path(width: int) -> Path:
    return CIRCUIT_DIR / f"fidelity_{width}q.qpy"


def circuit_metadata_path(width: int) -> Path:
    return CIRCUIT_DIR / f"fidelity_{width}q.json"


def _layout_value(circuit: QuantumCircuit) -> Any:
    try:
        return circuit.layout.final_index_layout(filter_ancillas=True)
    except Exception:
        return str(circuit.layout)


def ensure_transpiled_circuits(backend: Any, campaign: dict[str, Any]) -> None:
    CIRCUIT_DIR.mkdir(parents=True, exist_ok=True)
    properties = backend.properties()
    if properties is not None and not (CIRCUIT_DIR / "initial_backend_properties.json").exists():
        atomic_write_json(
            CIRCUIT_DIR / "initial_backend_properties.json",
            properties.to_dict(),
        )
    for width in (4, 8, 10):
        if circuit_path(width).is_file() and circuit_metadata_path(width).is_file():
            campaign["circuits"][str(width)] = read_json(circuit_metadata_path(width))
            continue
        raw, left, right = _fidelity_circuit(width)
        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=OPTIMIZATION_LEVEL,
            seed_transpiler=TRANSPILER_SEED,
        )
        isa = pass_manager.run(raw)
        if set(isa.parameters) != set(left) | set(right):
            raise RuntimeError(f"Transpilation changed {width}q parameter set")
        with circuit_path(width).open("wb") as handle:
            qpy.dump(isa, handle)
        try:
            estimated_duration_seconds: float | None = float(
                isa.estimate_duration(backend.target, unit="s")
            )
        except Exception:
            estimated_duration_seconds = None
        metadata = {
            "width": width,
            "backend": backend.name,
            "optimization_level": OPTIMIZATION_LEVEL,
            "transpiler_seed": TRANSPILER_SEED,
            "depth": int(isa.depth()),
            "size": int(isa.size()),
            "estimated_duration_seconds": estimated_duration_seconds,
            "backend_circuit_qubits": int(isa.num_qubits),
            "logical_qubits": width,
            "operations": {key: int(value) for key, value in isa.count_ops().items()},
            "final_index_layout": _layout_value(isa),
            "created_at_utc": utc_now(),
        }
        atomic_write_json(circuit_metadata_path(width), metadata)
        campaign["circuits"][str(width)] = metadata
        save_campaign(campaign)


def load_transpiled_circuit(width: int) -> QuantumCircuit:
    with circuit_path(width).open("rb") as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise RuntimeError(f"Expected one {width}q QPY circuit")
    return circuits[0]


def aer_circuit_path(width: int) -> Path:
    return CIRCUIT_DIR / f"aer_active_fidelity_{width}q.qpy"


def aer_circuit_metadata_path(width: int) -> Path:
    return CIRCUIT_DIR / f"aer_active_fidelity_{width}q.json"


def aer_noise_model_path(width: int) -> Path:
    return CIRCUIT_DIR / f"aer_active_noise_model_{width}q.json"


def _dense_active_circuit(
    circuit: QuantumCircuit,
) -> tuple[QuantumCircuit, dict[int, int]]:
    """Remove idle backend wires while retaining the physical-wire identity.

    IBM ISA circuits contain every backend qubit even though this experiment
    acts on only 4, 8, or 10 of them.  Aer otherwise propagates a 120-wire MPS
    for millions of noisy shots.  Relabeling the active physical wires densely
    is circuit-isomorphic; the returned mapping is also used to relabel the
    corresponding physical-qubit noise channels.
    """

    active_physical = sorted(
        {
            circuit.find_bit(qubit).index
            for instruction in circuit.data
            for qubit in instruction.qubits
        }
    )
    physical_to_dense = {
        physical: dense for dense, physical in enumerate(active_physical)
    }
    dense = QuantumCircuit(
        len(active_physical),
        circuit.num_clbits,
        name=f"{circuit.name}_active",
    )
    dense.global_phase = circuit.global_phase
    for instruction in circuit.data:
        qargs = [
            dense.qubits[
                physical_to_dense[circuit.find_bit(qubit).index]
            ]
            for qubit in instruction.qubits
        ]
        cargs = [
            dense.clbits[circuit.find_bit(clbit).index]
            for clbit in instruction.clbits
        ]
        dense.append(instruction.operation, qargs, cargs)
    if set(dense.parameters) != set(circuit.parameters):
        raise RuntimeError("Active-wire relabeling changed circuit parameters")
    if dense.count_ops() != circuit.count_ops() or dense.depth() != circuit.depth():
        raise RuntimeError("Active-wire relabeling changed circuit resources")
    return dense, physical_to_dense


def _dense_active_noise_model(
    noise_model: NoiseModel,
    physical_to_dense: Mapping[int, int],
) -> tuple[NoiseModel, dict[str, Any]]:
    """Restrict a backend noise model to active wires and relabel them densely."""

    dense = NoiseModel(basis_gates=noise_model.basis_gates)
    for instruction, error in noise_model._default_quantum_errors.items():
        dense.add_all_qubit_quantum_error(error, instruction, warnings=False)
    copied_quantum_errors = 0
    copied_by_instruction: Counter[str] = Counter()
    for instruction, qubit_errors in noise_model._local_quantum_errors.items():
        for physical_qubits, error in qubit_errors.items():
            if all(qubit in physical_to_dense for qubit in physical_qubits):
                dense.add_quantum_error(
                    error,
                    instruction,
                    [physical_to_dense[qubit] for qubit in physical_qubits],
                    warnings=False,
                )
                copied_quantum_errors += 1
                copied_by_instruction[instruction] += 1
    if noise_model._default_readout_error is not None:
        dense.add_all_qubit_readout_error(
            noise_model._default_readout_error, warnings=False
        )
    copied_readout_errors = 0
    for physical_qubits, error in noise_model._local_readout_errors.items():
        if all(qubit in physical_to_dense for qubit in physical_qubits):
            dense.add_readout_error(
                error,
                [physical_to_dense[qubit] for qubit in physical_qubits],
                warnings=False,
            )
            copied_readout_errors += 1
    metadata = {
        "physical_to_dense": {
            str(physical): dense_index
            for physical, dense_index in physical_to_dense.items()
        },
        "copied_quantum_errors": copied_quantum_errors,
        "copied_quantum_errors_by_instruction": dict(copied_by_instruction),
        "copied_readout_errors": copied_readout_errors,
        "omitted_custom_noise_passes": len(noise_model._custom_noise_passes),
        "custom_pass_note": (
            "Backend delay-relaxation passes are omitted because the frozen ISA "
            "fidelity circuits contain no delay instructions."
        ),
    }
    return dense, metadata


def prepare_aer_execution(
    width: int,
    noise_model: NoiseModel,
) -> tuple[QuantumCircuit, AerSimulator, AerSimulator]:
    """Build auditable dense-wire ideal and noisy Aer execution objects."""

    hardware_circuit = load_transpiled_circuit(width)
    if hardware_circuit.count_ops().get("delay", 0):
        raise RuntimeError("Dense Aer path does not permit delay instructions")
    active_circuit, physical_to_dense = _dense_active_circuit(hardware_circuit)
    active_noise, noise_metadata = _dense_active_noise_model(
        noise_model, physical_to_dense
    )
    with aer_circuit_path(width).open("wb") as handle:
        qpy.dump(active_circuit, handle)
    metadata = {
        "logical_qubits": width,
        "hardware_circuit_qubits": hardware_circuit.num_qubits,
        "aer_circuit_qubits": active_circuit.num_qubits,
        "classical_bits": active_circuit.num_clbits,
        "depth": active_circuit.depth(),
        "size": active_circuit.size(),
        "operations": {
            key: int(value) for key, value in active_circuit.count_ops().items()
        },
        **noise_metadata,
        "created_at_utc": utc_now(),
    }
    atomic_write_json(aer_circuit_metadata_path(width), metadata)
    atomic_write_json(
        aer_noise_model_path(width), active_noise.to_dict(serializable=True)
    )
    ideal = AerSimulator(method="statevector")
    noisy = AerSimulator(
        method="statevector",
        noise_model=active_noise,
        shot_branching_enable=True,
        shot_branching_sampling_enable=True,
        max_parallel_experiments=1,
    )
    return active_circuit, ideal, noisy


def load_frozen_aer_execution(
    width: int,
) -> tuple[QuantumCircuit, AerSimulator, AerSimulator]:
    """Reload the exact dense circuit and noise model used by an interrupted run."""

    circuit_path_value = aer_circuit_path(width)
    metadata_path = aer_circuit_metadata_path(width)
    noise_path = aer_noise_model_path(width)
    missing = [
        str(path)
        for path in (circuit_path_value, metadata_path, noise_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Cannot resume Aer with the frozen calibration; missing artifacts: "
            + ", ".join(missing)
        )

    with circuit_path_value.open("rb") as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise RuntimeError(f"Expected one frozen {width}q Aer QPY circuit")
    circuit = circuits[0]
    metadata = read_json(metadata_path)
    if int(metadata.get("logical_qubits", -1)) != width:
        raise RuntimeError(f"Frozen {width}q Aer metadata has the wrong width")
    if int(metadata.get("aer_circuit_qubits", -1)) != circuit.num_qubits:
        raise RuntimeError(f"Frozen {width}q Aer circuit does not match its metadata")
    expected_operations = {
        key: int(value) for key, value in metadata.get("operations", {}).items()
    }
    observed_operations = {
        key: int(value) for key, value in circuit.count_ops().items()
    }
    if expected_operations != observed_operations:
        raise RuntimeError(f"Frozen {width}q Aer circuit operations changed")

    serialized_noise = read_json(noise_path)
    for error in serialized_noise.get("errors", []):
        for instruction_sequence in error.get("instructions", []):
            for operation in instruction_sequence:
                if operation.get("name") != "kraus":
                    continue
                operation["params"] = [
                    [
                        [
                            complex(float(entry[0]), float(entry[1]))
                            if (
                                isinstance(entry, list)
                                and len(entry) == 2
                                and all(
                                    isinstance(value, (int, float))
                                    for value in entry
                                )
                            )
                            else entry
                            for entry in row
                        ]
                        for row in matrix
                    ]
                    for matrix in operation.get("params", [])
                ]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="from_dict has been deprecated",
            category=DeprecationWarning,
        )
        noise_model = NoiseModel.from_dict(serialized_noise)
    if any(qubit < 0 or qubit >= circuit.num_qubits for qubit in noise_model.noise_qubits):
        raise RuntimeError(f"Frozen {width}q Aer noise model has out-of-range qubits")
    ideal = AerSimulator(method="statevector")
    noisy = AerSimulator(
        method="statevector",
        noise_model=noise_model,
        shot_branching_enable=True,
        shot_branching_sampling_enable=True,
        max_parallel_experiments=1,
    )
    return circuit, ideal, noisy


_PARAMETER_PATTERN = re.compile(r"^(left|right)\[(\d+)]$")


def parameter_values(
    circuit: QuantumCircuit,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> np.ndarray:
    if len(left_values) != len(right_values):
        raise ValueError("Left/right parameter batches differ")
    columns: list[np.ndarray] = []
    for parameter in circuit.parameters:
        match = _PARAMETER_PATTERN.match(parameter.name)
        if not match:
            raise ValueError(f"Unexpected parameter name {parameter.name!r}")
        side, index_text = match.groups()
        values = left_values if side == "left" else right_values
        columns.append(values[:, int(index_text)])
    return np.column_stack(columns)


def condition_parameter_values(
    config_name: str,
    fold: int,
    circuit: QuantumCircuit,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(prepared_condition_path(config_name, fold)) as prepared:
        x_train = np.asarray(prepared["x_train"], dtype=float)
        x_test = np.asarray(prepared["x_test"], dtype=float)
        train_pairs = np.asarray(prepared["train_pairs"], dtype=int)
        test_pairs = np.asarray(prepared["test_pairs"], dtype=int)
    return (
        parameter_values(
            circuit,
            x_train[train_pairs[:, 0]],
            x_train[train_pairs[:, 1]],
        ),
        parameter_values(
            circuit,
            x_test[test_pairs[:, 0]],
            x_train[test_pairs[:, 1]],
        ),
    )


def _fidelities(pub_result: Any, expected: int) -> tuple[np.ndarray, int]:
    bit_array = pub_result.join_data()
    if bit_array.size != expected:
        raise RuntimeError(f"Expected {expected} results; received {bit_array.size}")
    shots = int(bit_array.num_shots)
    values = np.asarray(
        [bit_array.get_int_counts(index).get(0, 0) / shots for index in range(expected)],
        dtype=float,
    )
    return values, shots


def kernels_from_fidelities(
    train_fidelities: np.ndarray,
    test_fidelities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_pairs, test_pairs = train_test_pairs(TRAIN_ROWS, TEST_ROWS)
    if len(train_fidelities) != len(train_pairs) or len(test_fidelities) != len(test_pairs):
        raise ValueError("Fidelity result length mismatch")
    train = np.eye(TRAIN_ROWS, dtype=float)
    for value, (left, right) in zip(train_fidelities, train_pairs):
        train[left, right] = value
        train[right, left] = value
    test = np.empty((TEST_ROWS, TRAIN_ROWS), dtype=float)
    for value, (left, right) in zip(test_fidelities, test_pairs):
        test[left, right] = value
    return train, test


def _run_aer_pubs(
    simulator: AerSimulator,
    circuit: QuantumCircuit,
    train_values: np.ndarray,
    test_values: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    def run_chunks(values: np.ndarray, series: str) -> np.ndarray:
        chunks: list[np.ndarray] = []
        total = len(values)
        for start in range(0, total, AER_PARAMETER_CHUNK_SIZE):
            end = min(start + AER_PARAMETER_CHUNK_SIZE, total)
            chunk_seed = stable_seed(
                f"aer:{seed}:{series}:{start}:{end}", base=seed
            )
            sampler = AerSamplerV2.from_backend(simulator, seed=chunk_seed)
            result = sampler.run(
                [(circuit, values[start:end])], shots=SHOTS
            ).result()
            fidelities, observed_shots = _fidelities(result[0], end - start)
            if observed_shots != SHOTS:
                raise RuntimeError("Aer returned an unexpected shot count")
            chunks.append(fidelities)
            if end == total or end % (8 * AER_PARAMETER_CHUNK_SIZE) == 0:
                print(f"  Aer {series}: {end:,}/{total:,} fidelities", flush=True)
        return np.concatenate(chunks)

    train_fidelities = run_chunks(train_values, "train")
    test_fidelities = run_chunks(test_values, "test")
    return kernels_from_fidelities(train_fidelities, test_fidelities)


def simulate_aer_condition(
    config: Configuration,
    fold: int,
    mode: str,
    simulator: AerSimulator,
    circuit: QuantumCircuit,
) -> None:
    if mode not in {"ideal_aer", "noisy_aer"}:
        raise ValueError(mode)
    directory = simulation_condition_dir(config.name, fold)
    if (directory / f"{mode}.json").is_file() and (directory / f"{mode}.npz").is_file():
        return
    exact_json = directory / "exact.json"
    exact_npz = directory / "exact.npz"
    if not exact_json.is_file() or not exact_npz.is_file():
        raise RuntimeError("Exact simulation must precede Aer")
    with np.load(prepared_condition_path(config.name, fold)) as prepared:
        y_train = np.asarray(prepared["y_train"], dtype=int)
        y_test = np.asarray(prepared["y_test"], dtype=int)
    with np.load(exact_npz) as exact:
        exact_train = np.asarray(exact["raw_train_kernel"], dtype=float)
        exact_test = np.asarray(exact["test_kernel"], dtype=float)
    train_values, test_values = condition_parameter_values(config.name, fold, circuit)
    started = time.perf_counter()
    raw_train, test_kernel = _run_aer_pubs(
        simulator,
        circuit,
        train_values,
        test_values,
        stable_seed(f"{config.name}:fold{fold}:{mode}"),
    )
    psd_train, psd_diagnostics = _project_psd(raw_train)
    metrics, predictions, scores = _classification_metrics(
        psd_train, test_kernel, y_train, y_test
    )
    _save_mode_result(
        config,
        fold,
        mode,
        raw_train,
        psd_train,
        test_kernel,
        y_train,
        y_test,
        predictions,
        scores,
        metrics,
        psd_diagnostics,
        exact_train,
        exact_test,
        time.perf_counter() - started,
        {
            "circuit": str(aer_circuit_path(config.qubits)),
            "circuit_metadata": str(aer_circuit_metadata_path(config.qubits)),
            "noise_model": (
                None
                if mode == "ideal_aer"
                else str(aer_noise_model_path(config.qubits))
            ),
            "active_wire_relabeling": True,
            "simulator_method": "statevector",
            "shot_branching": mode == "noisy_aer",
            "parameter_chunk_size": AER_PARAMETER_CHUNK_SIZE,
            "max_parallel_experiments": 1 if mode == "noisy_aer" else "automatic",
        },
    )


def simulate_all(include_aer: bool = True) -> None:
    campaign = load_or_create_campaign()
    if campaign.get("state") == "planned":
        raise RuntimeError("Run prepare before simulate")
    for config in CONFIGURATIONS:
        for fold in range(N_FOLDS):
            print(f"Exact simulation: {config.name} fold {fold}", flush=True)
            simulate_exact_condition(config, fold)
            campaign["simulations"][f"{config.name}__fold{fold}:exact"] = "done"
            save_campaign(campaign)
    write_pre_hardware_summary()
    if include_aer:
        simulators: dict[tuple[int, str], AerSimulator] = {}
        aer_circuits: dict[int, QuantumCircuit] = {}
        completed_aer_results = [
            (config.name, fold, mode)
            for config in CONFIGURATIONS
            for fold in range(N_FOLDS)
            for mode in ("ideal_aer", "noisy_aer")
            if (
                simulation_condition_dir(config.name, fold) / f"{mode}.json"
            ).is_file()
            and (
                simulation_condition_dir(config.name, fold) / f"{mode}.npz"
            ).is_file()
        ]
        expected_aer_results = len(CONFIGURATIONS) * N_FOLDS * 2
        missing_aer_results = [
            (config.name, config.qubits, fold, mode)
            for config in CONFIGURATIONS
            for fold in range(N_FOLDS)
            for mode in ("ideal_aer", "noisy_aer")
            if not (
                (
                    simulation_condition_dir(config.name, fold)
                    / f"{mode}.json"
                ).is_file()
                and (
                    simulation_condition_dir(config.name, fold)
                    / f"{mode}.npz"
                ).is_file()
            )
        ]
        if len(completed_aer_results) < expected_aer_results:
            if completed_aer_results:
                print(
                    "Resuming Aer with the frozen dense circuits and noise model "
                    f"({len(completed_aer_results)}/{expected_aer_results} complete)",
                    flush=True,
                )
                required_widths = sorted(
                    {width for _, width, _, _ in missing_aer_results}
                )
                for width in required_widths:
                    circuit, ideal, noisy = load_frozen_aer_execution(width)
                    aer_circuits[width] = circuit
                    simulators[(width, "ideal_aer")] = ideal
                    simulators[(width, "noisy_aer")] = noisy
            else:
                service = _service()
                backend = service.backend(BACKEND_NAME)
                if not backend.status().operational:
                    raise RuntimeError(f"Backend {BACKEND_NAME} is not operational")
                ensure_transpiled_circuits(backend, campaign)
                noise_model = NoiseModel.from_backend(backend)
                atomic_write_json(
                    CIRCUIT_DIR / "active_noise_source_snapshot.json",
                    noise_model.to_dict(serializable=True),
                )
                for width in (4, 8, 10):
                    circuit, ideal, noisy = prepare_aer_execution(
                        width, noise_model
                    )
                    aer_circuits[width] = circuit
                    simulators[(width, "ideal_aer")] = ideal
                    simulators[(width, "noisy_aer")] = noisy
        for config in CONFIGURATIONS:
            for fold in range(N_FOLDS):
                for mode in ("ideal_aer", "noisy_aer"):
                    print(f"{mode}: {config.name} fold {fold}", flush=True)
                    directory = simulation_condition_dir(config.name, fold)
                    if (
                        (directory / f"{mode}.json").is_file()
                        and (directory / f"{mode}.npz").is_file()
                    ):
                        campaign["simulations"][
                            f"{config.name}__fold{fold}:{mode}"
                        ] = "done"
                        continue
                    circuit = aer_circuits[config.qubits]
                    simulate_aer_condition(
                        config,
                        fold,
                        mode,
                        simulators[(config.qubits, mode)],
                        circuit,
                    )
                    campaign["simulations"][f"{config.name}__fold{fold}:{mode}"] = "done"
                    save_campaign(campaign)
    campaign["state"] = "simulated" if include_aer else "exact_simulated"
    save_campaign(campaign)


def _status_name(status: Any) -> str:
    return getattr(status, "name", str(status))


def _schedule_entry(campaign: dict[str, Any], condition_id: str) -> dict[str, Any]:
    for row in campaign["hardware_schedule"]:
        if row["condition_id"] == condition_id:
            return row
    raise KeyError(condition_id)


def _attempts(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row.setdefault("attempts", [])


def _pending_schedule(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    for row in campaign["hardware_schedule"]:
        attempts = _attempts(row)
        if any(attempt.get("status") == "DONE" for attempt in attempts):
            continue
        active = any(
            attempt.get("status") not in TERMINAL_JOB_STATES for attempt in attempts
        )
        if active:
            continue
        if len(attempts) < MAX_ATTEMPTS:
            pending.append(row)
    return pending


def _save_backend_snapshot(backend: Any, path: Path) -> None:
    properties = backend.properties()
    value = {
        "captured_at_utc": utc_now(),
        "backend": backend.name,
        "status": {
            "operational": backend.status().operational,
            "pending_jobs": backend.status().pending_jobs,
        },
        "properties": properties.to_dict() if properties is not None else None,
    }
    atomic_write_json(path, value)


def submit_next_batch(
    authorize_qpu: bool,
    allow_concurrent_simulation: bool = False,
) -> int:
    if not authorize_qpu:
        raise RuntimeError(
            "Real-QPU submission requires the literal --authorize-qpu flag"
        )
    from qiskit_ibm_runtime import Batch, SamplerV2

    campaign = load_or_create_campaign()
    missing_exact = [
        key
        for key in (
            f"{config.name}__fold{fold}:exact"
            for config in CONFIGURATIONS
            for fold in range(N_FOLDS)
        )
        if campaign["simulations"].get(key) != "done"
    ]
    if missing_exact:
        raise RuntimeError(
            "Refusing QPU submission; "
            f"{len(missing_exact)} exact simulations are incomplete"
        )
    missing_aer = [
        key
        for key in (
            f"{config.name}__fold{fold}:{mode}"
            for config in CONFIGURATIONS
            for fold in range(N_FOLDS)
            for mode in ("ideal_aer", "noisy_aer")
        )
        if campaign["simulations"].get(key) != "done"
    ]
    if missing_aer and not allow_concurrent_simulation:
        raise RuntimeError(
            "Refusing QPU submission; "
            f"{len(missing_aer)} Aer simulations are incomplete. Pass the literal "
            "--allow-concurrent-simulation flag only when the local simulator "
            "will be resumed after submission."
        )
    if missing_aer:
        print(
            f"Concurrent submission authorized with {len(missing_aer)} Aer "
            "simulations still pending; all prepared data and exact kernels are complete.",
            flush=True,
        )
    service = _service()
    backend = service.backend(BACKEND_NAME)
    status = backend.status()
    if not status.operational:
        raise RuntimeError(f"Backend {BACKEND_NAME} is not operational")
    ensure_transpiled_circuits(backend, campaign)
    pending = _pending_schedule(campaign)[:CONDITIONS_PER_BATCH]
    if not pending:
        print("No pending QPU conditions", flush=True)
        return 0

    batch_record = {
        "opened_at_utc": utc_now(),
        "backend": BACKEND_NAME,
        "condition_ids": [row["condition_id"] for row in pending],
        "concurrent_simulation_authorized": bool(missing_aer),
        "aer_simulations_pending_at_open": len(missing_aer),
        "jobs": [],
    }
    campaign["batches"].append(batch_record)
    save_campaign(campaign)
    with Batch(backend=backend, max_time="8h") as batch:
        batch_record["batch_id"] = getattr(batch, "session_id", None)
        save_campaign(campaign)
        sampler = SamplerV2(mode=batch)
        sampler.options.max_execution_time = MAX_EXECUTION_TIME_SECONDS
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
        sampler.options.execution.meas_type = "classified"
        for row in pending:
            config = CONFIG_BY_NAME[row["configuration"]]
            fold = int(row["fold"])
            attempt_number = len(_attempts(row)) + 1
            attempt_dir = (
                HARDWARE_DIR
                / row["condition_id"]
                / f"attempt_{attempt_number}"
            )
            attempt_dir.mkdir(parents=True, exist_ok=True)
            _save_backend_snapshot(backend, attempt_dir / "backend_before_submission.json")
            circuit = load_transpiled_circuit(config.qubits)
            train_values, test_values = condition_parameter_values(
                config.name, fold, circuit
            )
            executions = (len(train_values) + len(test_values)) * SHOTS
            if executions >= MAX_EXECUTIONS_PER_JOB:
                raise RuntimeError(
                    f"{row['condition_id']} has {executions:,} executions"
                )
            job = sampler.run(
                [(circuit, train_values), (circuit, test_values)],
                shots=SHOTS,
            )
            attempt = {
                "attempt": attempt_number,
                "job_id": job.job_id(),
                "batch_id": batch_record.get("batch_id"),
                "submitted_at_utc": utc_now(),
                "status": "SUBMITTED",
                "shots": SHOTS,
                "train_evaluations": len(train_values),
                "test_evaluations": len(test_values),
                "total_executions": executions,
                "directory": str(attempt_dir),
            }
            _attempts(row).append(attempt)
            batch_record["jobs"].append(
                {"condition_id": row["condition_id"], "job_id": job.job_id()}
            )
            atomic_write_json(attempt_dir / "submission.json", attempt)
            save_campaign(campaign)
            print(
                f"Submitted {row['condition_id']} as {job.job_id()}", flush=True
            )
    batch_record["closed_at_utc"] = utc_now()
    campaign["state"] = "hardware_submitted"
    save_campaign(campaign)
    return len(pending)


_API_STATUS_TO_JOB_STATUS = {
    "QUEUED": "QUEUED",
    "RUNNING": "RUNNING",
    "COMPLETED": "DONE",
    "FAILED": "ERROR",
    "CANCELLED": "CANCELLED",
}


def _batch_job_snapshots(
    service: Any,
    campaign: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Fetch one compact job-status page per Runtime batch.

    RuntimeJobV2.status() performs a separate HTTP request for every job.  The
    publication campaign submits jobs in batches, so the Runtime batch-list API
    gives the same authoritative state in two requests instead of 21.
    """

    wanted = {
        attempt["job_id"]
        for row in campaign["hardware_schedule"]
        for attempt in row.get("attempts", [])
    }
    snapshots: dict[str, dict[str, Any]] = {}
    for batch in campaign.get("batches", []):
        batch_id = batch.get("batch_id")
        if not batch_id:
            continue
        response = service._active_api_client.jobs_get(
            limit=max(CONDITIONS_PER_BATCH, len(batch.get("jobs", []))),
            session_id=batch_id,
            descending=False,
        )
        for raw_job in response.get("jobs", []):
            job_id = raw_job.get("id")
            if job_id in wanted:
                snapshots[job_id] = raw_job
    return snapshots


def refresh_hardware_status(service: Any | None = None) -> dict[str, int]:
    campaign = load_or_create_campaign()
    service = service or _service()
    snapshots = _batch_job_snapshots(service, campaign)
    counts: Counter[str] = Counter()
    for row in campaign["hardware_schedule"]:
        for attempt in _attempts(row):
            raw_job = snapshots.get(attempt["job_id"])
            if raw_job is None:
                status = attempt.get("status", "UNKNOWN")
                attempt["status_query_warning"] = (
                    "Job was absent from its Runtime batch listing"
                )
                counts[status] += 1
                continue
            api_status = str(raw_job.get("state", {}).get("status", "UNKNOWN")).upper()
            status = _API_STATUS_TO_JOB_STATUS.get(api_status, api_status)
            attempt["status"] = status
            attempt["status_checked_at_utc"] = utc_now()
            if status in {"ERROR", "CANCELLED"}:
                state = raw_job.get("state", {})
                attempt["error_message"] = state.get("reason") or state.get(
                    "reasonCode", "No reason returned by Runtime"
                )
            counts[status] += 1
    save_campaign(campaign)
    print(json.dumps(dict(counts), indent=2), flush=True)
    return dict(counts)


def wait_for_active_jobs(poll_seconds: int = 30) -> None:
    service = _service()
    while True:
        counts = refresh_hardware_status(service=service)
        active = sum(
            count for status, count in counts.items() if status not in TERMINAL_JOB_STATES
        )
        if active == 0:
            return
        print(f"{utc_now()} active_jobs={active}", flush=True)
        time.sleep(poll_seconds)


def _hardware_result_dir(row: Mapping[str, Any], attempt: Mapping[str, Any]) -> Path:
    return Path(str(attempt["directory"]))


def retrieve_done_jobs(service: Any | None = None) -> int:
    campaign = load_or_create_campaign()
    service = service or _service()
    snapshots = _batch_job_snapshots(service, campaign)
    retrieved = 0
    for row in campaign["hardware_schedule"]:
        config = CONFIG_BY_NAME[row["configuration"]]
        fold = int(row["fold"])
        for attempt in _attempts(row):
            if attempt.get("status") != "DONE" or attempt.get("retrieved_at_utc"):
                continue
            raw_job = snapshots.get(attempt["job_id"])
            if raw_job is None:
                job = service.job(attempt["job_id"])
            else:
                job = service._decode_job(raw_job)
                job._set_status(raw_job)
                job._set_error_message(raw_job)
            result = job.result()
            train_values, train_shots = _fidelities(
                result[0], TRAIN_ROWS * (TRAIN_ROWS - 1) // 2
            )
            test_values, test_shots = _fidelities(
                result[1], TRAIN_ROWS * TEST_ROWS
            )
            if train_shots != SHOTS or test_shots != SHOTS:
                raise RuntimeError(f"Unexpected shots for {row['condition_id']}")
            raw_train, test_kernel = kernels_from_fidelities(
                train_values, test_values
            )
            psd_train, psd_diagnostics = _project_psd(raw_train)
            with np.load(prepared_condition_path(config.name, fold)) as prepared:
                y_train = np.asarray(prepared["y_train"], dtype=int)
                y_test = np.asarray(prepared["y_test"], dtype=int)
            exact_path = simulation_condition_dir(config.name, fold) / "exact.npz"
            with np.load(exact_path) as exact:
                exact_train = np.asarray(exact["raw_train_kernel"], dtype=float)
                exact_test = np.asarray(exact["test_kernel"], dtype=float)
                exact_predictions = np.asarray(exact["predictions"], dtype=int)
            metrics, predictions, scores = _classification_metrics(
                psd_train, test_kernel, y_train, y_test
            )
            mask = ~np.eye(TRAIN_ROWS, dtype=bool)
            comparison = {
                "train_off_diagonal": _kernel_comparison(
                    raw_train[mask], exact_train[mask]
                ),
                "test": _kernel_comparison(test_kernel, exact_test),
                "prediction_disagreement_fraction": float(
                    np.mean(predictions != exact_predictions)
                ),
            }
            directory = _hardware_result_dir(row, attempt)
            np.savez_compressed(
                directory / "results.npz",
                raw_train_kernel=raw_train,
                psd_train_kernel=psd_train,
                test_kernel=test_kernel,
                train_fidelities=train_values,
                test_fidelities=test_values,
                predictions=predictions,
                decision_scores=scores,
                y_train=y_train,
                y_test=y_test,
            )
            job_details: dict[str, Any] = {}
            try:
                job_details["metrics"] = job.metrics()
                usage = job_details["metrics"].get("usage", {})
                if usage.get("status", "pending") != "pending":
                    job_details["usage_seconds"] = usage.get(
                        "qpu_charge_time_seconds"
                    )
                else:
                    job_details["usage_seconds"] = 0
            except Exception as error:
                message = f"Unavailable: {type(error).__name__}: {error}"
                job_details["metrics"] = message
                job_details["usage_seconds"] = message
            summary = {
                "condition": row,
                "attempt": attempt,
                "configuration": asdict(config),
                "mode": "hardware",
                "metrics": metrics,
                "psd": psd_diagnostics,
                "kernel_vs_exact": comparison,
                "job_details": job_details,
                "retrieved_at_utc": utc_now(),
            }
            atomic_write_json(directory / "summary.json", summary)
            if isinstance(job_details.get("usage_seconds"), (int, float)):
                attempt["qpu_seconds"] = job_details["usage_seconds"]
            attempt["retrieved_at_utc"] = summary["retrieved_at_utc"]
            attempt["summary"] = str(directory / "summary.json")
            retrieved += 1
            save_campaign(campaign)
    print(f"Retrieved {retrieved} completed jobs", flush=True)
    return retrieved


def _all_hardware_complete(campaign: Mapping[str, Any]) -> bool:
    return all(
        any(
            attempt.get("status") == "DONE" and attempt.get("retrieved_at_utc")
            for attempt in row.get("attempts", [])
        )
        for row in campaign["hardware_schedule"]
    )


def run_all(authorize_qpu: bool) -> None:
    prepare_all()
    simulate_all(include_aer=True)
    while True:
        campaign = load_or_create_campaign()
        if _all_hardware_complete(campaign):
            break
        submitted = submit_next_batch(authorize_qpu=authorize_qpu)
        if submitted == 0:
            refresh_hardware_status()
            retrieve_done_jobs()
            campaign = load_or_create_campaign()
            if _all_hardware_complete(campaign):
                break
            failed_permanently = [
                row["condition_id"]
                for row in campaign["hardware_schedule"]
                if len(row.get("attempts", [])) >= MAX_ATTEMPTS
                and not any(
                    attempt.get("status") == "DONE"
                    for attempt in row.get("attempts", [])
                )
            ]
            if failed_permanently:
                raise RuntimeError(
                    "Hardware conditions failed twice: "
                    + ", ".join(failed_permanently)
                )
            time.sleep(30)
            continue
        wait_for_active_jobs()
        retrieve_done_jobs()
    analyze_results()


def _mode_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for config in CONFIGURATIONS:
        for fold in range(N_FOLDS):
            with np.load(prepared_condition_path(config.name, fold)) as prepared:
                source_ids = np.asarray(prepared["test_source_ids"], dtype=str)
                y_test = np.asarray(prepared["y_test"], dtype=int)
            for mode in ("exact", "ideal_aer", "noisy_aer"):
                summary_path = simulation_condition_dir(config.name, fold) / f"{mode}.json"
                result_path = simulation_condition_dir(config.name, fold) / f"{mode}.npz"
                if not summary_path.is_file() or not result_path.is_file():
                    continue
                summary = read_json(summary_path)
                with np.load(result_path) as result:
                    predictions = np.asarray(result["predictions"], dtype=int)
                    scores = np.asarray(result["decision_scores"], dtype=float)
                metric_rows.append(
                    {
                        "configuration": config.name,
                        "dataset": config.dataset,
                        "representation": config.representation,
                        "qubits": config.qubits,
                        "fold": fold,
                        "repeat": 0,
                        "mode": mode,
                        **summary["metrics"],
                        "test_relative_frobenius_error": (
                            0.0
                            if mode == "exact"
                            else summary["kernel_vs_exact"]["test"][
                                "relative_frobenius_error"
                            ]
                        ),
                        "test_kernel_correlation": (
                            1.0
                            if mode == "exact"
                            else summary["kernel_vs_exact"]["test"][
                                "pearson_correlation"
                            ]
                        ),
                    }
                )
                for index, (source_id, truth, prediction, score) in enumerate(
                    zip(source_ids, y_test, predictions, scores)
                ):
                    prediction_rows.append(
                        {
                            "configuration": config.name,
                            "dataset": config.dataset,
                            "representation": config.representation,
                            "qubits": config.qubits,
                            "fold": fold,
                            "repeat": 0,
                            "mode": mode,
                            "test_position": index,
                            "source_id": source_id,
                            "truth": int(truth),
                            "prediction": int(prediction),
                            "decision_score": float(score),
                        }
                    )

    campaign = load_or_create_campaign()
    for row in campaign["hardware_schedule"]:
        successful = [
            attempt
            for attempt in row.get("attempts", [])
            if attempt.get("status") == "DONE" and attempt.get("summary")
        ]
        if not successful:
            continue
        attempt = successful[-1]
        summary = read_json(Path(attempt["summary"]))
        result_path = Path(attempt["directory"]) / "results.npz"
        config = CONFIG_BY_NAME[row["configuration"]]
        fold = int(row["fold"])
        with np.load(prepared_condition_path(config.name, fold)) as prepared:
            source_ids = np.asarray(prepared["test_source_ids"], dtype=str)
            y_test = np.asarray(prepared["y_test"], dtype=int)
        with np.load(result_path) as result:
            predictions = np.asarray(result["predictions"], dtype=int)
            scores = np.asarray(result["decision_scores"], dtype=float)
        usage = summary.get("job_details", {}).get("usage_seconds")
        if not isinstance(usage, (int, float)):
            usage = attempt.get("qpu_seconds")
        metric_rows.append(
            {
                "configuration": config.name,
                "dataset": config.dataset,
                "representation": config.representation,
                "qubits": config.qubits,
                "fold": fold,
                "repeat": int(row["repeat"]),
                "mode": "hardware",
                **summary["metrics"],
                "test_relative_frobenius_error": summary["kernel_vs_exact"][
                    "test"
                ]["relative_frobenius_error"],
                "test_kernel_correlation": summary["kernel_vs_exact"]["test"][
                    "pearson_correlation"
                ],
                "prediction_disagreement_fraction": summary["kernel_vs_exact"][
                    "prediction_disagreement_fraction"
                ],
                "job_id": attempt["job_id"],
                "qpu_seconds": usage,
            }
        )
        for index, (source_id, truth, prediction, score) in enumerate(
            zip(source_ids, y_test, predictions, scores)
        ):
            prediction_rows.append(
                {
                    "configuration": config.name,
                    "dataset": config.dataset,
                    "representation": config.representation,
                    "qubits": config.qubits,
                    "fold": fold,
                    "repeat": int(row["repeat"]),
                    "mode": "hardware",
                    "test_position": index,
                    "source_id": source_id,
                    "truth": int(truth),
                    "prediction": int(prediction),
                    "decision_score": float(score),
                    "job_id": attempt["job_id"],
                }
            )
    return metric_rows, prediction_rows


def _validate_analysis_rows(
    metric_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> None:
    metric_frame = pd.DataFrame(metric_rows)
    prediction_frame = pd.DataFrame(prediction_rows)
    expected_metric_modes = {
        "exact": len(CONFIGURATIONS) * N_FOLDS,
        "ideal_aer": len(CONFIGURATIONS) * N_FOLDS,
        "noisy_aer": len(CONFIGURATIONS) * N_FOLDS,
        "hardware": len(planned_hardware_schedule()),
    }
    observed_metric_modes = metric_frame["mode"].value_counts().to_dict()
    if observed_metric_modes != expected_metric_modes:
        raise RuntimeError(
            f"Incomplete run metrics: expected {expected_metric_modes}, "
            f"observed {observed_metric_modes}"
        )
    expected_predictions = {
        mode: count * TEST_ROWS for mode, count in expected_metric_modes.items()
    }
    observed_predictions = prediction_frame["mode"].value_counts().to_dict()
    if observed_predictions != expected_predictions:
        raise RuntimeError(
            f"Incomplete predictions: expected {expected_predictions}, "
            f"observed {observed_predictions}"
        )
    primary = metric_frame[metric_frame["repeat"] == 0]
    expected_primary = len(CONFIGURATIONS) * N_FOLDS * len(EXECUTION_MODES)
    if len(primary) != expected_primary:
        raise RuntimeError(
            f"Expected {expected_primary} primary run rows; observed {len(primary)}"
        )
    keys = primary[["configuration", "fold", "mode"]].astype(str).agg(":".join, axis=1)
    if keys.duplicated().any():
        raise RuntimeError("Duplicate primary configuration/fold/mode rows detected")


def _classical_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in CONFIGURATIONS:
        for fold in range(N_FOLDS):
            path = simulation_condition_dir(config.name, fold) / "classical.json"
            if not path.is_file():
                raise RuntimeError(f"Missing classical baseline: {path}")
            result = read_json(path)
            for kernel in ("linear", "rbf"):
                if kernel not in result:
                    raise RuntimeError(f"Missing {kernel} baseline in {path}")
                rows.append(
                    {
                        "configuration": config.name,
                        "dataset": config.dataset,
                        "representation": config.representation,
                        "qubits": config.qubits,
                        "fold": fold,
                        "kernel": kernel,
                        **result[kernel],
                    }
                )
    expected_rows = len(CONFIGURATIONS) * N_FOLDS * 2
    if len(rows) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} classical baseline rows")
    return rows


def _bootstrap_metric_intervals(
    prediction_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(prediction_rows)
    frame = frame[frame["repeat"] == 0]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(stable_seed("publication-bootstrap"))
    for (configuration, mode), group in frame.groupby(["configuration", "mode"]):
        fold_arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _, fold_group in group.groupby("fold", sort=True):
            fold_arrays.append(
                (
                    fold_group["truth"].to_numpy(dtype=int),
                    fold_group["prediction"].to_numpy(dtype=int),
                    fold_group["decision_score"].to_numpy(dtype=float),
                )
            )
        if len(fold_arrays) != N_FOLDS:
            raise RuntimeError(
                f"Expected {N_FOLDS} folds for {configuration} {mode} bootstrap"
            )
        values = np.empty((BOOTSTRAP_REPLICATES, 3), dtype=float)
        for replicate in range(BOOTSTRAP_REPLICATES):
            fold_metrics = []
            for truth, prediction, scores in fold_arrays:
                indices_by_label = [
                    np.flatnonzero(truth == label) for label in (0, 1)
                ]
                selected = np.concatenate(
                    [
                        rng.choice(indices, size=len(indices), replace=True)
                        for indices in indices_by_label
                    ]
                )
                fold_metrics.append(
                    (
                        accuracy_score(truth[selected], prediction[selected]),
                        balanced_accuracy_score(truth[selected], prediction[selected]),
                        roc_auc_score(truth[selected], scores[selected]),
                    )
                )
            values[replicate] = np.mean(fold_metrics, axis=0)
        observed = np.mean(
            [
                (
                    accuracy_score(truth, prediction),
                    balanced_accuracy_score(truth, prediction),
                    roc_auc_score(truth, scores),
                )
                for truth, prediction, scores in fold_arrays
            ],
            axis=0,
        )
        for metric_index, metric in enumerate(
            ("accuracy", "balanced_accuracy", "roc_auc")
        ):
            low, high = np.quantile(values[:, metric_index], [0.025, 0.975])
            rows.append(
                {
                    "configuration": configuration,
                    "mode": mode,
                    "metric": metric,
                    "estimate": float(observed[metric_index]),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "test_rows": len(group),
                    "bootstrap_design": (
                        "class-stratified row resampling within fold; "
                        f"metric averaged over {N_FOLDS} held-out splits"
                    ),
                }
            )
    return rows


def _holm_adjust(p_values: Sequence[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _exact_mcnemar(left: np.ndarray, right: np.ndarray, truth: np.ndarray) -> tuple[int, int, float]:
    left_correct = left == truth
    right_correct = right == truth
    left_only = int(np.sum(left_correct & ~right_correct))
    right_only = int(np.sum(~left_correct & right_correct))
    discordant = left_only + right_only
    if discordant == 0:
        return left_only, right_only, 1.0
    tail = sum(
        math.comb(discordant, value)
        for value in range(min(left_only, right_only) + 1)
    )
    return left_only, right_only, min(1.0, 2 * tail / (2**discordant))


def _planned_contrasts(prediction_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(prediction_rows)
    frame = frame[frame["repeat"] == 0]
    comparisons: list[tuple[str, str, str]] = []
    for config in CONFIGURATIONS:
        comparisons.append(
            (f"{config.name}:hardware_vs_exact", f"{config.name}:exact", f"{config.name}:hardware")
        )
    comparisons.extend(
        [
            ("clamp:hardware_native_vs_pca", "clamp_native10:hardware", "clamp_pca4:hardware"),
            ("fabiana:hardware_native_vs_pca", "fabiana_native8:hardware", "fabiana_pca4:hardware"),
        ]
    )
    output: list[dict[str, Any]] = []
    raw_p: list[float] = []
    for name, left_key, right_key in comparisons:
        left_config, left_mode = left_key.split(":")
        right_config, right_mode = right_key.split(":")
        left = frame[
            (frame["configuration"] == left_config) & (frame["mode"] == left_mode)
        ][["source_id", "truth", "prediction"]].rename(
            columns={"prediction": "left_prediction"}
        )
        right = frame[
            (frame["configuration"] == right_config) & (frame["mode"] == right_mode)
        ][["source_id", "truth", "prediction"]].rename(
            columns={"prediction": "right_prediction"}
        )
        paired = left.merge(right, on=["source_id", "truth"], validate="one_to_one")
        left_only, right_only, p_value = _exact_mcnemar(
            paired["left_prediction"].to_numpy(),
            paired["right_prediction"].to_numpy(),
            paired["truth"].to_numpy(),
        )
        output.append(
            {
                "contrast": name,
                "paired_test_rows": len(paired),
                "left_only_correct": left_only,
                "right_only_correct": right_only,
                "mcnemar_exact_p": p_value,
            }
        )
        raw_p.append(p_value)
    adjusted = _holm_adjust(raw_p)
    for row, value in zip(output, adjusted):
        row["holm_adjusted_p"] = value
    return output


def _compression_effects(
    metric_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate the preregistered native/PCA difference-in-differences."""

    frame = pd.DataFrame(metric_rows)
    frame = frame[frame["repeat"] == 0]
    pairs = {
        "clamp": ("clamp_native10", "clamp_pca4"),
        "fabiana": ("fabiana_native8", "fabiana_pca4"),
    }
    fold_rows: list[dict[str, Any]] = []
    for dataset, (native, pca) in pairs.items():
        for fold in range(N_FOLDS):
            values: dict[tuple[str, str], float] = {}
            for configuration in (native, pca):
                for mode in ("exact", "hardware"):
                    selected = frame[
                        (frame["configuration"] == configuration)
                        & (frame["fold"] == fold)
                        & (frame["mode"] == mode)
                    ]
                    if len(selected) != 1:
                        raise RuntimeError(
                            f"Missing {configuration} fold {fold} {mode} metric"
                        )
                    values[(configuration, mode)] = float(
                        selected.iloc[0]["balanced_accuracy"]
                    )
            native_degradation = values[(native, "hardware")] - values[(native, "exact")]
            pca_degradation = values[(pca, "hardware")] - values[(pca, "exact")]
            fold_rows.append(
                {
                    "dataset": dataset,
                    "fold": fold,
                    "native_configuration": native,
                    "pca_configuration": pca,
                    "exact_pca_minus_native": values[(pca, "exact")]
                    - values[(native, "exact")],
                    "hardware_pca_minus_native": values[(pca, "hardware")]
                    - values[(native, "hardware")],
                    "native_hardware_minus_exact": native_degradation,
                    "pca_hardware_minus_exact": pca_degradation,
                    "difference_in_differences": pca_degradation
                    - native_degradation,
                }
            )
    summary_rows: list[dict[str, Any]] = []
    effects = pd.DataFrame(fold_rows)
    degrees_of_freedom = N_FOLDS - 1
    critical_t = float(student_t.ppf(0.975, df=degrees_of_freedom))
    columns = (
        "exact_pca_minus_native",
        "hardware_pca_minus_native",
        "native_hardware_minus_exact",
        "pca_hardware_minus_exact",
        "difference_in_differences",
    )
    for dataset, group in effects.groupby("dataset"):
        for metric in columns:
            values = group[metric].to_numpy(dtype=float)
            mean = float(np.mean(values))
            standard_deviation = float(np.std(values, ddof=1))
            half_width = critical_t * standard_deviation / math.sqrt(len(values))
            summary_rows.append(
                {
                    "dataset": dataset,
                    "effect": metric,
                    "folds": len(values),
                    "mean": mean,
                    "standard_deviation": standard_deviation,
                    "degrees_of_freedom": degrees_of_freedom,
                    "ci_low_t": mean - half_width,
                    "ci_high_t": mean + half_width,
                    "interpretation": (
                        f"Descriptive {N_FOLDS}-split paired t interval"
                    ),
                }
            )
    return fold_rows, summary_rows


def _resource_tables(metric_rows: Sequence[Mapping[str, Any]]) -> None:
    circuit_rows = []
    qpu_frame = pd.DataFrame(metric_rows)
    for width in (4, 8, 10):
        metadata = read_json(circuit_metadata_path(width))
        hardware = qpu_frame[
            (qpu_frame["qubits"] == width)
            & (qpu_frame["mode"] == "hardware")
        ]
        numeric_qpu = pd.to_numeric(hardware.get("qpu_seconds"), errors="coerce")
        circuit_rows.append(
            {
                **metadata,
                "cz_per_fidelity_circuit": metadata["operations"].get("cz", 0),
                "median_qpu_seconds_per_condition": (
                    float(numeric_qpu.median()) if numeric_qpu.notna().any() else ""
                ),
            }
        )
    atomic_write_csv(ANALYSIS_DIR / "circuit_resources.csv", circuit_rows)

    scenarios = tuple(
        dict.fromkeys(
            (
                (TRAIN_ROWS, TEST_ROWS),
                (100, 100),
                (500, 500),
                (1_000, 1_000),
                (5_000, 1_000),
            )
        )
    )
    scaling_rows = []
    for train_rows, test_rows in scenarios:
        evaluations = evaluation_count(train_rows, test_rows)
        for circuit in circuit_rows:
            cz = int(circuit["cz_per_fidelity_circuit"])
            gates = int(circuit["size"])
            median_condition = circuit["median_qpu_seconds_per_condition"]
            projected = ""
            if median_condition != "":
                projected = float(median_condition) * evaluations / evaluation_count(
                    TRAIN_ROWS, TEST_ROWS
                )
            scaling_rows.append(
                {
                    "train_rows": train_rows,
                    "test_rows": test_rows,
                    "logical_qubits": circuit["logical_qubits"],
                    "kernel_evaluations": evaluations,
                    "shots_per_evaluation": SHOTS,
                    "total_circuit_shots": evaluations * SHOTS,
                    "gates_per_circuit": gates,
                    "total_gate_applications": evaluations * SHOTS * gates,
                    "cz_per_circuit": cz,
                    "total_cz_applications": evaluations * SHOTS * cz,
                    "projected_qpu_seconds": projected,
                    "projection_note": "Linear extrapolation from observed median; not a measured run",
                }
            )
    atomic_write_csv(ANALYSIS_DIR / "resource_scaling_estimates.csv", scaling_rows)


def _aggregate_metric_rows(
    metric_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    frame = frame[frame["repeat"] == 0]
    rows: list[dict[str, Any]] = []
    for (configuration, mode), group in frame.groupby(
        ["configuration", "mode"], sort=False
    ):
        row: dict[str, Any] = {
            "configuration": configuration,
            "mode": mode,
            "folds": len(group),
        }
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "test_relative_frobenius_error",
            "test_kernel_correlation",
        ):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_standard_deviation"] = float(values.std(ddof=1))
        rows.append(row)
    return rows


def _aggregate_classical_rows(
    classical_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(classical_rows)
    rows: list[dict[str, Any]] = []
    for (configuration, kernel), group in frame.groupby(
        ["configuration", "kernel"], sort=False
    ):
        row: dict[str, Any] = {
            "configuration": configuration,
            "kernel": kernel,
            "folds": len(group),
        }
        for metric in ("accuracy", "balanced_accuracy", "roc_auc"):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_standard_deviation"] = float(values.std(ddof=1))
        rows.append(row)
    return rows


def _aggregate_anchor_repeats(
    metric_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(metric_rows)
    hardware = frame[frame["mode"] == "hardware"]
    rows: list[dict[str, Any]] = []
    for configuration, fold in ANCHORS:
        group = hardware[
            (hardware["configuration"] == configuration)
            & (hardware["fold"] == fold)
        ]
        expected_repeats = set(range(EXTRA_RUNS_PER_ANCHOR + 1))
        if (
            len(group) != len(expected_repeats)
            or set(group["repeat"].astype(int)) != expected_repeats
        ):
            raise RuntimeError(
                f"Expected {len(expected_repeats)} hardware measurements for "
                f"{configuration} fold {fold}"
            )
        row: dict[str, Any] = {
            "configuration": configuration,
            "fold": fold,
            "qubits": int(group.iloc[0]["qubits"]),
            "hardware_measurements": len(group),
        }
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "roc_auc",
            "test_relative_frobenius_error",
            "test_kernel_correlation",
            "qpu_seconds",
        ):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_standard_deviation"] = float(values.std(ddof=1))
            row[f"{metric}_minimum"] = float(values.min())
            row[f"{metric}_maximum"] = float(values.max())
        rows.append(row)
    return rows


def _hardware_job_rows(
    campaign: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    usage_by_job = {
        str(row.get("job_id")): row.get("qpu_seconds")
        for row in metric_rows
        if row.get("mode") == "hardware" and row.get("job_id")
    }
    rows: list[dict[str, Any]] = []
    for condition in campaign["hardware_schedule"]:
        for attempt in condition.get("attempts", []):
            rows.append(
                {
                    "condition_id": condition["condition_id"],
                    "order": condition["order"],
                    "configuration": condition["configuration"],
                    "dataset": condition["dataset"],
                    "qubits": condition["qubits"],
                    "fold": condition["fold"],
                    "repeat": condition["repeat"],
                    "anchor": condition["anchor"],
                    "attempt": attempt["attempt"],
                    "job_id": attempt["job_id"],
                    "batch_id": attempt.get("batch_id"),
                    "status": attempt.get("status"),
                    "submitted_at_utc": attempt.get("submitted_at_utc"),
                    "retrieved_at_utc": attempt.get("retrieved_at_utc"),
                    "total_executions": attempt.get("total_executions"),
                    "qpu_seconds": attempt.get(
                        "qpu_seconds", usage_by_job.get(str(attempt["job_id"]), "")
                    ),
                    "error_message": attempt.get("error_message", ""),
                }
            )
    return rows


def _format_report_number(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}f}"


def _write_publication_report(
    campaign: Mapping[str, Any],
    metric_rows: Sequence[Mapping[str, Any]],
    aggregate_metrics: Sequence[Mapping[str, Any]],
    aggregate_classical: Sequence[Mapping[str, Any]],
    anchor_summary: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    effect_summary: Sequence[Mapping[str, Any]],
    job_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "SRP2026 PUBLICATION QUANTUM-KERNEL HARDWARE RESULTS",
        "=" * 57,
        f"Generated (UTC): {utc_now()}",
        "",
        "SCOPE",
        "-----",
        "Controlled small-sample NISQ hardware study of representation compression,",
        "kernel geometry, and execution noise in malware classification. These results",
        "do not demonstrate quantum advantage or full-corpus detector performance.",
        "",
        "FROZEN PROTOCOL",
        "---------------",
        f"Configurations: {len(CONFIGURATIONS)}",
        f"Folds per configuration: {N_FOLDS}",
        f"Rows per split: {TRAIN_ROWS} train / {TEST_ROWS} test "
        f"({ROWS_PER_CLASS}/{ROWS_PER_CLASS} by class)",
        f"Feature map: zz_feature_map, reps={REPS}, entanglement={ENTANGLEMENT}",
        f"Classifier: precomputed SVC(C={SVC_C:g})",
        f"Shot-based modes: {SHOTS} shots; mitigation disabled",
        "Execution ladder: exact -> ideal_aer -> noisy_aer -> hardware",
    ]
    concurrent_batches = [
        batch
        for batch in campaign.get("batches", [])
        if batch.get("concurrent_simulation_authorized")
    ]
    if concurrent_batches:
        lines.extend(
            [
                "",
                "PROCEDURAL NOTE",
                "---------------",
                "At the user's request, hardware submission began after every prepared split",
                "and exact kernel passed validation but before the ideal/noisy Aer ladder",
                "finished locally. Inputs, circuits, shots, seeds, and the hardware schedule",
                "were already frozen; no condition was tuned after observing hardware results.",
            ]
        )
    lines.extend(
        [
            "",
            "HARDWARE WORKLOAD",
            "-----------------",
        ]
    )
    workload = campaign["workload"]
    for key in (
        "evaluations_per_condition",
        "executions_per_condition",
        "primary_hardware_conditions",
        "repeat_anchor_conditions",
        "hardware_condition_runs",
        "total_hardware_evaluations",
        "total_hardware_executions",
    ):
        lines.append(f"{key}: {int(workload[key]):,}")

    metric_frame = pd.DataFrame(metric_rows)
    primary = metric_frame[metric_frame["repeat"] == 0]
    hardware = primary[primary["mode"] == "hardware"].copy()
    exact = primary[primary["mode"] == "exact"][
        ["configuration", "fold", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "exact_balanced_accuracy"})
    cost = hardware.merge(exact, on=["configuration", "fold"], validate="one_to_one")
    cz_by_width = {
        width: read_json(circuit_metadata_path(width))["operations"].get("cz", 0)
        for width in (4, 8, 10)
    }
    cost["cz_count"] = cost["qubits"].map(cz_by_width)
    cost["hardware_minus_exact_balanced_accuracy"] = (
        cost["balanced_accuracy"] - cost["exact_balanced_accuracy"]
    )

    def descriptive_correlation(outcome: str, method: str) -> float:
        if cost["cz_count"].nunique() < 2 or cost[outcome].nunique() < 2:
            return float("nan")
        return float(cost["cz_count"].corr(cost[outcome], method=method))

    distortion_pearson = descriptive_correlation(
        "test_relative_frobenius_error", "pearson"
    )
    distortion_spearman = descriptive_correlation(
        "test_relative_frobenius_error", "spearman"
    )
    performance_pearson = descriptive_correlation(
        "hardware_minus_exact_balanced_accuracy", "pearson"
    )
    performance_spearman = descriptive_correlation(
        "hardware_minus_exact_balanced_accuracy", "spearman"
    )
    lines.extend(
        [
            "",
            "DESCRIPTIVE CIRCUIT-COST ASSOCIATIONS (18 primary runs)",
            "--------------------------------------------------------",
            "These repeated-width correlations are descriptive and confounded by dataset",
            "and representation; they are not an independent circuit-depth experiment.",
            "CZ count vs test relative Frobenius error: "
            f"Pearson r={_format_report_number(distortion_pearson)}, "
            f"Spearman rho={_format_report_number(distortion_spearman)}",
            "CZ count vs hardware-minus-exact balanced accuracy: "
            f"Pearson r={_format_report_number(performance_pearson)}, "
            f"Spearman rho={_format_report_number(performance_spearman)}",
        ]
    )
    lines.extend(
        [
            "",
            "PRIMARY QUANTUM RESULTS (mean +/- fold SD; primary runs only)",
            "------------------------------------------------------------",
            "configuration | mode | balanced accuracy | ROC AUC | test rel. error | test correlation",
        ]
    )
    mode_rank = {mode: index for index, mode in enumerate(EXECUTION_MODES)}
    sorted_metrics = sorted(
        aggregate_metrics,
        key=lambda row: (
            next(
                index
                for index, config in enumerate(CONFIGURATIONS)
                if config.name == row["configuration"]
            ),
            mode_rank[str(row["mode"])],
        ),
    )
    for row in sorted_metrics:
        lines.append(
            " | ".join(
                [
                    str(row["configuration"]),
                    str(row["mode"]),
                    f"{_format_report_number(row['balanced_accuracy_mean'])} +/- "
                    f"{_format_report_number(row['balanced_accuracy_standard_deviation'])}",
                    f"{_format_report_number(row['roc_auc_mean'])} +/- "
                    f"{_format_report_number(row['roc_auc_standard_deviation'])}",
                    _format_report_number(row["test_relative_frobenius_error_mean"]),
                    _format_report_number(row["test_kernel_correlation_mean"]),
                ]
            )
        )

    balanced_intervals = [
        row for row in intervals if row["metric"] == "balanced_accuracy"
    ]
    lines.extend(
        [
            "",
            "FOLD-STRATIFIED BALANCED-ACCURACY BOOTSTRAP INTERVALS",
            "-----------------------------------------------------",
            "configuration | mode | estimate | 95% interval | total test rows",
        ]
    )
    for row in sorted(
        balanced_intervals,
        key=lambda item: (str(item["configuration"]), mode_rank[str(item["mode"])]),
    ):
        lines.append(
            f"{row['configuration']} | {row['mode']} | "
            f"{_format_report_number(row['estimate'])} | "
            f"[{_format_report_number(row['ci_low'])}, "
            f"{_format_report_number(row['ci_high'])}] | {int(row['test_rows'])}"
        )

    lines.extend(
        [
            "",
            "CLASSICAL SANITY CHECKS (mean +/- fold SD)",
            "-----------------------------------------",
            "configuration | kernel | balanced accuracy | ROC AUC",
        ]
    )
    for row in aggregate_classical:
        lines.append(
            f"{row['configuration']} | {row['kernel']} | "
            f"{_format_report_number(row['balanced_accuracy_mean'])} +/- "
            f"{_format_report_number(row['balanced_accuracy_standard_deviation'])} | "
            f"{_format_report_number(row['roc_auc_mean'])} +/- "
            f"{_format_report_number(row['roc_auc_standard_deviation'])}"
        )

    lines.extend(
        [
            "",
            "ANCHOR HARDWARE REPEATABILITY (two measurements of split 0)",
            "------------------------------------------------------------",
            "configuration | qubits | balanced accuracy mean +/- SD | test error mean +/- SD | QPU seconds mean +/- SD",
        ]
    )
    for row in anchor_summary:
        lines.append(
            f"{row['configuration']} | {row['qubits']} | "
            f"{_format_report_number(row['balanced_accuracy_mean'])} +/- "
            f"{_format_report_number(row['balanced_accuracy_standard_deviation'])} | "
            f"{_format_report_number(row['test_relative_frobenius_error_mean'])} +/- "
            f"{_format_report_number(row['test_relative_frobenius_error_standard_deviation'])} | "
            f"{_format_report_number(row['qpu_seconds_mean'], 3)} +/- "
            f"{_format_report_number(row['qpu_seconds_standard_deviation'], 3)}"
        )

    lines.extend(
        [
            "",
            f"NATIVE/PCA PAIRED EFFECTS (mean and descriptive {N_FOLDS}-split t interval)",
            "--------------------------------------------------------------------",
            "dataset | effect | mean | 95% interval",
        ]
    )
    for row in effect_summary:
        lines.append(
            f"{row['dataset']} | {row['effect']} | "
            f"{_format_report_number(row['mean'])} | "
            f"[{_format_report_number(row['ci_low_t'])}, "
            f"{_format_report_number(row['ci_high_t'])}]"
        )

    lines.extend(
        [
            "",
            "PLANNED EXACT MCNEMAR CONTRASTS (Holm-adjusted)",
            "------------------------------------------------",
            "contrast | paired rows | left-only correct | right-only correct | raw p | adjusted p",
        ]
    )
    for row in contrasts:
        lines.append(
            f"{row['contrast']} | {int(row['paired_test_rows'])} | "
            f"{int(row['left_only_correct'])} | {int(row['right_only_correct'])} | "
            f"{_format_report_number(row['mcnemar_exact_p'], 6)} | "
            f"{_format_report_number(row['holm_adjusted_p'], 6)}"
        )

    successful_jobs = [
        row
        for row in job_rows
        if row.get("status") == "DONE" and row.get("retrieved_at_utc")
    ]
    qpu_values = []
    for row in job_rows:
        try:
            value = float(row["qpu_seconds"])
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            qpu_values.append(value)
    failed_attempts = [
        row for row in job_rows if row.get("status") in {"ERROR", "CANCELLED"}
    ]
    lines.extend(
        [
            "",
            "HARDWARE COMPLETION",
            "-------------------",
            "Successful retrieved conditions: "
            f"{len(successful_jobs)} / {len(planned_hardware_schedule())}",
            f"Total submitted attempts: {len(job_rows)}",
            f"Failed/cancelled attempts: {len(failed_attempts)}",
            f"Total reported QPU usage (seconds): {_format_report_number(sum(qpu_values), 3)}",
            "condition | attempt | job ID | QPU seconds | status",
        ]
    )
    for row in job_rows:
        lines.append(
            f"{row['condition_id']} | {row['attempt']} | {row['job_id']} | "
            f"{_format_report_number(row['qpu_seconds'], 3)} | {row['status']}"
        )

    lines.extend(
        [
            "",
            "CIRCUIT RESOURCES",
            "-----------------",
            "logical qubits | depth | CZ per circuit | estimated circuit seconds",
        ]
    )
    for width in (4, 8, 10):
        metadata = read_json(circuit_metadata_path(width))
        lines.append(
            f"{width} | {metadata['depth']} | "
            f"{metadata['operations'].get('cz', 0)} | "
            f"{_format_report_number(metadata.get('estimated_duration_seconds'), 9)}"
        )

    lines.extend(
        [
            "",
            "OUTPUT INDEX",
            "------------",
            "run_metrics.csv: one row per quantum condition/mode/repeat",
            "predictions.csv: one row per held-out prediction",
            "bootstrap_intervals.csv: 10,000-replicate within-fold/class intervals",
            "planned_mcnemar_contrasts.csv: paired tests and Holm adjustment",
            "compression_effects_by_fold.csv and compression_effects_summary.csv",
            "classical_baselines.csv and aggregate_classical_baselines.csv",
            "anchor_repeat_summary.csv: 4q/8q/10q hardware repeatability",
            "circuit_resources.csv and resource_scaling_estimates.csv",
            "hardware_jobs.csv: every submitted attempt and traceability metadata",
            "tables/*.tex: captioned LaTeX publication tables",
            "figure_captions.tex: LaTeX figure environments, captions, and labels",
            "figure1_noise_ladder_balanced_accuracy.[png|pdf|svg]",
            "figure2_kernel_distortion.[png|pdf|svg]",
            "figure3_circuit_cost_vs_degradation.[png|pdf|svg]",
        ]
    )
    atomic_write_text(ANALYSIS_DIR / "publication_results.txt", "\n".join(lines))


def _mean_sd(mean: Any, standard_deviation: Any, digits: int = 3) -> str:
    return (
        f"{_format_report_number(mean, digits)} "
        f"({_format_report_number(standard_deviation, digits)})"
    )


def _interval(estimate: Any, low: Any, high: Any, digits: int = 3) -> str:
    return (
        f"{_format_report_number(estimate, digits)} "
        f"[{_format_report_number(low, digits)}, "
        f"{_format_report_number(high, digits)}]"
    )


def _write_latex_table(
    filename: str,
    frame: pd.DataFrame,
    caption: str,
    label: str,
    *,
    longtable: bool = False,
    column_format: str | None = None,
) -> None:
    """Write a paper-ready LaTeX table while retaining CSV audit files."""

    LATEX_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    options: dict[str, Any] = {
        "index": False,
        "escape": True,
        "na_rep": "--",
        "caption": caption,
        "label": label,
        "longtable": longtable,
        "column_format": column_format,
    }
    if not longtable:
        options["position"] = "tbp"
    latex = frame.to_latex(**options)
    requirements = (
        "% Generated by publication_hardware_experiment.py.\n"
        "% Requires \\usepackage{booktabs}; long tables also require "
        "\\usepackage{longtable}.\n"
    )
    atomic_write_text(LATEX_TABLE_DIR / filename, requirements + latex)


def _publication_latex_tables(
    aggregate_metrics: Sequence[Mapping[str, Any]],
    aggregate_classical: Sequence[Mapping[str, Any]],
    anchor_summary: Sequence[Mapping[str, Any]],
    intervals: Sequence[Mapping[str, Any]],
    contrasts: Sequence[Mapping[str, Any]],
    effect_summary: Sequence[Mapping[str, Any]],
    job_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Create the paper tables as captioned LaTeX fragments."""

    mode_rank = {mode: index for index, mode in enumerate(EXECUTION_MODES)}
    config_rank = {
        config.name: index for index, config in enumerate(CONFIGURATIONS)
    }

    primary_rows = []
    for row in sorted(
        aggregate_metrics,
        key=lambda item: (
            config_rank[str(item["configuration"])],
            mode_rank[str(item["mode"])],
        ),
    ):
        primary_rows.append(
            {
                "Configuration": CONFIGURATION_LABELS[str(row["configuration"])],
                "Environment": MODE_LABELS[str(row["mode"])],
                "Balanced accuracy, mean (SD)": _mean_sd(
                    row["balanced_accuracy_mean"],
                    row["balanced_accuracy_standard_deviation"],
                ),
                "ROC AUC, mean (SD)": _mean_sd(
                    row["roc_auc_mean"], row["roc_auc_standard_deviation"]
                ),
                "Relative error": _format_report_number(
                    row["test_relative_frobenius_error_mean"], 3
                ),
                "Correlation": _format_report_number(
                    row["test_kernel_correlation_mean"], 3
                ),
            }
        )
    _write_latex_table(
        "table1_primary_quantum_results.tex",
        pd.DataFrame(primary_rows),
        "Quantum-kernel results across execution environments; values are means "
        "with split standard deviations in parentheses.",
        "tab:primary-quantum-results",
        longtable=True,
        column_format="llrrrr",
    )

    interval_rows = []
    for row in sorted(
        (item for item in intervals if item["metric"] == "balanced_accuracy"),
        key=lambda item: (
            config_rank[str(item["configuration"])],
            mode_rank[str(item["mode"])],
        ),
    ):
        interval_rows.append(
            {
                "Configuration": CONFIGURATION_LABELS[str(row["configuration"])],
                "Environment": MODE_LABELS[str(row["mode"])],
                "Balanced accuracy [95% CI]": _interval(
                    row["estimate"], row["ci_low"], row["ci_high"]
                ),
                "Test samples": int(row["test_rows"]),
            }
        )
    _write_latex_table(
        "table2_balanced_accuracy_intervals.tex",
        pd.DataFrame(interval_rows),
        "Balanced accuracy with 95\\% class-stratified bootstrap intervals.",
        "tab:balanced-accuracy-intervals",
        longtable=True,
        column_format="llrr",
    )

    classical_rows = []
    for row in aggregate_classical:
        classical_rows.append(
            {
                "Configuration": CONFIGURATION_LABELS[str(row["configuration"])],
                "Kernel": str(row["kernel"]).upper(),
                "Balanced accuracy, mean (SD)": _mean_sd(
                    row["balanced_accuracy_mean"],
                    row["balanced_accuracy_standard_deviation"],
                ),
                "ROC AUC, mean (SD)": _mean_sd(
                    row["roc_auc_mean"], row["roc_auc_standard_deviation"]
                ),
            }
        )
    _write_latex_table(
        "table3_classical_baselines.tex",
        pd.DataFrame(classical_rows),
        "Classical SVC sanity checks on the matched transformed samples.",
        "tab:classical-baselines",
        column_format="llrr",
    )

    effect_labels = {
        "exact_pca_minus_native": "Exact: PCA minus native",
        "hardware_pca_minus_native": "Hardware: PCA minus native",
        "native_hardware_minus_exact": "Native: hardware minus exact",
        "pca_hardware_minus_exact": "PCA: hardware minus exact",
        "difference_in_differences": "Compression difference-in-differences",
    }
    effect_rows = [
        {
            "Dataset": str(row["dataset"]).upper(),
            "Effect": effect_labels[str(row["effect"])],
            "Mean [95% CI]": _interval(
                row["mean"], row["ci_low_t"], row["ci_high_t"]
            ),
        }
        for row in effect_summary
    ]
    _write_latex_table(
        "table4_compression_effects.tex",
        pd.DataFrame(effect_rows),
        "Paired native/PCA effects with descriptive three-split t intervals.",
        "tab:compression-effects",
        column_format="llr",
    )

    contrast_rows = [
        {
            "Contrast": str(row["contrast"]).replace(":", ": ").replace("_", " "),
            "Pairs": int(row["paired_test_rows"]),
            "Left only": int(row["left_only_correct"]),
            "Right only": int(row["right_only_correct"]),
            "Raw p": _format_report_number(row["mcnemar_exact_p"], 4),
            "Holm p": _format_report_number(row["holm_adjusted_p"], 4),
        }
        for row in contrasts
    ]
    _write_latex_table(
        "table5_paired_contrasts.tex",
        pd.DataFrame(contrast_rows),
        "Prespecified exact McNemar tests with Holm adjustment.",
        "tab:paired-contrasts",
        column_format="lrrrrr",
    )

    anchor_rows = [
        {
            "Configuration": CONFIGURATION_LABELS[str(row["configuration"])],
            "Qubits": int(row["qubits"]),
            "Balanced accuracy, mean (SD)": _mean_sd(
                row["balanced_accuracy_mean"],
                row["balanced_accuracy_standard_deviation"],
            ),
            "Relative error, mean (SD)": _mean_sd(
                row["test_relative_frobenius_error_mean"],
                row["test_relative_frobenius_error_standard_deviation"],
            ),
            "QPU seconds, mean (SD)": _mean_sd(
                row["qpu_seconds_mean"], row["qpu_seconds_standard_deviation"], 1
            ),
        }
        for row in anchor_summary
    ]
    _write_latex_table(
        "table6_anchor_repeatability.tex",
        pd.DataFrame(anchor_rows),
        "Hardware repeatability for the three prespecified split-zero anchors.",
        "tab:anchor-repeatability",
        column_format="lrrrr",
    )

    circuit_frame = pd.read_csv(ANALYSIS_DIR / "circuit_resources.csv")
    circuit_rows = []
    for _, row in circuit_frame.sort_values("logical_qubits").iterrows():
        circuit_rows.append(
            {
                "Qubits": int(row["logical_qubits"]),
                "Depth": int(row["depth"]),
                "Gates": int(row["size"]),
                "CZ gates": int(row["cz_per_fidelity_circuit"]),
                "Circuit duration (microseconds)": _format_report_number(
                    float(row["estimated_duration_seconds"]) * 1e6, 3
                ),
                "Median QPU seconds per condition": _format_report_number(
                    row["median_qpu_seconds_per_condition"], 1
                ),
            }
        )
    _write_latex_table(
        "table7_circuit_resources.tex",
        pd.DataFrame(circuit_rows),
        "Measured transpiled circuit resources on IBM Rensselaer.",
        "tab:circuit-resources",
        column_format="rrrrrr",
    )

    scaling_frame = pd.read_csv(ANALYSIS_DIR / "resource_scaling_estimates.csv")
    scaling_rows = []
    for _, row in scaling_frame.iterrows():
        scaling_rows.append(
            {
                "Train/test": f"{int(row['train_rows']):,}/{int(row['test_rows']):,}",
                "Qubits": int(row["logical_qubits"]),
                "Kernel evaluations": f"{int(row['kernel_evaluations']):,}",
                "Circuit shots": f"{int(row['total_circuit_shots']):,}",
                "Gate applications": f"{int(row['total_gate_applications']):,}",
                "CZ applications": f"{int(row['total_cz_applications']):,}",
                "Projected QPU hours": _format_report_number(
                    float(row["projected_qpu_seconds"]) / 3600, 1
                ),
            }
        )
    _write_latex_table(
        "table8_resource_scaling_estimates.tex",
        pd.DataFrame(scaling_rows),
        "Kernel-workload scaling estimates from observed median QPU time; larger "
        "scenarios are extrapolations, not measured runs.",
        "tab:resource-scaling",
        longtable=True,
        column_format="rrrrrrr",
    )

    hardware_rows = [
        {
            "Condition": str(row["condition_id"]).replace("_", " "),
            "Job ID": str(row["job_id"]),
            "Status": str(row["status"]),
            "QPU seconds": _format_report_number(row["qpu_seconds"], 1),
        }
        for row in job_rows
    ]
    _write_latex_table(
        "tableS1_hardware_jobs.tex",
        pd.DataFrame(hardware_rows),
        "IBM Runtime traceability for every scheduled hardware run.",
        "tab:hardware-jobs",
        longtable=True,
        column_format="llrr",
    )

    table_files = (
        "table1_primary_quantum_results.tex",
        "table2_balanced_accuracy_intervals.tex",
        "table3_classical_baselines.tex",
        "table4_compression_effects.tex",
        "table5_paired_contrasts.tex",
        "table6_anchor_repeatability.tex",
        "table7_circuit_resources.tex",
        "table8_resource_scaling_estimates.tex",
        "tableS1_hardware_jobs.tex",
    )
    atomic_write_text(
        LATEX_TABLE_DIR / "all_tables.tex",
        "% Include from the analysis directory.\n"
        + "\n".join(f"\\input{{tables/{name}}}" for name in table_files),
    )


FIGURE_CAPTIONS = {
    "figure1_noise_ladder_balanced_accuracy": (
        "Balanced accuracy across exact, ideal-Aer, noisy-Aer, and IBM hardware "
        "execution. Thin lines pair held-out splits; thick lines show split means "
        "with 95\\% t intervals."
    ),
    "figure2_kernel_distortion": (
        "Test-kernel distortion relative to the exact kernel across execution "
        "environments. Lines show split-averaged relative Frobenius error and "
        "Pearson correlation."
    ),
    "figure3_circuit_cost_vs_degradation": (
        "Circuit cost and hardware degradation. Each point is one held-out split; "
        "CZ count is the transpiled two-qubit-gate count per fidelity circuit."
    ),
}


def _write_figure_captions() -> None:
    blocks = []
    labels = {
        "figure1_noise_ladder_balanced_accuracy": "fig:noise-ladder",
        "figure2_kernel_distortion": "fig:kernel-distortion",
        "figure3_circuit_cost_vs_degradation": "fig:circuit-cost",
    }
    widths = {
        "figure1_noise_ladder_balanced_accuracy": "\\textwidth",
        "figure2_kernel_distortion": "\\textwidth",
        "figure3_circuit_cost_vs_degradation": "0.92\\textwidth",
    }
    for stem, caption in FIGURE_CAPTIONS.items():
        blocks.append(
            "\n".join(
                [
                    "\\begin{figure*}[tbp]",
                    "  \\centering",
                    f"  \\includegraphics[width={widths[stem]}]{{{stem}.pdf}}",
                    f"  \\caption{{{caption}}}",
                    f"  \\label{{{labels[stem]}}}",
                    "\\end{figure*}",
                ]
            )
        )
    atomic_write_text(
        ANALYSIS_DIR / "figure_captions.tex",
        "% Requires \\usepackage{graphicx}.\n\n" + "\n\n".join(blocks),
    )


def _publication_figures(metric_rows: Sequence[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame(metric_rows)
    primary = frame[frame["repeat"] == 0].copy()
    mode_order = list(EXECUTION_MODES)
    primary["mode"] = pd.Categorical(primary["mode"], mode_order, ordered=True)
    mode_positions = np.arange(len(mode_order))
    mode_tick_labels = [MODE_LABELS[mode] for mode in mode_order]
    pair_palette = ("#0072B2", "#D55E00")
    configuration_palette = {
        config.name: plt.get_cmap("tab10")(index)
        for index, config in enumerate(CONFIGURATIONS)
    }
    configuration_markers = {
        config.name: marker
        for config, marker in zip(CONFIGURATIONS, ("o", "s", "^", "D", "P", "X"))
    }
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, dataset in zip(axes.flat, ("clamp", "fabiana", "cic", "ember2018")):
        subset = primary[primary["dataset"] == dataset]
        configurations = [
            config.name for config in CONFIGURATIONS if config.dataset == dataset
        ]
        for color, configuration in zip(pair_palette, configurations):
            group = subset[subset["configuration"] == configuration]
            for _, fold_group in group.groupby("fold"):
                paired = fold_group.set_index("mode").reindex(mode_order)
                axis.plot(
                    mode_positions,
                    paired["balanced_accuracy"],
                    color=color,
                    alpha=0.18,
                    linewidth=0.8,
                )
            summary = group.groupby("mode", observed=False)["balanced_accuracy"].agg(
                ["mean", "sem"]
            ).reindex(mode_order)
            critical_t = float(student_t.ppf(0.975, df=N_FOLDS - 1))
            half_width = critical_t * summary["sem"].to_numpy()
            means = summary["mean"].to_numpy()
            lower = np.clip(means - half_width, 0, 1)
            upper = np.clip(means + half_width, 0, 1)
            axis.errorbar(
                mode_positions,
                summary["mean"],
                yerr=np.vstack([means - lower, upper - means]),
                marker="o",
                capsize=3,
                color=color,
                linewidth=1.8,
                label=CONFIGURATION_SHORT_LABELS[configuration],
            )
        axis.set_title("EMBER2018" if dataset == "ember2018" else dataset.upper())
        axis.set_ylim(0, 1.02)
        axis.set_xticks(mode_positions, mode_tick_labels, rotation=18, ha="right")
        axis.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False)
    fig.supylabel("Balanced accuracy")
    fig.supxlabel("Execution environment")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(ANALYSIS_DIR / f"figure1_noise_ladder_balanced_accuracy.{suffix}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for config in CONFIGURATIONS:
        configuration = config.name
        group = primary[primary["configuration"] == configuration]
        summary = group.groupby("mode", observed=False).agg(
            relative_error=("test_relative_frobenius_error", "mean"),
            correlation=("test_kernel_correlation", "mean"),
        ).reindex(mode_order)
        plot_options = {
            "marker": configuration_markers[configuration],
            "color": configuration_palette[configuration],
            "linewidth": 1.5,
            "label": CONFIGURATION_LABELS[configuration],
        }
        axes[0].plot(mode_positions, summary["relative_error"], **plot_options)
        axes[1].plot(mode_positions, summary["correlation"], **plot_options)
    axes[0].set_ylabel("Test-kernel relative Frobenius error")
    axes[1].set_ylabel("Test-kernel Pearson correlation vs exact")
    for axis in axes:
        axis.set_xticks(mode_positions, mode_tick_labels, rotation=18, ha="right")
        axis.set_xlabel("Execution environment")
        axis.grid(axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(ANALYSIS_DIR / f"figure2_kernel_distortion.{suffix}", dpi=300)
    plt.close(fig)

    hardware = primary[primary["mode"] == "hardware"]
    exact = primary[primary["mode"] == "exact"][
        ["configuration", "fold", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "exact_balanced_accuracy"})
    cost = hardware.merge(exact, on=["configuration", "fold"], validate="one_to_one")
    resource = {
        width: read_json(circuit_metadata_path(width))["operations"].get("cz", 0)
        for width in (4, 8, 10)
    }
    cost["cz_count"] = cost["qubits"].map(resource)
    cost["hardware_minus_exact_balanced_accuracy"] = (
        cost["balanced_accuracy"] - cost["exact_balanced_accuracy"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for config in CONFIGURATIONS:
        configuration = config.name
        group = cost[cost["configuration"] == configuration]
        scatter_options = {
            "s": 42,
            "marker": configuration_markers[configuration],
            "color": configuration_palette[configuration],
            "edgecolors": "white",
            "linewidths": 0.5,
        }
        axes[0].scatter(
            group["cz_count"],
            group["hardware_minus_exact_balanced_accuracy"],
            label=CONFIGURATION_LABELS[configuration],
            **scatter_options,
        )
        axes[1].scatter(
            group["cz_count"],
            group["test_relative_frobenius_error"],
            **scatter_options,
        )
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Hardware − exact balanced accuracy")
    axes[1].set_ylabel("Hardware test-kernel relative error")
    for axis in axes:
        axis.set_xlabel("CZ gates per fidelity circuit")
        axis.grid(color="#d0d0d0", linewidth=0.6, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=6.7)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(ANALYSIS_DIR / f"figure3_circuit_cost_vs_degradation.{suffix}", dpi=300)
    plt.close(fig)
    _write_figure_captions()


def analyze_results() -> None:
    campaign = load_or_create_campaign()
    if not _all_hardware_complete(campaign):
        raise RuntimeError(
            "Cannot run final analysis before all "
            f"{len(planned_hardware_schedule())} hardware conditions"
        )
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows = _mode_rows()
    _validate_analysis_rows(metric_rows, prediction_rows)
    classical_rows = _classical_rows()
    aggregate_metrics = _aggregate_metric_rows(metric_rows)
    aggregate_classical = _aggregate_classical_rows(classical_rows)
    anchor_summary = _aggregate_anchor_repeats(metric_rows)
    job_rows = _hardware_job_rows(campaign, metric_rows)
    atomic_write_csv(ANALYSIS_DIR / "run_metrics.csv", metric_rows)
    atomic_write_csv(ANALYSIS_DIR / "predictions.csv", prediction_rows)
    atomic_write_csv(ANALYSIS_DIR / "aggregate_run_metrics.csv", aggregate_metrics)
    atomic_write_csv(ANALYSIS_DIR / "classical_baselines.csv", classical_rows)
    atomic_write_csv(
        ANALYSIS_DIR / "aggregate_classical_baselines.csv", aggregate_classical
    )
    atomic_write_csv(ANALYSIS_DIR / "anchor_repeat_summary.csv", anchor_summary)
    atomic_write_csv(ANALYSIS_DIR / "hardware_jobs.csv", job_rows)
    intervals = _bootstrap_metric_intervals(prediction_rows)
    atomic_write_csv(ANALYSIS_DIR / "bootstrap_intervals.csv", intervals)
    contrasts = _planned_contrasts(prediction_rows)
    atomic_write_csv(ANALYSIS_DIR / "planned_mcnemar_contrasts.csv", contrasts)
    effect_rows, effect_summary = _compression_effects(metric_rows)
    atomic_write_csv(ANALYSIS_DIR / "compression_effects_by_fold.csv", effect_rows)
    atomic_write_csv(ANALYSIS_DIR / "compression_effects_summary.csv", effect_summary)
    _resource_tables(metric_rows)
    _publication_latex_tables(
        aggregate_metrics,
        aggregate_classical,
        anchor_summary,
        intervals,
        contrasts,
        effect_summary,
        job_rows,
    )
    _publication_figures(metric_rows)
    _write_publication_report(
        campaign,
        metric_rows,
        aggregate_metrics,
        aggregate_classical,
        anchor_summary,
        intervals,
        contrasts,
        effect_summary,
        job_rows,
    )
    campaign["state"] = "complete"
    campaign["analysis_completed_at_utc"] = utc_now()
    save_campaign(campaign)
    print(f"Publication analysis written to {ANALYSIS_DIR}", flush=True)


def print_plan() -> None:
    validate_protocol_constants()
    template = campaign_template()
    output = {
        "configurations": template["configurations"],
        "protocol": template["protocol"],
        "workload": template["workload"],
        "hardware_schedule": template["hardware_schedule"],
        "estimated_batches": math.ceil(
            template["workload"]["hardware_condition_runs"] / CONDITIONS_PER_BATCH
        ),
        "result_root": str(RESULT_ROOT),
        "qpu_submission_requires": "--authorize-qpu",
    }
    print(json.dumps(output, indent=2), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "plan",
            "prepare",
            "simulate",
            "submit",
            "status",
            "wait",
            "retrieve",
            "analyze",
            "all",
        ),
    )
    parser.add_argument(
        "--authorize-qpu",
        action="store_true",
        help="Required explicit authorization for submit/all real-QPU commands.",
    )
    parser.add_argument(
        "--exact-only",
        action="store_true",
        help="For simulate, skip ideal/noisy Aer and IBM service access.",
    )
    parser.add_argument(
        "--allow-concurrent-simulation",
        action="store_true",
        help=(
            "For submit only, permit QPU submission after every exact kernel is "
            "complete while ideal/noisy Aer simulations continue locally."
        ),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        print_plan()
    elif args.command == "prepare":
        prepare_all()
    elif args.command == "simulate":
        simulate_all(include_aer=not args.exact_only)
    elif args.command == "submit":
        submit_next_batch(
            args.authorize_qpu,
            allow_concurrent_simulation=args.allow_concurrent_simulation,
        )
    elif args.command == "status":
        refresh_hardware_status()
    elif args.command == "wait":
        wait_for_active_jobs(args.poll_seconds)
    elif args.command == "retrieve":
        service = _service()
        refresh_hardware_status(service=service)
        retrieve_done_jobs(service=service)
    elif args.command == "analyze":
        analyze_results()
    elif args.command == "all":
        run_all(args.authorize_qpu)


if __name__ == "__main__":
    main()
