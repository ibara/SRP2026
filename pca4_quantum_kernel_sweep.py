#!/usr/bin/env python3
"""Tune a prepared PCA-4 quantum kernel without selecting on the outer test set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC


RANGE_NAMES = ("minus_one_to_one", "zero_to_one")
REPETITIONS = (1, 2)
ENTANGLEMENTS = ("linear", "full")
C_VALUES = (0.1, 1.0, 10.0, 100.0)


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


def _transform_range(values: np.ndarray, range_name: str) -> np.ndarray:
    if range_name == "minus_one_to_one":
        return values.copy()
    if range_name == "zero_to_one":
        # This is exactly the affine relationship between MinMaxScaler's
        # (-1, 1) and (0, 1) outputs when fitted on the same training data.
        return (values + 1.0) / 2.0
    raise ValueError(f"Unknown input range: {range_name}")


def _make_kernel(reps: int, entanglement: str) -> FidelityStatevectorKernel:
    feature_map = zz_feature_map(
        feature_dimension=4,
        reps=reps,
        entanglement=entanglement,
    )
    return FidelityStatevectorKernel(
        feature_map=feature_map,
        shots=None,
        auto_clear_cache=False,
        enforce_psd=True,
    )


def _kernel_diagnostics(kernel_matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    n_rows = len(labels)
    row_index, col_index = np.triu_indices(n_rows, k=1)
    off_diagonal = kernel_matrix[row_index, col_index]
    same_class = labels[row_index] == labels[col_index]

    centering = np.eye(n_rows) - np.ones((n_rows, n_rows)) / n_rows
    centered_kernel = centering @ kernel_matrix @ centering
    signed_labels = 2 * labels.astype(float) - 1
    target_kernel = np.outer(signed_labels, signed_labels)
    centered_target = centering @ target_kernel @ centering
    denominator = np.linalg.norm(centered_kernel) * np.linalg.norm(centered_target)
    alignment = (
        float(np.sum(centered_kernel * centered_target) / denominator)
        if denominator
        else float("nan")
    )

    raw_eigenvalues = np.linalg.eigvalsh((kernel_matrix + kernel_matrix.T) / 2)
    eigenvalues = np.clip(raw_eigenvalues, 0, None)
    squared_sum = float(np.sum(eigenvalues**2))
    effective_rank = (
        float(np.sum(eigenvalues) ** 2 / squared_sum) if squared_sum else 0.0
    )
    return {
        "centered_kernel_target_alignment": alignment,
        "within_class_similarity_mean": float(np.mean(off_diagonal[same_class])),
        "between_class_similarity_mean": float(np.mean(off_diagonal[~same_class])),
        "within_minus_between_similarity": float(
            np.mean(off_diagonal[same_class])
            - np.mean(off_diagonal[~same_class])
        ),
        "off_diagonal_mean": float(np.mean(off_diagonal)),
        "off_diagonal_std": float(np.std(off_diagonal)),
        "effective_rank": effective_rank,
        "minimum_eigenvalue": float(np.min(raw_eigenvalues)),
    }


def _cross_validate_c(
    kernel_matrix: np.ndarray,
    labels: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    c_value: float,
) -> dict[str, Any]:
    accuracies: list[float] = []
    balanced_accuracies: list[float] = []
    roc_aucs: list[float] = []

    for fit_indices, validation_indices in folds:
        fit_kernel = kernel_matrix[np.ix_(fit_indices, fit_indices)]
        validation_kernel = kernel_matrix[np.ix_(validation_indices, fit_indices)]
        model = SVC(kernel="precomputed", C=c_value)
        model.fit(fit_kernel, labels[fit_indices])
        predictions = model.predict(validation_kernel)
        decision_scores = model.decision_function(validation_kernel)
        accuracies.append(
            float(accuracy_score(labels[validation_indices], predictions))
        )
        balanced_accuracies.append(
            float(
                balanced_accuracy_score(
                    labels[validation_indices], predictions
                )
            )
        )
        roc_aucs.append(
            float(roc_auc_score(labels[validation_indices], decision_scores))
        )

    return {
        "fold_accuracy": accuracies,
        "fold_balanced_accuracy": balanced_accuracies,
        "fold_roc_auc": roc_aucs,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "mean_balanced_accuracy": float(np.mean(balanced_accuracies)),
        "std_balanced_accuracy": float(np.std(balanced_accuracies)),
        "mean_roc_auc": float(np.mean(roc_aucs)),
        "std_roc_auc": float(np.std(roc_aucs)),
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Prefer CV performance, then stability and the simpler configuration."""

    return (
        row["mean_balanced_accuracy"],
        row["mean_roc_auc"],
        -row["std_balanced_accuracy"],
        -row["reps"],
        1.0 if row["entanglement"] == "linear" else 0.0,
        -abs(math.log10(row["C"])),
        1.0 if row["input_range"] == "minus_one_to_one" else 0.0,
    )


def _test_metrics(
    k_train: np.ndarray,
    k_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    c_value: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model = SVC(kernel="precomputed", C=c_value)
    model.fit(k_train, y_train)
    predictions = model.predict(k_test)
    decision_scores = model.decision_function(k_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "roc_auc": float(roc_auc_score(y_test, decision_scores)),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Benign", "Malware"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, predictions, decision_scores


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = []
    for row in rows:
        scalar_rows.append(
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-results",
        type=Path,
        default=Path("results/cic_pca4_exact_seed42/results.npz"),
        help="Prepared outer split and reference exact kernel from the repaired run.",
    )
    parser.add_argument(
        "--dataset-name",
        default="CIC-YNU-IoT",
        help="Dataset label recorded in the output summary.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/cic_pca4_kernel_sweep_seed42"),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    archive = np.load(args.prepared_results)
    x_train = np.asarray(archive["x_train"], dtype=float)
    y_train = np.asarray(archive["y_train"], dtype=int)

    if x_train.shape != (140, 4):
        raise ValueError(f"Expected prepared 140 x 4 training data, got {x_train.shape}")
    if not np.array_equal(np.unique(y_train, return_counts=True)[1], [70, 70]):
        raise ValueError("Expected 70 benign and 70 malware training rows")

    splitter = StratifiedKFold(
        n_splits=args.folds,
        shuffle=True,
        random_state=args.random_state,
    )
    folds = list(splitter.split(x_train, y_train))
    rows: list[dict[str, Any]] = []
    matrices: dict[tuple[str, int, str], np.ndarray] = {}

    total_variants = len(RANGE_NAMES) * len(REPETITIONS) * len(ENTANGLEMENTS)
    variant_number = 0
    reference_difference = None

    for range_name in RANGE_NAMES:
        transformed_train = _transform_range(x_train, range_name)
        for reps in REPETITIONS:
            for entanglement in ENTANGLEMENTS:
                variant_number += 1
                print(
                    f"[{variant_number}/{total_variants}] range={range_name}, "
                    f"reps={reps}, entanglement={entanglement}",
                    flush=True,
                )
                kernel = _make_kernel(reps, entanglement)
                k_train = np.asarray(kernel.evaluate(transformed_train), dtype=float)
                matrices[(range_name, reps, entanglement)] = k_train

                if (
                    range_name == "minus_one_to_one"
                    and reps == 2
                    and entanglement == "linear"
                ):
                    reference_difference = float(
                        np.max(np.abs(k_train - archive["k_train"]))
                    )
                    if reference_difference > 1e-10:
                        raise RuntimeError(
                            "Statevector sweep kernel does not match the saved "
                            f"FidelityQuantumKernel (max error {reference_difference})"
                        )

                diagnostics = _kernel_diagnostics(k_train, y_train)
                for c_value in C_VALUES:
                    cv_metrics = _cross_validate_c(
                        k_train, y_train, folds, c_value
                    )
                    rows.append(
                        {
                            "input_range": range_name,
                            "reps": reps,
                            "entanglement": entanglement,
                            "C": c_value,
                            **cv_metrics,
                            **diagnostics,
                        }
                    )

    if reference_difference is None:
        raise RuntimeError("The reference kernel variant was not evaluated")

    selected = max(rows, key=_selection_key)
    selected_key = (
        selected["input_range"],
        selected["reps"],
        selected["entanglement"],
    )
    print(
        "Selected from training CV: "
        f"range={selected['input_range']}, reps={selected['reps']}, "
        f"entanglement={selected['entanglement']}, C={selected['C']}",
        flush=True,
    )

    # Access the outer test arrays only after model selection is complete.
    x_test = np.asarray(archive["x_test"], dtype=float)
    y_test = np.asarray(archive["y_test"], dtype=int)
    selected_train = _transform_range(x_train, selected["input_range"])
    selected_test = _transform_range(x_test, selected["input_range"])
    selected_kernel = _make_kernel(selected["reps"], selected["entanglement"])
    k_train = matrices[selected_key]
    # Populate the selected kernel's cache with training states before inference.
    selected_kernel.evaluate(selected_train)
    k_test = np.asarray(
        selected_kernel.evaluate(selected_test, selected_train), dtype=float
    )
    test_metrics, predictions, decision_scores = _test_metrics(
        k_train, k_test, y_train, y_test, selected["C"]
    )

    original_metrics, _, _ = _test_metrics(
        np.asarray(archive["k_train"], dtype=float),
        np.asarray(archive["k_test"], dtype=float),
        y_train,
        y_test,
        1.0,
    )
    original_predictions = np.asarray(archive["predictions"], dtype=int)
    original_correct = original_predictions == y_test
    selected_correct = predictions == y_test
    fixed_errors = int(np.sum(~original_correct & selected_correct))
    introduced_errors = int(np.sum(original_correct & ~selected_correct))
    discordant = fixed_errors + introduced_errors
    lower_tail = sum(
        math.comb(discordant, index)
        for index in range(min(fixed_errors, introduced_errors) + 1)
    )
    exact_mcnemar_p = (
        min(1.0, 2 * lower_tail / (2**discordant)) if discordant else 1.0
    )
    paired_comparison = {
        "accuracy_delta": test_metrics["accuracy"] - original_metrics["accuracy"],
        "balanced_accuracy_delta": (
            test_metrics["balanced_accuracy"]
            - original_metrics["balanced_accuracy"]
        ),
        "roc_auc_delta": test_metrics["roc_auc"] - original_metrics["roc_auc"],
        "original_errors_fixed": fixed_errors,
        "new_errors_introduced": introduced_errors,
        "both_correct": int(np.sum(original_correct & selected_correct)),
        "both_wrong": int(np.sum(~original_correct & ~selected_correct)),
        "exact_mcnemar_two_sided_p": exact_mcnemar_p,
        "interpretation": (
            "The paired accuracy gain is not statistically significant at "
            "alpha=0.05 on this single 60-row test set."
        ),
    }
    elapsed = time.perf_counter() - started
    summary = {
        "experiment": (
            f"{args.dataset_name} PCA-4 exact quantum-kernel training-only sweep"
        ),
        "dataset": args.dataset_name,
        "prepared_results": args.prepared_results,
        "outer_split": {
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "random_state": 42,
            "note": (
                "The sweep does not use outer-test features or labels until after "
                "selection. The outer test had already been evaluated in prior "
                "project work, so it is not pristine at the project level."
            ),
        },
        "selection": {
            "method": f"{args.folds}-fold stratified CV on the outer training set",
            "metric": "mean balanced accuracy",
            "tie_breakers": [
                "mean ROC AUC",
                "lower balanced-accuracy standard deviation",
                "fewer feature-map repetitions",
                "linear entanglement",
                "C closest to 1",
                "minus_one_to_one range",
            ],
            "candidate_space": {
                "input_ranges": RANGE_NAMES,
                "repetitions": REPETITIONS,
                "entanglements": ENTANGLEMENTS,
                "C_values": C_VALUES,
            },
            "selected": selected,
        },
        "kernel_equivalence_check": {
            "reference": "saved FidelityQuantumKernel training matrix",
            "replacement": "exact FidelityStatevectorKernel training matrix",
            "maximum_absolute_difference": reference_difference,
            "tolerance": 1e-10,
            "passed": reference_difference <= 1e-10,
        },
        "selected_outer_test_metrics": test_metrics,
        "original_outer_test_metrics": original_metrics,
        "paired_outer_test_comparison": paired_comparison,
        "elapsed_seconds": elapsed,
        "software": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
            "qiskit": version("qiskit"),
            "qiskit-machine-learning": version("qiskit-machine-learning"),
        },
        "all_cv_results": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(args.output_dir / "cv_results.csv", rows)
    np.savez_compressed(
        args.output_dir / "selected_kernel_results.npz",
        x_train=selected_train,
        x_test=selected_test,
        y_train=y_train,
        y_test=y_test,
        k_train=k_train,
        k_test=k_test,
        predictions=predictions,
        decision_scores=decision_scores,
    )

    print(
        f"CV balanced accuracy: {selected['mean_balanced_accuracy']:.4f} "
        f"+/- {selected['std_balanced_accuracy']:.4f}",
        flush=True,
    )
    print(
        f"Outer-test accuracy: {test_metrics['accuracy']:.4f}; "
        f"balanced accuracy: {test_metrics['balanced_accuracy']:.4f}; "
        f"ROC AUC: {test_metrics['roc_auc']:.4f}",
        flush=True,
    )
    print(f"Confusion matrix: {test_metrics['confusion_matrix']}", flush=True)
    print(f"Saved results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
