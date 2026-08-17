"""Submit and retrieve the small CLaMP quantum-kernel hardware experiment.

No API token is stored here. The script loads an existing named Qiskit Runtime
account and resolves the requested IBM instance by its human-readable name.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from qiskit import qpy
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit_machine_learning.utils import algorithm_globals


ACCOUNT_NAME = "fishing"
INSTANCE_NAME = "General-dedicated"
BACKEND_NAME = "ibm_rensselaer"
DATA_PATH = Path("ClaMP_Raw-5184.csv")
RESULT_DIR = Path("hardware_results/clamp_10q_r2_20train_10test")
METADATA_PATH = RESULT_DIR / "submission.json"
PREPARED_PATH = RESULT_DIR / "prepared_data.npz"
BASELINES_PATH = RESULT_DIR / "baseline_summary.json"
HARDWARE_PATH = RESULT_DIR / "hardware_results.npz"
SUMMARY_PATH = RESULT_DIR / "hardware_summary.json"
CIRCUIT_PATH = RESULT_DIR / "transpiled_fidelity_circuit.qpy"

FEATURES = [
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
]

NUM_QUBITS = 10
REPS = 2
SHOTS = 1024
SEED = 42


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _status_name(status: Any) -> str:
    return getattr(status, "name", str(status))


def _service() -> QiskitRuntimeService:
    account_service = QiskitRuntimeService(name=ACCOUNT_NAME)
    matches = [
        item
        for item in account_service.instances()
        if item.get("name") == INSTANCE_NAME and item.get("plan") == "on-prem"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one on-prem instance named {INSTANCE_NAME!r}; found {len(matches)}."
        )
    # The explicit instance overrides the retired instance saved under this account.
    return QiskitRuntimeService(name=ACCOUNT_NAME, instance=matches[0]["crn"])


def _data() -> dict[str, np.ndarray]:
    frame = pd.read_csv(DATA_PATH)
    x = frame[FEATURES]
    y = frame["class"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=10, stratify=y
    )
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # Reproduce the exact hardware subset in the working notebook. The preceding
    # simulator draws affect NumPy's legacy RNG state, so they are retained here.
    np.random.seed(SEED)
    np.random.choice(len(x_train_scaled), 500, replace=False)
    np.random.choice(len(x_test_scaled), 150, replace=False)
    train_positions = np.random.choice(len(x_train_scaled), 20, replace=False)
    test_positions = np.random.choice(len(x_test_scaled), 10, replace=False)

    return {
        "x_train": x_train_scaled[train_positions],
        "x_test": x_test_scaled[test_positions],
        "y_train": y_train.iloc[train_positions].to_numpy(dtype=int),
        "y_test": y_test.iloc[test_positions].to_numpy(dtype=int),
        "train_positions": train_positions,
        "test_positions": test_positions,
        "train_source_indices": x_train.index.to_numpy()[train_positions],
        "test_source_indices": x_test.index.to_numpy()[test_positions],
        "scaler_min": scaler.data_min_,
        "scaler_max": scaler.data_max_,
    }


def _pairs(n_train: int, n_test: int) -> tuple[np.ndarray, np.ndarray]:
    train_pairs = np.asarray(
        [(i, j) for i in range(n_train) for j in range(i + 1, n_train)], dtype=int
    )
    test_pairs = np.asarray(
        [(i, j) for i in range(n_test) for j in range(n_train)], dtype=int
    )
    return train_pairs, test_pairs


def _classification_metrics(
    k_train: np.ndarray,
    k_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model = SVC(kernel="precomputed", C=1.0)
    model.fit(k_train, y_train)
    predictions = model.predict(k_test)
    scores = model.decision_function(k_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, scores)),
        "predictions": predictions.tolist(),
        "decision_scores": scores.tolist(),
        "truth": y_test.tolist(),
    }
    return metrics, predictions, scores


def _project_psd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(matrix)
    return (vectors @ np.diag(np.maximum(0, values)) @ vectors.transpose()).real


def _prepare_baselines(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    feature_map = zz_feature_map(
        feature_dimension=NUM_QUBITS, reps=REPS, entanglement="linear"
    )
    exact_kernel = FidelityStatevectorKernel(
        feature_map=feature_map, auto_clear_cache=False
    )
    exact_train = exact_kernel.evaluate(data["x_train"])
    exact_test = exact_kernel.evaluate(data["x_test"], data["x_train"])
    exact_metrics, _, _ = _classification_metrics(
        exact_train, exact_test, data["y_train"], data["y_test"]
    )

    algorithm_globals.random_seed = SEED
    shot_kernel = FidelityStatevectorKernel(
        feature_map=feature_map,
        auto_clear_cache=False,
        shots=SHOTS,
        enforce_psd=True,
    )
    shot_train = shot_kernel.evaluate(data["x_train"])
    shot_test = shot_kernel.evaluate(data["x_test"], data["x_train"])
    shot_metrics, _, _ = _classification_metrics(
        shot_train, shot_test, data["y_train"], data["y_test"]
    )

    classical: dict[str, Any] = {}
    for kernel in ("linear", "rbf"):
        model = SVC(kernel=kernel, C=1.0)
        model.fit(data["x_train"], data["y_train"])
        predictions = model.predict(data["x_test"])
        scores = model.decision_function(data["x_test"])
        classical[kernel] = {
            "accuracy": float(accuracy_score(data["y_test"], predictions)),
            "balanced_accuracy": float(
                balanced_accuracy_score(data["y_test"], predictions)
            ),
            "roc_auc": float(roc_auc_score(data["y_test"], scores)),
            "predictions": predictions.tolist(),
        }

    _write_json(
        BASELINES_PATH,
        {
            "exact_quantum_kernel": exact_metrics,
            "ideal_1024_shot_quantum_kernel": shot_metrics,
            "classical_svc": classical,
        },
    )
    return exact_train, exact_test


def _fidelity_circuit() -> tuple[Any, ParameterVector, ParameterVector]:
    left_parameters = ParameterVector("left", NUM_QUBITS)
    right_parameters = ParameterVector("right", NUM_QUBITS)
    left = zz_feature_map(
        feature_dimension=NUM_QUBITS, reps=REPS, entanglement="linear"
    ).assign_parameters(left_parameters)
    right = zz_feature_map(
        feature_dimension=NUM_QUBITS, reps=REPS, entanglement="linear"
    ).assign_parameters(right_parameters)
    circuit = left.compose(right.inverse())
    circuit.measure_all()
    return circuit, left_parameters, right_parameters


def _parameter_values(
    circuit: Any,
    left_parameters: ParameterVector,
    right_parameters: ParameterVector,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> np.ndarray:
    rows = []
    for left_row, right_row in zip(left_values, right_values):
        assignments = {
            **{parameter: value for parameter, value in zip(left_parameters, left_row)},
            **{parameter: value for parameter, value in zip(right_parameters, right_row)},
        }
        rows.append([assignments[parameter] for parameter in circuit.parameters])
    return np.asarray(rows, dtype=float)


def submit() -> None:
    if METADATA_PATH.exists():
        previous = _read_json(METADATA_PATH)
        raise RuntimeError(
            "A submission already exists at "
            f"{METADATA_PATH} (job {previous.get('job_id')}); refusing to duplicate it."
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    data = _data()
    exact_train, exact_test = _prepare_baselines(data)
    train_pairs, test_pairs = _pairs(len(data["x_train"]), len(data["x_test"]))

    service = _service()
    backend = service.backend(BACKEND_NAME)
    status = backend.status()
    if not status.operational:
        raise RuntimeError(f"Backend {BACKEND_NAME} is not operational.")

    raw_circuit, left_parameters, right_parameters = _fidelity_circuit()
    pass_manager = generate_preset_pass_manager(
        backend=backend, optimization_level=3, seed_transpiler=SEED
    )
    isa_circuit = pass_manager.run(raw_circuit)
    expected_parameters = set(left_parameters) | set(right_parameters)
    if set(isa_circuit.parameters) != expected_parameters:
        raise RuntimeError("Transpilation changed the fidelity circuit parameter set.")

    train_values = _parameter_values(
        isa_circuit,
        left_parameters,
        right_parameters,
        data["x_train"][train_pairs[:, 0]],
        data["x_train"][train_pairs[:, 1]],
    )
    test_values = _parameter_values(
        isa_circuit,
        left_parameters,
        right_parameters,
        data["x_test"][test_pairs[:, 0]],
        data["x_train"][test_pairs[:, 1]],
    )

    with CIRCUIT_PATH.open("wb") as handle:
        qpy.dump(isa_circuit, handle)

    layout: Any = None
    try:
        layout = isa_circuit.layout.final_index_layout(filter_ancillas=True)
    except Exception:
        layout = str(isa_circuit.layout)

    np.savez_compressed(
        PREPARED_PATH,
        **data,
        train_pairs=train_pairs,
        test_pairs=test_pairs,
        exact_train_kernel=exact_train,
        exact_test_kernel=exact_test,
        train_parameter_values=train_values,
        test_parameter_values=test_values,
    )

    sampler = SamplerV2(mode=backend)
    job = sampler.run(
        [
            (isa_circuit, train_values),
            (isa_circuit, test_values),
        ],
        shots=SHOTS,
    )

    try:
        job_status_at_submission = _status_name(job.status())
    except Exception as error:
        # The job ID is the important durable handle; a transient status-query
        # failure must not leave a successfully submitted job orphaned.
        job_status_at_submission = f"Unavailable: {type(error).__name__}: {error}"

    properties_date = None
    try:
        properties_date = backend.properties().last_update_date
    except Exception:
        pass
    metadata = {
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "account_name": ACCOUNT_NAME,
        "instance_name": INSTANCE_NAME,
        "backend": backend.name,
        "backend_properties_last_update": properties_date,
        "backend_pending_jobs_at_submission": status.pending_jobs,
        "job_id": job.job_id(),
        "job_status_at_submission": job_status_at_submission,
        "shots": SHOTS,
        "train_size": int(len(data["x_train"])),
        "test_size": int(len(data["x_test"])),
        "train_kernel_circuits": int(len(train_pairs)),
        "test_kernel_circuits": int(len(test_pairs)),
        "total_executions": int((len(train_pairs) + len(test_pairs)) * SHOTS),
        "feature_names": FEATURES,
        "feature_map": {
            "name": "zz_feature_map",
            "num_qubits": NUM_QUBITS,
            "reps": REPS,
            "entanglement": "linear",
        },
        "transpilation": {
            "optimization_level": 3,
            "seed_transpiler": SEED,
            "depth": int(isa_circuit.depth()),
            "size": int(isa_circuit.size()),
            "num_qubits": int(isa_circuit.num_qubits),
            "num_clbits": int(isa_circuit.num_clbits),
            "operations": {key: int(value) for key, value in isa_circuit.count_ops().items()},
            "final_index_layout": layout,
        },
    }
    _write_json(METADATA_PATH, metadata)
    print(json.dumps(metadata, indent=2, default=str), flush=True)


def _load_job() -> tuple[dict[str, Any], Any]:
    if not METADATA_PATH.exists():
        raise RuntimeError(f"No submission metadata found at {METADATA_PATH}.")
    metadata = _read_json(METADATA_PATH)
    service = _service()
    return metadata, service.job(metadata["job_id"])


def status() -> None:
    metadata, job = _load_job()
    current = job.status()
    value = {
        "job_id": metadata["job_id"],
        "backend": metadata["backend"],
        "status": _status_name(current),
        "error_message": job.error_message() if _status_name(current) in {"ERROR", "CANCELLED"} else None,
    }
    print(json.dumps(value, indent=2, default=str), flush=True)


def _fidelities(pub_result: Any, expected: int) -> tuple[np.ndarray, int]:
    bit_array = pub_result.join_data()
    if bit_array.size != expected:
        raise RuntimeError(f"Expected {expected} parameter results; received {bit_array.size}.")
    shots = int(bit_array.num_shots)
    values = np.asarray(
        [bit_array.get_int_counts(i).get(0, 0) / shots for i in range(expected)],
        dtype=float,
    )
    return values, shots


def _comparison(observed: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    delta = observed - exact
    correlation = float(np.corrcoef(observed.ravel(), exact.ravel())[0, 1])
    denominator = float(np.linalg.norm(exact))
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta**2))),
        "pearson_correlation": correlation,
        "relative_frobenius_error": float(np.linalg.norm(delta) / denominator),
    }


def retrieve() -> None:
    metadata, job = _load_job()
    current = _status_name(job.status())
    if current != "DONE":
        raise RuntimeError(f"Job {metadata['job_id']} is {current}, not DONE.")

    prepared = np.load(PREPARED_PATH)
    result = job.result()
    train_pairs = prepared["train_pairs"]
    test_pairs = prepared["test_pairs"]
    train_fidelities, train_shots = _fidelities(result[0], len(train_pairs))
    test_fidelities, test_shots = _fidelities(result[1], len(test_pairs))

    n_train = len(prepared["y_train"])
    n_test = len(prepared["y_test"])
    raw_train = np.eye(n_train)
    for value, (i, j) in zip(train_fidelities, train_pairs):
        raw_train[i, j] = value
        raw_train[j, i] = value
    hardware_test = np.empty((n_test, n_train), dtype=float)
    for value, (i, j) in zip(test_fidelities, test_pairs):
        hardware_test[i, j] = value
    psd_train = _project_psd(raw_train)

    metrics, predictions, scores = _classification_metrics(
        psd_train,
        hardware_test,
        prepared["y_train"],
        prepared["y_test"],
    )
    exact_metrics, exact_predictions, _ = _classification_metrics(
        prepared["exact_train_kernel"],
        prepared["exact_test_kernel"],
        prepared["y_train"],
        prepared["y_test"],
    )

    train_mask = ~np.eye(n_train, dtype=bool)
    comparison = {
        "train_off_diagonal": _comparison(
            raw_train[train_mask], prepared["exact_train_kernel"][train_mask]
        ),
        "test": _comparison(hardware_test, prepared["exact_test_kernel"]),
        "prediction_disagreement_fraction": float(
            np.mean(predictions != exact_predictions)
        ),
        "raw_train_min_eigenvalue": float(np.linalg.eigvalsh(raw_train).min()),
        "psd_train_min_eigenvalue": float(np.linalg.eigvalsh(psd_train).min()),
    }

    np.savez_compressed(
        HARDWARE_PATH,
        raw_train_kernel=raw_train,
        psd_train_kernel=psd_train,
        test_kernel=hardware_test,
        train_fidelities=train_fidelities,
        test_fidelities=test_fidelities,
        predictions=predictions,
        decision_scores=scores,
    )

    job_details: dict[str, Any] = {}
    for name, function in (("metrics", job.metrics), ("usage_seconds", job.usage)):
        try:
            job_details[name] = function()
        except Exception as error:
            job_details[name] = f"Unavailable: {type(error).__name__}: {error}"

    summary = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": metadata["job_id"],
        "backend": metadata["backend"],
        "status": current,
        "actual_shots": {"train": train_shots, "test": test_shots},
        "hardware_quantum_kernel": metrics,
        "exact_quantum_kernel": exact_metrics,
        "hardware_vs_exact_kernel": comparison,
        "job_details": job_details,
    }
    _write_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2, default=str), flush=True)


def wait_and_retrieve() -> None:
    metadata, job = _load_job()
    print(
        f"Waiting for IBM job {metadata['job_id']} on {metadata['backend']}...",
        flush=True,
    )
    final_states = {"DONE", "ERROR", "CANCELLED"}
    previous = None
    while True:
        current = _status_name(job.status())
        if current != previous:
            print(
                f"{datetime.now(timezone.utc).isoformat()} status={current}",
                flush=True,
            )
            previous = current
        if current in final_states:
            break
        time.sleep(20)
    print(f"Final status: {current}", flush=True)
    retrieve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("submit", "status", "retrieve", "wait"))
    args = parser.parse_args()
    {"submit": submit, "status": status, "retrieve": retrieve, "wait": wait_and_retrieve}[
        args.command
    ]()


if __name__ == "__main__":
    main()
