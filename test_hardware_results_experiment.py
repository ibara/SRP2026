"""Offline integrity tests for the frozen hardware results experiment.

These tests use synthetic arrays only.  They do not read project datasets,
connect to IBM Runtime, or submit quantum jobs.
"""

from __future__ import annotations

import unittest
from unittest import mock
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from qiskit import qpy
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    thermal_relaxation_error,
)

import hardware_results_experiment as experiment


class ProtocolTests(unittest.TestCase):
    def test_frozen_workload_and_schedule(self) -> None:
        experiment.validate_protocol_constants()
        campaign = experiment.campaign_template()
        experiment.validate_campaign_protocol(campaign)
        workload = campaign["workload"]

        self.assertEqual(len(campaign["configurations"]), 6)
        self.assertEqual(workload["evaluations_per_condition"], 9_560)
        self.assertEqual(workload["executions_per_condition"], 4_894_720)
        self.assertEqual(workload["primary_hardware_conditions"], 18)
        self.assertEqual(workload["repeat_anchor_conditions"], 3)
        self.assertEqual(workload["hardware_condition_runs"], 21)
        self.assertEqual(workload["total_hardware_evaluations"], 200_760)
        self.assertEqual(workload["total_hardware_executions"], 102_789_120)

        schedule = campaign["hardware_schedule"]
        self.assertEqual(len({row["condition_id"] for row in schedule}), 21)
        repeats = [row for row in schedule if row["repeat"] > 0]
        self.assertEqual(len(repeats), 3)
        self.assertEqual(
            {(row["configuration"], row["fold"]) for row in repeats},
            set(experiment.ANCHORS),
        )
        for anchor in experiment.ANCHORS:
            matching = [
                row
                for row in schedule
                if (row["configuration"], row["fold"]) == anchor
            ]
            self.assertEqual({row["repeat"] for row in matching}, {0, 1})

        changed = experiment._jsonable(campaign)
        changed["protocol"]["shots"] = 1_024
        with self.assertRaisesRegex(RuntimeError, "frozen protocol"):
            experiment.validate_campaign_protocol(changed)

    def test_pair_counts(self) -> None:
        train_pairs, test_pairs = experiment.train_test_pairs(
            experiment.TRAIN_ROWS, experiment.TEST_ROWS
        )
        self.assertEqual(train_pairs.shape, (3_160, 2))
        self.assertEqual(test_pairs.shape, (6_400, 2))
        self.assertTrue(np.all(train_pairs[:, 0] < train_pairs[:, 1]))
        self.assertEqual(
            experiment.evaluation_count(experiment.TRAIN_ROWS, experiment.TEST_ROWS),
            9_560,
        )

    def test_real_qpu_requires_explicit_flag(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "--authorize-qpu"):
            experiment.submit_next_batch(False)


class DataIntegrityTests(unittest.TestCase):
    def test_deduplication_and_conflict_exclusion(self) -> None:
        x = np.asarray(
            [
                [1.0, 2.0],
                [1.0, 2.0],
                [3.0, 4.0],
                [3.0, 4.0],
                [5.0, np.nan],
                [5.0, np.nan],
            ]
        )
        y = np.asarray([0, 0, 0, 1, 1, 1])
        source_ids = np.asarray(["z", "a", "c", "d", "f", "e"])
        pool, summary = experiment._deduplicate_pool(
            x, y, source_ids, architectures=None
        )

        self.assertEqual(pool["source_ids"].tolist(), ["a", "e"])
        self.assertEqual(pool["y"].tolist(), [0, 1])
        self.assertEqual(summary["duplicate_rows_removed"], 2)
        self.assertEqual(summary["conflicting_groups_excluded"], 1)
        self.assertEqual(summary["conflicting_rows_excluded"], 2)
        self.assertEqual(
            experiment.row_hashes(np.asarray([[np.nan], [np.nan]])).tolist()[0],
            experiment.row_hashes(np.asarray([[np.nan], [np.nan]])).tolist()[1],
        )
        self.assertEqual(
            experiment.row_hashes(np.asarray([[-0.0]])).item(),
            experiment.row_hashes(np.asarray([[0.0]])).item(),
        )

    @staticmethod
    def _ordinary_pool(rows_per_class: int = 350) -> dict[str, np.ndarray]:
        total = rows_per_class * 2
        return {
            "x": np.arange(total * 6, dtype=float).reshape(total, 6),
            "y": np.repeat([0, 1], rows_per_class),
            "source_ids": np.asarray([f"row:{index}" for index in range(total)]),
            "architectures": np.full(total, "", dtype="U1"),
        }

    def test_non_cic_folds_are_balanced_disjoint_and_reserved(self) -> None:
        pool = self._ordinary_pool()
        folds = experiment.build_fold_indices("clamp", pool)
        all_test = np.concatenate([fold["test"] for fold in folds])

        self.assertEqual(
            len(np.unique(all_test)), experiment.N_FOLDS * experiment.TEST_ROWS
        )
        reserved = set(all_test.tolist())
        for fold in folds:
            self.assertEqual(len(fold["train"]), experiment.TRAIN_ROWS)
            self.assertEqual(len(fold["test"]), experiment.TEST_ROWS)
            self.assertEqual(
                np.bincount(pool["y"][fold["train"]], minlength=2).tolist(),
                [experiment.ROWS_PER_CLASS, experiment.ROWS_PER_CLASS],
            )
            self.assertEqual(
                np.bincount(pool["y"][fold["test"]], minlength=2).tolist(),
                [experiment.ROWS_PER_CLASS, experiment.ROWS_PER_CLASS],
            )
            self.assertFalse(set(fold["train"].tolist()) & reserved)

    def test_cic_folds_match_architecture_within_class(self) -> None:
        rows_per_stratum = 90
        labels: list[int] = []
        architectures: list[str] = []
        for label in (0, 1):
            for architecture in experiment.CIC_ARCHITECTURES:
                labels.extend([label] * rows_per_stratum)
                architectures.extend([architecture] * rows_per_stratum)
        y = np.asarray(labels)
        arch = np.asarray(architectures)
        pool = {
            "x": np.arange(len(y) * 6, dtype=float).reshape(len(y), 6),
            "y": y,
            "source_ids": np.asarray([f"cic:{index}" for index in range(len(y))]),
            "architectures": arch,
        }

        folds = experiment.build_fold_indices("cic", pool)
        self.assertEqual(
            len(np.unique(np.concatenate([fold["test"] for fold in folds]))),
            experiment.N_FOLDS * experiment.TEST_ROWS,
        )
        for fold_number, fold in enumerate(folds):
            expected = experiment.cic_arch_counts(fold_number)
            for kind in ("train", "test"):
                for architecture, count in expected.items():
                    for label in (0, 1):
                        observed = np.sum(
                            (arch[fold[kind]] == architecture)
                            & (y[fold[kind]] == label)
                        )
                        self.assertEqual(observed, count)

    def test_preprocessing_shapes_range_and_training_only_fit(self) -> None:
        rng = np.random.default_rng(17)
        raw_train = rng.normal(size=(experiment.TRAIN_ROWS, 8))
        raw_test = rng.normal(loc=0.25, size=(experiment.TEST_ROWS, 8))
        raw_train[0, 0] = np.nan
        raw_test[1, 1] = np.nan

        native_train, native_test, native_meta, native_arrays = (
            experiment._transform_fold(raw_train, raw_test, "native")
        )
        pca_train, pca_test, pca_meta, _ = experiment._transform_fold(
            raw_train, raw_test, "pca"
        )

        self.assertEqual(native_train.shape, (experiment.TRAIN_ROWS, 8))
        self.assertEqual(native_test.shape, (experiment.TEST_ROWS, 8))
        self.assertEqual(pca_train.shape, (experiment.TRAIN_ROWS, 4))
        self.assertEqual(pca_test.shape, (experiment.TEST_ROWS, 4))
        for values in (native_train, native_test, pca_train, pca_test):
            self.assertTrue(np.isfinite(values).all())
            self.assertGreaterEqual(float(values.min()), 0.0)
            self.assertLessEqual(float(values.max()), 1.0)
        self.assertAlmostEqual(
            native_arrays["imputer_statistics"][0],
            float(np.nanmedian(raw_train[:, 0])),
        )
        self.assertEqual(native_meta["test_rows_matching_transformed_train_vector"], 0)
        self.assertEqual(pca_meta["pca_components"], 4)

    def test_wide_ember_loader_selects_rows_in_column_chunks(self) -> None:
        row_count = 200
        feature_count = 70
        columns = {
            f"feature_{column}": np.arange(row_count, dtype=float) * 1_000 + column
            for column in range(feature_count)
        }
        columns["Label"] = np.arange(row_count, dtype=np.int8) % 2
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wide.parquet"
            pq.write_table(pa.table(columns), path, row_group_size=row_count)
            parquet_file = pq.ParquetFile(path)
            selected = np.asarray([199, 3, 51])
            names = [f"feature_{column}" for column in range(feature_count)]
            x, y, source_ids = experiment._load_selected_ember_rows(
                parquet_file,
                names,
                selected,
                feature_chunk_size=16,
                row_batch_size=31,
            )

        expected_indices = np.sort(selected)
        self.assertEqual(x.shape, (3, feature_count))
        np.testing.assert_array_equal(y, expected_indices % 2)
        self.assertEqual(
            source_ids.tolist(),
            [f"ember2018:{index}" for index in expected_indices],
        )
        for row, source_index in enumerate(expected_indices):
            np.testing.assert_array_equal(
                x[row], source_index * 1_000 + np.arange(feature_count)
            )


class KernelAndAnalysisTests(unittest.TestCase):
    @staticmethod
    def _synthetic_metric_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        mode_effect = {
            "exact": 0.00,
            "ideal_aer": -0.01,
            "noisy_aer": -0.04,
            "hardware": -0.06,
        }
        for configuration in experiment.CONFIGURATIONS:
            for fold in range(experiment.N_FOLDS):
                for mode in experiment.EXECUTION_MODES:
                    value = 0.75 + 0.01 * fold + mode_effect[mode]
                    rows.append(
                        {
                            "configuration": configuration.name,
                            "dataset": configuration.dataset,
                            "representation": configuration.representation,
                            "qubits": configuration.qubits,
                            "fold": fold,
                            "repeat": 0,
                            "mode": mode,
                            "accuracy": value,
                            "balanced_accuracy": value,
                            "roc_auc": min(value + 0.1, 1.0),
                            "test_relative_frobenius_error": (
                                0.0 if mode == "exact" else 0.1 - mode_effect[mode]
                            ),
                            "test_kernel_correlation": (
                                1.0 if mode == "exact" else 0.95 + mode_effect[mode]
                            ),
                            "qpu_seconds": 100 + configuration.qubits
                            if mode == "hardware"
                            else "",
                        }
                    )
        for configuration_name, fold in experiment.ANCHORS:
            configuration = experiment.CONFIG_BY_NAME[configuration_name]
            for repeat in range(1, experiment.EXTRA_RUNS_PER_ANCHOR + 1):
                rows.append(
                    {
                        "configuration": configuration.name,
                        "dataset": configuration.dataset,
                        "representation": configuration.representation,
                        "qubits": configuration.qubits,
                        "fold": fold,
                        "repeat": repeat,
                        "mode": "hardware",
                        "accuracy": 0.70 + 0.01 * repeat,
                        "balanced_accuracy": 0.70 + 0.01 * repeat,
                        "roc_auc": 0.80 + 0.01 * repeat,
                        "test_relative_frobenius_error": 0.16 + 0.01 * repeat,
                        "test_kernel_correlation": 0.88 - 0.01 * repeat,
                        "qpu_seconds": 100 + configuration.qubits + repeat,
                    }
                )
        return rows

    def test_psd_projection_is_symmetric_unit_diagonal_and_psd(self) -> None:
        indefinite = np.asarray(
            [[1.0, 1.4, -0.2], [1.4, 1.0, 0.8], [-0.2, 0.8, 1.0]]
        )
        projected, diagnostics = experiment._project_psd(indefinite)
        np.testing.assert_allclose(projected, projected.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(projected), 1.0, atol=1e-12)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(projected).min()), -1e-10)
        self.assertLess(diagnostics["raw_minimum_eigenvalue"], 0)

    def test_parameter_binding_follows_transpiled_parameter_order(self) -> None:
        circuit, _, _ = experiment._fidelity_circuit(4)
        left = np.arange(12, dtype=float).reshape(3, 4)
        right = left + 100
        bound = experiment.parameter_values(circuit, left, right)

        self.assertEqual(bound.shape, (3, 8))
        for column, parameter in enumerate(circuit.parameters):
            side, index_text = experiment._PARAMETER_PATTERN.match(parameter.name).groups()
            expected = left if side == "left" else right
            np.testing.assert_array_equal(bound[:, column], expected[:, int(index_text)])

    def test_dense_aer_relabeling_preserves_circuit_and_physical_noise(self) -> None:
        angle = Parameter("angle")
        circuit = QuantumCircuit(7, 2)
        circuit.rz(angle, 1)
        circuit.cx(1, 5)
        circuit.measure(1, 0)
        circuit.measure(5, 1)
        dense, mapping = experiment._dense_active_circuit(circuit)

        self.assertEqual(mapping, {1: 0, 5: 1})
        self.assertEqual(dense.num_qubits, 2)
        self.assertEqual(dense.num_clbits, 2)
        self.assertEqual(dense.count_ops(), circuit.count_ops())
        self.assertEqual(set(dense.parameters), {angle})

        noise = NoiseModel(basis_gates=["rz", "cx"])
        noise.add_quantum_error(depolarizing_error(0.01, 1), "rz", [1])
        noise.add_quantum_error(depolarizing_error(0.02, 2), "cx", [1, 5])
        noise.add_quantum_error(depolarizing_error(0.03, 1), "rz", [3])
        readout = ReadoutError([[0.98, 0.02], [0.03, 0.97]])
        noise.add_readout_error(readout, [1])
        noise.add_readout_error(readout, [5])
        noise.add_readout_error(readout, [3])

        dense_noise, metadata = experiment._dense_active_noise_model(noise, mapping)
        self.assertEqual(metadata["copied_quantum_errors"], 2)
        self.assertEqual(metadata["copied_readout_errors"], 2)
        serialized = dense_noise.to_dict(serializable=True)["errors"]
        gate_qubits = {
            tuple(error["gate_qubits"][0])
            for error in serialized
            if "gate_qubits" in error
        }
        self.assertIn((0,), gate_qubits)
        self.assertIn((1,), gate_qubits)
        self.assertIn((0, 1), gate_qubits)

    def test_frozen_aer_execution_can_be_reloaded(self) -> None:
        circuit = QuantumCircuit(2, 2)
        circuit.sx(0)
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        noise = NoiseModel(basis_gates=["sx", "cx"])
        noise.add_quantum_error(
            thermal_relaxation_error(50_000, 70_000, 100), "sx", [0]
        )
        noise.add_quantum_error(depolarizing_error(0.02, 2), "cx", [0, 1])

        with tempfile.TemporaryDirectory() as temporary:
            circuits = Path(temporary)
            with (circuits / "aer_active_fidelity_2q.qpy").open("wb") as handle:
                qpy.dump(circuit, handle)
            experiment.atomic_write_json(
                circuits / "aer_active_fidelity_2q.json",
                {
                    "logical_qubits": 2,
                    "aer_circuit_qubits": 2,
                    "operations": {
                        key: int(value) for key, value in circuit.count_ops().items()
                    },
                },
            )
            experiment.atomic_write_json(
                circuits / "aer_active_noise_model_2q.json",
                noise.to_dict(serializable=True),
            )
            with mock.patch.object(experiment, "CIRCUIT_DIR", circuits):
                loaded, ideal, noisy = experiment.load_frozen_aer_execution(2)

        self.assertEqual(loaded.count_ops(), circuit.count_ops())
        self.assertEqual(ideal.options.method, "statevector")
        self.assertEqual(noisy.options.method, "statevector")
        self.assertEqual(noisy.options.noise_model.noise_qubits, [0, 1])

    def test_kernel_reconstruction(self) -> None:
        train_values = np.linspace(0.0, 1.0, 3_160)
        test_values = np.linspace(1.0, 0.0, 6_400)
        train, test = experiment.kernels_from_fidelities(
            train_values, test_values
        )
        self.assertEqual(train.shape, (80, 80))
        self.assertEqual(test.shape, (80, 80))
        np.testing.assert_allclose(train, train.T)
        np.testing.assert_array_equal(np.diag(train), np.ones(80))
        self.assertAlmostEqual(train[0, 1], train_values[0])
        self.assertAlmostEqual(test[0, 0], test_values[0])

    def test_exact_mcnemar_and_holm(self) -> None:
        truth = np.asarray([0, 0, 1, 1])
        left = np.asarray([0, 1, 1, 0])
        right = np.asarray([0, 0, 0, 1])
        left_only, right_only, p_value = experiment._exact_mcnemar(
            left, right, truth
        )
        self.assertEqual((left_only, right_only), (1, 2))
        self.assertEqual(p_value, 1.0)
        adjusted = experiment._holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_bootstrap_resamples_within_each_fold(self) -> None:
        rows = []
        for fold in range(experiment.N_FOLDS):
            truth = np.repeat([0, 1], 10)
            prediction = truth.copy()
            prediction[:fold] = 1
            scores = np.linspace(-1, 1, len(truth))
            for position in range(len(truth)):
                rows.append(
                    {
                        "configuration": "synthetic",
                        "mode": "exact",
                        "fold": fold,
                        "repeat": 0,
                        "truth": int(truth[position]),
                        "prediction": int(prediction[position]),
                        "decision_score": float(scores[position]),
                    }
                )
        with mock.patch.object(experiment, "BOOTSTRAP_REPLICATES", 25):
            intervals = experiment._bootstrap_metric_intervals(rows)

        self.assertEqual(len(intervals), 3)
        balanced = next(
            row for row in intervals if row["metric"] == "balanced_accuracy"
        )
        expected = np.mean(
            [1 - fold / 20 for fold in range(experiment.N_FOLDS)]
        )
        self.assertAlmostEqual(balanced["estimate"], expected)
        self.assertIn("within fold", balanced["bootstrap_design"])

    def test_results_figures_render_all_formats(self) -> None:
        rows = self._synthetic_metric_rows()
        predictions = [
            {"mode": row["mode"]}
            for row in rows
            for _ in range(experiment.TEST_ROWS)
        ]
        experiment._validate_analysis_rows(rows, predictions)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis = root / "analysis"
            circuits = root / "circuits"
            analysis.mkdir()
            circuits.mkdir()
            for width, cz in ((4, 22), (8, 50), (10, 70)):
                experiment.atomic_write_json(
                    circuits / f"fidelity_{width}q.json",
                    {
                        "logical_qubits": width,
                        "depth": width * 10,
                        "size": width * 20,
                        "estimated_duration_seconds": width * 1e-6,
                        "operations": {"cz": cz},
                    },
                )
            with (
                mock.patch.object(experiment, "ANALYSIS_DIR", analysis),
                mock.patch.object(
                    experiment, "LATEX_TABLE_DIR", analysis / "tables"
                ),
                mock.patch.object(experiment, "CIRCUIT_DIR", circuits),
            ):
                experiment._results_figures(rows)
                anchors = experiment._aggregate_anchor_repeats(rows)
                aggregate = experiment._aggregate_metric_rows(rows)
                _, effect_summary = experiment._compression_effects(rows)
                classical = [
                    {
                        "configuration": configuration.name,
                        "kernel": kernel,
                        "folds": experiment.N_FOLDS,
                        "balanced_accuracy_mean": 0.7,
                        "balanced_accuracy_standard_deviation": 0.02,
                        "roc_auc_mean": 0.8,
                        "roc_auc_standard_deviation": 0.03,
                    }
                    for configuration in experiment.CONFIGURATIONS
                    for kernel in ("linear", "rbf")
                ]
                intervals = [
                    {
                        "configuration": configuration.name,
                        "mode": mode,
                        "metric": "balanced_accuracy",
                        "estimate": 0.7,
                        "ci_low": 0.65,
                        "ci_high": 0.75,
                        "test_rows": experiment.N_FOLDS * experiment.TEST_ROWS,
                    }
                    for configuration in experiment.CONFIGURATIONS
                    for mode in experiment.EXECUTION_MODES
                ]
                contrasts = [
                    {
                        "contrast": "synthetic",
                        "paired_test_rows": experiment.N_FOLDS * experiment.TEST_ROWS,
                        "left_only_correct": 5,
                        "right_only_correct": 7,
                        "mcnemar_exact_p": 0.5,
                        "holm_adjusted_p": 1.0,
                    }
                ]
                job_rows = [
                    {
                        "condition_id": condition["condition_id"],
                        "attempt": 1,
                        "job_id": f"job-{condition['order']}",
                        "qpu_seconds": 100,
                        "status": "DONE",
                        "retrieved_at_utc": "2026-01-01T00:00:00+00:00",
                    }
                    for condition in experiment.planned_hardware_schedule()
                ]
                experiment._resource_tables(rows)
                experiment._results_latex_tables(
                    aggregate,
                    classical,
                    anchors,
                    intervals,
                    contrasts,
                    effect_summary,
                    job_rows,
                )
                experiment._write_results_report(
                    experiment.campaign_template(),
                    rows,
                    aggregate,
                    classical,
                    anchors,
                    intervals,
                    contrasts,
                    effect_summary,
                    job_rows,
                )

            self.assertEqual(len(anchors), 3)
            for figure in (
                "figure1_noise_ladder_balanced_accuracy",
                "figure2_kernel_distortion",
                "figure3_circuit_cost_vs_degradation",
            ):
                for suffix in ("png", "pdf", "svg"):
                    path = analysis / f"{figure}.{suffix}"
                    self.assertTrue(path.is_file())
                    self.assertGreater(path.stat().st_size, 0)
            report = analysis / "results.txt"
            self.assertIn("Successful retrieved conditions: 21 / 21", report.read_text())
            self.assertTrue((analysis / "resource_scaling_estimates.csv").is_file())
            captions = analysis / "figure_captions.tex"
            self.assertIn("\\caption{", captions.read_text())
            self.assertIn("95\\%", captions.read_text())
            tables = analysis / "tables"
            self.assertEqual(len(list(tables.glob("table*.tex"))), 9)
            self.assertIn(
                "\\caption",
                (tables / "table1_primary_quantum_results.tex").read_text(),
            )
            self.assertTrue((tables / "all_tables.tex").is_file())


if __name__ == "__main__":
    unittest.main()
