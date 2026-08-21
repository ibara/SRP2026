# SRP2026

## Scope

I reviewed the group's workflows and adapted Allison's CLaMP version for the RPI-IBM 127-qubit Quantum System One. I treat the completed run as evidence of hardware feasibility, not quantum advantage. For our paper, I propose studying how feature compression and NISQ hardware noise affect quantum-kernel geometry and downstream malware classification.

## Completed CLaMP hardware run

I adapted `QuantumSimCLaMP(2).ipynb` to the current IBM Runtime interface in `clamp_hardware_runner.py` and executed the resulting fidelity-kernel circuit on the QC.

### Configuration

- Backend: `ibm_rensselaer`
- Model: precomputed fidelity-kernel SVC, `C=1.0`
- Feature map: 10-qubit `ZZFeatureMap`, 2 reps, linear entanglement
- Data: 20 train / 10 test, 10 CLaMP PE-header features
- Workload: 390 kernel evaluations, 1,024 shots each (399,360 total shots)
- Transpilation: optimization level 3, depth 206, 70 CZ gates
- QPU usage: 110 s

### Results

| Kernel | Accuracy | Balanced accuracy | ROC AUC |
|---|---:|---:|---:|
| Hardware quantum | 1.00 | 1.00 | 1.00 |
| Exact quantum | 0.70 | 0.75 | 1.00 |
| Classical linear | 0.70 | 0.625 | 0.75 |
| Classical RBF | 0.70 | 0.625 | 0.625 |

I measured hardware/exact kernel correlations of 0.978 on the training kernel and 0.987 on the test kernel. The relative Frobenius errors were 0.474 and 0.475, respectively, and hardware and exact predictions disagreed on 3/10 test samples. All three disagreements were close to the SVC decision boundary; hardware noise moved them in the favorable direction.

I do not interpret the 10/10 hardware accuracy as evidence that noise improved the classifier or that this method has a quantum advantage.

### Metric definitions

- **Accuracy:** fraction of all predictions that are correct.
- **Balanced accuracy:** average of malware recall and benign recall. This weights both classes equally when their counts differ. For the exact kernel, malware recall was 1.00 and benign recall was 0.50, so balanced accuracy was `(1.00 + 0.50) / 2 = 0.75`.
- **ROC AUC:** area under the receiver operating characteristic curve to measure how well decision scores rank malware above benign samples over all thresholds. A value of 0.50 represents random ranking and 1.00 represents perfect ranking. The exact kernel had ROC AUC 1.00 but accuracy 0.70 because its ranking was perfect while three benign scores fell just above the default SVC threshold.

The original CLaMP notebook calculates ROC AUC from hard predictions. For subsequent experiments, I would calculate it from `decision_function` scores, as my hardware runner does.

## CLaMP duplicate analysis

I think there's some duplication happening in the datasets that we should worry about as it may skew our data. E.g., in the 5,184-row CLaMP dataset:

- Across all 55 non-label columns, 770 rows belong to 146 duplicate groups.
- Across the selected 10 quantum features, 1,461 rows belong to 317 duplicate groups.
- No duplicate groups have conflicting labels.
- In the notebook's original 80/20 split, 141/1,037 test rows (13.6%) match a full training feature vector and 267/1,037 (25.7%) match a selected 10-feature training vector. This isn't good.
- In the notebook's 200/100 Aer subset, 5/100 test rows match a full training vector and 9/100 match a selected 10-feature vector.
- In my completed 20/10 hardware subset, 0/10 test rows match a training vector under either definition.

The completed hardware sample has no direct duplicate leakage, but the larger notebook result may be optimistic. I would deduplicate or use group-aware splitting before making comparative claims.

## My thoughts on the group's quantum workflows

I do not compare the saved accuracies as if they came from one benchmark: the notebooks use different datasets, feature transformations, circuit widths, sample sizes, and splits.

| Workflow | Current experiment | Saved result | Approximate kernel evaluations | My hardware assessment |
|---|---|---:|---:|---|
| Allison: CLaMP | 200 train / 100 test; 10 native features; 10 qubits | Aer accuracy 0.860 | 39,900 | Best anchor bc already ran on hardware, but expensive and may have duplication |
| Sean: EMBER 2018 | 140 train / 60 test; PCA to 4 features; 4 qubits | Exact/simulator accuracy 0.767 | 18,130 | Best comparison candidate after obtaining the missing parquet data |
| Allison: EMBER2024 | 200 train / 50 test; 10 features; 10 qubits | Aer accuracy 0.540 | 29,900 | Not hardware-ready as a primary model; malware recall is only 0.04 |
| Sean: CIC-YNU-IoT | 140 train / 60 test; PCA to 4 features; 4 qubits | Saved accuracy 0.733 | 18,130 | Circuit is feasible, but the saved result is invalid for comparison until preprocessing is repaired |
| Fabiana: malware-recognition prototype | Approximately 928 train / 232 test; 8 features; 8 qubits | No valid quantum result | About 645,424 | Not hardware-ready without subsampling and code repair |

### Interpretation

- CLaMP is not clearly superior to EMBER 2018. The approximate Wilson 95% accuracy intervals overlap: 0.779–0.915 for CLaMP Aer and 0.646–0.856 for EMBER 2018. The 10/10 CLaMP hardware result has a much wider interval of approximately 0.722–1.000.
- Sean's EMBER 2018 pipeline is the strongest hardware control because PCA reduces it to four qubits while keeping a balanced benign/malware task.
- A fresh matched transpilation on `ibm_rensselaer` produced depth 98 and 22 CZ gates for the four-qubit fidelity circuit, versus depth 206 and 70 CZ gates for the ten-qubit circuit. This would give a concrete circuit-cost comparison.
- Allison's EMBER2024 model currently predicts nearly every sample as benign. I would redesign or compress it before spending QPU time.
- Sean's CIC-YNU-IoT notebook preprocesses the data once before constructing the training kernel, then fits a second preprocessing pipeline before constructing the test kernel. The test-kernel columns therefore no longer correspond to the training points used for the training kernel. I would repair and rerun this result before including it.
- Fabiana's notebook evaluates an unbounded full kernel and later references `qy_test` instead of `y_test`. I would add a controlled subset and fix the metric cell before considering hardware execution.

Only the CLaMP raw dataset is currently available in this checkout. The EMBER 2018, EMBER2024, CIC-YNU-IoT, and Fabiana datasets are referenced through machine-specific absolute paths, so I cannot reproduce or submit those workflows until the group supplies the data or portable download/preparation scripts.

## My recommended paper experiment

My proposed research question is:

> How do feature compression and NISQ hardware noise affect quantum-kernel geometry and malware-classification performance?

I would compare three primary configurations:

1. **CLaMP-native-10:** Allison's 10 native CLaMP features mapped to 10 qubits.
2. **CLaMP-PCA-4:** Sean's four-component PCA strategy applied to the same CLaMP samples.
3. **EMBER2018-PCA-4:** Sean's four-qubit EMBER 2018 workflow, after its dataset is made available.

The first two configurations isolate the feature-compression and circuit-width tradeoff on one dataset. The third tests whether the four-qubit result transfers to a larger, established malware dataset.

### Benchmark protocol

For each configuration, I propose the following:

1. Deduplicate exact full feature rows before splitting. I would audit or remove groups with conflicting labels and keep each duplicate group wholly within one split.
2. For each of five seeds, create a group-aware stratified split with a balanced test set of 20 samples and nested balanced training sets of 8, 12, 16, and 20 samples.
3. Fit imputation, scaling, feature selection, and PCA on the training pool only. I will reuse the fitted transformation for every nested size and for the test set.
4. Use the same `ZZFeatureMap`, entanglement, SVC hyperparameters, transpiler settings, shots, and backend within each matched comparison. I will use 512 shots for the complete grid and repeat representative points at 1,024 shots.
5. Evaluate exact statevector/Aer, shot-matched ideal Aer, optional backend-noise-model Aer, and real IBM hardware on identical samples.
6. Train linear and RBF classical SVMs on the exact same transformed samples.
7. Repeat at least one representative hardware condition three times to separate device/run variability from sample variability.

For `n_train=20` and `n_test=20`, each condition requires `20*19/2 + 20*20 = 590` kernel evaluations. I call this a complete compact benchmark because every declared condition is executed end-to-end; it is not a full-5,184-sample quantum-kernel evaluation.

### Measurements

I would report:

- accuracy and balanced accuracy;
- ROC AUC from SVC decision scores;
- confusion matrices and per-class recall;
- hardware-versus-exact kernel MAE, RMSE, Pearson correlation, and relative Frobenius error;
- kernel PSD violations and any PSD correction applied;
- circuit depth, CZ count, physical layout, shots, QPU time, backend calibration date, and job ID;
- paired hardware-minus-Aer performance for every seed and training size.

I can save one row per run with at least `dataset`, `representation`, `seed`, `n_train`, `backend`, `shots`, `accuracy`, `balanced_accuracy`, `roc_auc`, `kernel_mae`, `kernel_correlation`, `depth`, `cz_count`, `qpu_seconds`, and `job_id`.

### Proposed figures

1. Plot training size on the x-axis and mean balanced accuracy on the y-axis, with separate exact/Aer, noisy-Aer, and hardware lines and 95% confidence intervals. I will facet this plot by dataset or representation.
2. Plot qubit count or CZ count against the hardware-minus-Aer accuracy gap and hardware/exact kernel error. This directly tests whether larger circuits lose usable kernel geometry on hardware.
3. Show paired hardware and Aer results for each seed rather than only aggregate bars.
4. Include confusion matrices at the largest training size to expose class-specific failure modes.

This design supports an empirical hardware-robustness claim. It does not, by itself, support a claim of computational quantum advantage.

## Artifacts

I saved the complete CLaMP run under `hardware_results/clamp_10q_r2_20train_10test/`:

- `submission.json`: job, circuit, and transpilation metadata
- `hardware_summary.json`: metrics, kernel comparisons, and QPU usage
- `baseline_summary.json`: exact, shot-based, and classical baselines
- `hardware_results.npz`: measured kernels, fidelities, scores, and predictions
- `prepared_data.npz`: exact sampled inputs, source indices, labels, and parameters
- `transpiled_fidelity_circuit.qpy`: submitted ISA circuit

## Retrieving the completed job

I configured the runner to use a locally saved Qiskit Runtime account; I did not commit an API token.

```bash
.venv/bin/python clamp_hardware_runner.py status
.venv/bin/python clamp_hardware_runner.py retrieve
```

## Caveats and future work

- I ran only one small 20/10 hardware split and one QPU job. This cannot establish generalization, statistical significance, or quantum advantage.
- Although my hardware subset has no direct duplicate overlap, the source CLaMP split was not group-deduplicated before sampling.
- The favorable hardware prediction changes occurred near the SVC threshold and may be noise-sensitive.
- The proposed four-qubit benchmark is intentionally classically simulable; I use it to validate hardware behavior, not to claim classical intractability.
- I need the group to provide the non-CLaMP datasets and I need to repair the CIC-YNU-IoT and Fabiana workflows before including them.
- If the compact results remain stable across seeds and hardware repetitions, I will extend the study to more features, deeper maps, additional malware datasets, and stronger classical baselines.
