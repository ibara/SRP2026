# SRP2026

## CLaMP quantum-kernel hardware run

The CLaMP workflow from `QuantumSimCLaMP(2).ipynb` was adapted to IBM Runtime in `clamp_hardware_runner.py` and executed on a real QPU.

### Run

- Backend: `ibm_rensselaer`
- Job: `da10qv63kjvs73865pfg` (`DONE`)
- Model: precomputed fidelity-kernel SVC
- Feature map: 10-qubit `ZZFeatureMap`, 2 reps, linear entanglement
- Data: 20 train / 10 test, 10 CLaMP PE-header features
- Workload: 390 kernel evaluations, 1,024 shots each (399,360 total shots)
- Transpiled circuit: depth 206, 70 CZ gates, optimization level 3
- QPU usage: 110 s

### Results

| Kernel | Accuracy | Balanced accuracy | ROC AUC |
|---|---:|---:|---:|
| Hardware quantum | 1.00 | 1.00 | 1.00 |
| Exact quantum | 0.70 | 0.75 | 1.00 |
| Classical linear | 0.70 | 0.625 | 0.75 |
| Classical RBF | 0.70 | 0.625 | 0.625 |

Hardware/exact kernel correlation was 0.978 (train) and 0.987 (test). The three hardware/exact prediction disagreements were near the SVC decision boundary.

### Metric definitions

- **Accuracy:** fraction of all predictions that are correct.
- **Balanced accuracy:** mean recall across the two classes, `(malware recall + benign recall) / 2`. It weights both classes equally even when their counts differ. For the exact kernel, malware recall was 1.00 and benign recall was 0.50, giving `(1.00 + 0.50) / 2 = 0.75`.
- **ROC AUC:** area under the receiver operating characteristic curve. It evaluates how well decision scores rank malware above benign samples over every possible threshold: 0.50 is random ranking and 1.00 is perfect ranking. It does not require good calibration at the SVC's selected threshold. The exact kernel therefore had ROC AUC 1.00 but accuracy 0.70: its ranking was perfect, while three benign scores fell just above the default decision boundary.

### Artifacts

- `hardware_results/clamp_10q_r2_20train_10test/submission.json`: job and transpilation metadata
- `hardware_results/clamp_10q_r2_20train_10test/hardware_summary.json`: metrics and QPU usage
- `hardware_results/clamp_10q_r2_20train_10test/baseline_summary.json`: exact, shot-based, and classical baselines
- `hardware_results/clamp_10q_r2_20train_10test/hardware_results.npz`: measured kernels, fidelities, scores, and predictions
- `hardware_results/clamp_10q_r2_20train_10test/prepared_data.npz`: exact sampled inputs and parameters
- `hardware_results/clamp_10q_r2_20train_10test/transpiled_fidelity_circuit.qpy`: submitted ISA circuit

### Retrieve

The runner uses a locally saved Qiskit Runtime account; no API token is committed.

```bash
.venv/bin/python clamp_hardware_runner.py status
.venv/bin/python clamp_hardware_runner.py retrieve
```

### Proposed compact benchmark

Use a shallow circuit so every declared condition can be run end-to-end on both hardware and Qiskit Aer:

1. Deduplicate exact full feature rows before splitting; audit or remove duplicate groups with conflicting labels. Keep each duplicate group wholly within one split.
2. For each seed, use a group-aware stratified split with a balanced test set of 20 samples and nested balanced training sets of 8, 12, 16, and 20 samples.
3. Select four fixed features from that seed's training pool only and reuse them at every training size. Use a 4-qubit `ZZFeatureMap` with 1 rep, linear entanglement, and 512 shots.
4. Repeat five seeded splits. For every split and training size, run the identical inputs on IBM hardware and shot-matched ideal Aer; optionally add Aer with the backend noise model.
5. Save one row per run with `seed`, `n_train`, `backend`, accuracy, balanced accuracy, ROC AUC, kernel MAE, and job ID.

Plot training size on the x-axis and mean test accuracy on the y-axis, with separate hardware and Aer lines and 95% confidence intervals. Because the test set is balanced, ordinary and class-balanced performance are directly comparable. Also report the paired hardware-minus-Aer accuracy for each seed and training size. The largest point requires `20*19/2 + 20*20 = 590` kernel evaluations. This is a complete compact benchmark, not a full-5,184-sample kernel evaluation.

### Caveats and future work

- The completed 20/10 run did not explicitly deduplicate before splitting, so duplicate leakage remains possible.
- One small split and one QPU job cannot establish generalization, statistical significance, or quantum advantage. Hardware noise moved three near-boundary predictions in the favorable direction.
- The proposed 4-qubit benchmark is intentionally classically simulable; its purpose is reproducible hardware validation. If results remain stable across seeds, extend to deeper maps, more features, repeated hardware jobs, and stronger classical baselines.
