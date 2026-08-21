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

Hardware/exact kernel correlation was 0.978 (train) and 0.987 (test). This run demonstrates successful real-hardware execution, not quantum advantage: it used one small split, one QPU job, and no explicit deduplication. The three hardware/exact prediction disagreements were near the SVC decision boundary.

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
