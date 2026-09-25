# Quantum Malware Hardware Results

This repository studies how feature compression and NISQ hardware noise affect quantum-kernel geometry and malware-classification performance. The central result is that IBM hardware noticeably distorted the measured quantum kernels—especially for wider circuits—while downstream classification remained remarkably stable. This supports hardware robustness in this experiment, not quantum advantage.

## Experiment

- Six malware-classification configurations across CLaMP, Fabiana, CIC-YNU-IoT, and EMBER 2018.
- Three deterministic, duplicate-safe splits per configuration, each with 80 balanced training rows and 80 balanced test rows.
- A fixed two-repetition, linearly entangled `ZZFeatureMap` and precomputed-kernel `SVC(C=1)`.
- Four execution modes: exact statevector, ideal Aer at 512 shots, noisy Aer at 512 shots, and IBM hardware at 512 shots.
- 54 simulation conditions and 21 IBM hardware runs completed successfully, with no failed or retried hardware jobs.

## Results

The values below are mean balanced accuracy across the three held-out splits.

| Configuration | Exact | IBM hardware | Change |
|---|---:|---:|---:|
| CLaMP native, 10 qubits | 84.2% | 84.6% | +0.4 pp |
| CLaMP PCA, 4 qubits | 75.0% | 74.6% | -0.4 pp |
| Fabiana native, 8 qubits | 97.5% | 95.8% | -1.7 pp |
| Fabiana PCA, 4 qubits | 93.8% | 92.9% | -0.8 pp |
| CIC PCA, 4 qubits | 69.2% | 70.4% | +1.3 pp |
| EMBER 2018 PCA, 4 qubits | 61.7% | 61.3% | -0.4 pp |

## Key findings

- Hardware changed balanced accuracy by only -1.67 to +1.25 percentage points. None of the six hardware-versus-exact differences was statistically significant after Holm correction.
- Kernel distortion increased sharply with circuit width. Mean test relative Frobenius error was approximately 0.112 at 4 qubits, 0.441 at 8 qubits, and 0.479 at 10 qubits.
- CZ count strongly tracked kernel distortion (`r=0.857`) but essentially did not track accuracy degradation (`r=-0.032`). Substantially altered kernel values could therefore still produce similar SVM decisions.
- PCA substantially reduced circuit cost. CLaMP's native 10-qubit circuit had depth 206 and 70 CZ gates, compared with depth 98 and 22 CZ gates for the 4-qubit representation.
- Compression carried a predictive tradeoff. CLaMP PCA lost about 10 percentage points of hardware balanced accuracy relative to native CLaMP; the paired hardware predictions favored the native representation after Holm correction (`p=0.0179`).
- Classical RBF models matched or exceeded the quantum method on CLaMP, CIC, and EMBER 2018. Fabiana's quantum results were stronger, but the overall evidence does not establish a consistent quantum advantage.

## Interpretation

Feature compression reduced circuit cost and hardware-induced kernel distortion, but it could also discard useful predictive information. At the same time, malware-classification decisions remained stable even when NISQ hardware substantially deformed the wider quantum kernels. The study therefore demonstrates robustness of this controlled small-sample workflow, not full-corpus detector performance or quantum advantage.

## Results files

- [Consolidated results report](hardware_results/results/analysis/results.txt)
- [Execution-noise ladder](hardware_results/results/analysis/figure1_noise_ladder_balanced_accuracy.png)
- [Kernel distortion](hardware_results/results/analysis/figure2_kernel_distortion.png)
- [Circuit cost versus degradation](hardware_results/results/analysis/figure3_circuit_cost_vs_degradation.png)
- [LaTeX tables](hardware_results/results/analysis/tables/all_tables.tex)
- [Figure captions](hardware_results/results/analysis/figure_captions.tex)

Machine-readable run metrics, predictions, confidence intervals, paired tests, resource estimates, and complete hardware-job traceability are available in [`hardware_results/results/analysis/`](hardware_results/results/analysis/).
