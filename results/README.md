# Core results

These are the **mean balanced accuracies across three held-out
splits**, each with 80 balanced training and 80 balanced test rows 
using a two-repetition, linearly entangled ZZ feature map, `SVC(C=1)`,
and 512 shots run on IBM-RPI Quantum System One Nighthawk r2 QPU hardware.

| Configuration | Exact | IBM hardware | Change |
|---|---:|---:|---:|
| CLaMP native, 10 qubits | 84.2% | 84.6% | +0.4 pp |
| CLaMP PCA, 4 qubits | 75.0% | 74.6% | -0.4 pp |
| MBR native, 8 qubits | 97.5% | 95.8% | -1.7 pp |
| MBR PCA, 4 qubits | 93.8% | 92.9% | -0.8 pp |
| CIC PCA, 4 qubits | 69.2% | 70.4% | +1.3 pp |
| EMBER 2018 PCA, 4 qubits | 61.7% | 61.3% | -0.4 pp |

[reported_metrics.csv](reported_metrics.csv) contains the same rounded values.
Changes were calculated before rounding; `pp` means percentage points.
These results describe the controlled small-sample experiment and do not establish
full-corpus performance or quantum advantage.

The main notebook saves `notebook_<dataset>_pca<qubits>_fold<fold>.csv` with accuracy,
balanced accuracy and ROC AUC as fractions from 0 to 1, plus dataset checksum and
run settings. Those files describe a **single local Aer split**, so they are not
reproductions of the hardware averages above. Rerunning the same setting replaces
its notebook CSV.

Detailed historical results and the other findings remain
in [Student_Work/](../Student_Work/README_before_reorganization.md), with full
source metrics in the [archived analysis folder](../Student_Work/hardware_results/results/analysis/).
