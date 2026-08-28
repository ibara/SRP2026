# One-shot Codex prompt: complete publication hardware campaign

Copy everything below this line into Codex once, from the repository root.

---

Work in the current SRP2026 repository and complete the frozen publication experiment described in `README.md`. Read the complete README and `publication_hardware_experiment.py` before acting. Also inspect `git status` and preserve every existing user/contributor change. Do not edit the original contributor notebooks (`QSVM big dataset-Copy1.ipynb`, `Ember 2018 Quantum.ipynb`, or `fabiana.ipynb`) or the completed pilot runner/results. Do not redesign, tune, or shrink the experiment after seeing results.

This prompt explicitly authorizes real IBM Quantum hardware submission and its associated QPU usage for the complete frozen campaign implemented by `publication_hardware_experiment.py`: 18 primary hardware conditions plus three preregistered anchor reruns, for 21 successful hardware runs total. It also authorizes the runner's single retry for a condition that ends in error or cancellation. Use the saved Qiskit Runtime account `fishing`, instance `General-dedicated`, and backend `ibm_rensselaer`. Do not expose, print, edit, or commit credentials. Do not submit any conditions beyond the frozen schedule and its permitted retries.

The protocol must remain exactly:

- configurations: `clamp_native10`, `clamp_pca4`, `fabiana_native8`, `fabiana_pca4`, `cic_pca4`, and `ember2018_pca4`;
- three deterministic balanced held-out splits using training seeds 101, 204, and 303, each with 80 training and 80 test rows;
- mutually disjoint test sets within each dataset, with all 240 reserved test rows excluded from every training set;
- CLaMP and Fabiana native/PCA pairs using identical source rows and labels in each split;
- median imputation and all scaling/PCA fitted on training data only; PCA has four components and final inputs are scaled to `[0,1]`;
- Qiskit `zz_feature_map`, two repetitions, linear entanglement, fidelity kernel, precomputed `SVC(C=1)`;
- exact, 512-shot ideal Aer, 512-shot initial-backend-noise Aer, and 512-shot hardware modes on identical inputs;
- no error mitigation;
- the three split-0 anchors at 4, 8, and 10 qubits measured twice each;
- a maximum of 12 conditions per eight-hour Runtime batch and no more than two attempts per condition.

Before any QPU submission, perform these preflight checks:

1. Compile the runner and run its offline test suite. Fix only genuine implementation defects needed to execute the frozen protocol; keep such fixes isolated in the new publication runner/tests and document them. Never change protocol constants just to make execution easier.
2. Run `.venv/bin/python publication_hardware_experiment.py plan` and verify it reports six configurations, 18 primary conditions, three repeats, 21 hardware condition runs, 9,560 fidelity evaluations and 4,894,720 circuit executions per condition, 200,760 total hardware fidelity evaluations, 102,789,120 total hardware circuit executions, and two planned batches.
3. Verify all declared local datasets exist and the saved IBM account/backend can be accessed without displaying credentials. Confirm the backend is operational.
4. Prepare the duplicate-safe splits and run every exact, ideal-Aer, noisy-Aer, linear-SVC, and RBF-SVC baseline. Do not submit hardware unless all 54 quantum simulation condition/mode results are complete and all built-in pairing/leakage assertions pass.

Then execute the complete resumable campaign with:

```bash
.venv/bin/python publication_hardware_experiment.py all --authorize-qpu
```

The experiment may run overnight or longer. Keep monitoring it until the terminal outcome rather than stopping after submission. Poll running commands/jobs often enough to provide concise progress updates at least once per minute while actively working. Let queued jobs remain queued. Use the campaign's recovery commands (`status`, `wait`, `retrieve`, and then `all --authorize-qpu`) after a process interruption or context compaction. Always inspect `hardware_results/publication_3split_80x80_final_v3/campaign.json` before resuming so no completed or active condition is resubmitted. Never create a second campaign directory to work around an interrupted job.

IBM limits or transient service failures do not authorize changing samples, shots, circuits, backend, or methodology. The runner already keeps each condition below the documented 10-million-execution limit. If a job genuinely fails or is cancelled, allow the one recorded retry. If the same condition exhausts both attempts, or credentials/backend access become a hard blocker, preserve all manifests and partial results, exhaust safe read-only diagnostics, and report the exact blocker and recovery command. Do not substitute a simulator result for hardware.

After all 21 hardware conditions are successful, retrieve every result and run the final analysis. Validate at minimum:

- campaign state is `complete`;
- 21 scheduled condition IDs each have a successful retrieved attempt, with three marked repeats and no untracked extra jobs;
- 18 prepared primary conditions exist;
- all primary condition/mode combinations use the same saved inputs and 512 shots where applicable;
- `run_metrics.csv` has 75 rows: 54 simulator/exact rows, 18 primary hardware rows, and three hardware-repeat rows;
- `predictions.csv` has 6,000 rows;
- every reported job ID, QPU usage value, circuit resource, kernel diagnostic, confusion matrix, and statistical result is traceable to a saved artifact.

Generate and inspect all publication outputs under `hardware_results/publication_3split_80x80_final_v3/analysis/`, including:

- `publication_results.txt`, a readable consolidated protocol, result, uncertainty, job, and resource summary;
- `run_metrics.csv` and `predictions.csv`;
- 10,000-replicate intervals that resample within class and split and average the three split metrics;
- exact McNemar contrasts with Holm adjustment;
- per-split and summarized native/PCA difference-in-differences;
- captioned LaTeX result, inference, measured-circuit-resource, and larger-dataset scaling tables under `analysis/tables/` (retain CSV copies as machine-readable audit data);
- the faceted noise-ladder balanced-accuracy line graph in PNG, PDF, and SVG;
- kernel-distortion and circuit-cost/degradation figures made directly with Matplotlib in PNG, PDF, and SVG, plus succinct paper-ready captions and labels in `figure_captions.tex`.

Visually inspect the generated PNG figures for clipped labels, missing series, misleading axes, or unreadable legends. Correct plotting defects without changing any measurements. Re-run analysis if needed. The primary figure must retain the ordered exact → ideal Aer → noisy Aer → hardware ladder, thin paired split trajectories, and summarized native/PCA lines where applicable. Treat the scaling table as an extrapolation, not a claim that large quantum kernels are currently practical.

Finally, give me a concise evidence-backed handoff with the primary findings, the native/PCA hardware-minus-exact difference-in-differences, uncertainty, whether kernel distortion tracked CZ count, total jobs and QPU time, failed/retried jobs, and links to the consolidated text, primary figure, detailed CSVs, resource table, and campaign manifest. Explicitly state that this is a controlled small-sample NISQ hardware study and does not demonstrate quantum advantage. Do not commit, push, open a PR, or upload data unless I separately ask.
