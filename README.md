# SRP2026

## Scope

I reviewed the group's workflows and first adapted Allison's CLaMP version for the RPI-IBM 127-qubit Quantum System One. I then completed a matched six-configuration hardware campaign studying how feature compression and NISQ hardware noise affect quantum-kernel geometry and downstream malware classification. I treat these runs as evidence about hardware behavior, not quantum advantage.

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
| Sean: EMBER 2018 | 140 train / 60 test; PCA to 4 features; 4 qubits | Reproduced baseline/tuned exact accuracy 0.767/0.767 | 18,130 | Best comparison candidate; tuning improved ROC AUC but not accuracy |
| Allison: EMBER2024 | 200 train / 50 test; 10 features; 10 qubits | Aer accuracy 0.540 | 29,900 | Not hardware-ready as a primary model; malware recall is only 0.04 |
| Sean: CIC-YNU-IoT | 140 train / 60 test; PCA to 4 features; 4 qubits | Repaired baseline/tuned exact accuracy 0.567/0.633 | 18,130 | Training-only tuning improved the result, but it remains a small single-split experiment |
| Fabiana: malware recognition | 200 train / 100 test; 8 native features; 8 qubits | Saved/exact-reproduced accuracy 0.980 | 39,900 | Strong simulator score, but too expensive as a first full hardware run and 12/100 test vectors overlap training in the selected representation |

### Interpretation

- CLaMP is not clearly superior to EMBER 2018. The approximate Wilson 95% accuracy intervals overlap: 0.779–0.915 for CLaMP Aer and 0.646–0.856 for EMBER 2018. The 10/10 CLaMP hardware result has a much wider interval of approximately 0.722–1.000.
- Sean's EMBER 2018 pipeline is the strongest hardware control because PCA reduces it to four qubits while keeping a balanced benign/malware task. A bounded-memory rerun reproduced his 0.767 accuracy and `[[21,9],[5,25]]` confusion matrix exactly; its decision-score ROC AUC is 0.741.
- A fresh matched transpilation on `ibm_rensselaer` produced depth 98 and 22 CZ gates for the four-qubit fidelity circuit, versus depth 206 and 70 CZ gates for the ten-qubit circuit. This would give a concrete circuit-cost comparison.
- Allison's EMBER2024 model currently predicts nearly every sample as benign. I would redesign or compress it before spending QPU time.
- Sean's original CIC-YNU-IoT notebook fitted a second preprocessing pipeline before constructing the test kernel, so its saved 0.733 accuracy was invalid. I repaired the notebook to select the 334 columns shared by all four files, retain the 314 numeric features, assign benign/malware labels, and fit median imputation, standard scaling, PCA, and range scaling on training data only. The corrected 140/60 exact-kernel run scored 0.567 accuracy, 0.567 balanced accuracy, and 0.663 ROC AUC. An additional 160/40 run matching the written 80/20 split scored 0.550, 0.550, and 0.603, respectively.
- I tuned the CIC kernel with five-fold stratified CV on the 140 outer-training rows, without consulting the 60 outer-test rows during selection. The grid covered input ranges `[−1,1]` and `[0,1]`, one or two `ZZFeatureMap` repetitions, linear or full entanglement, and `C` values 0.1, 1, 10, and 100. It selected Allison-style `[0,1]` scaling, one repetition, linear entanglement, and `C=100`, with CV balanced accuracy 0.793 ± 0.076. On the outer test it reached 0.633 accuracy/balanced accuracy and 0.764 ROC AUC, versus 0.567/0.567/0.663 for the repaired baseline.
- The identical training-only sweep on EMBER2018 selected `[0,1]` scaling, one repetition, full entanglement, and `C=10`, with CV balanced accuracy 0.829 ± 0.099. Outer-test accuracy remained 0.767, while ROC AUC increased from 0.741 to 0.777. Both datasets therefore selected Allison's `[0,1]` range and a shallower one-repetition map, but their preferred entanglement and SVC regularization differed.
- Fabiana's newly received notebook already downsamples to 200 training and 100 test rows. Its full shot-based Aer kernel exceeded 1,800 seconds locally and peaked at 5.87 GB, but the same eight-qubit feature map with an exact statevector fidelity kernel completed in 2:44 and reproduced the saved 0.980 accuracy and `[[47,0],[2,51]]` confusion matrix; corrected decision-score ROC AUC is 0.9908. This remains a prototype result: 12/100 test vectors exactly match a training vector across the selected eight features, and a matched RBF SVC already reaches 0.910 accuracy and 0.973 ROC AUC.

### Matched PCA-4 tuning results

| Dataset | Exact configuration | Accuracy | Balanced accuracy | ROC AUC |
|---|---|---:|---:|---:|
| CIC-YNU-IoT | Baseline: `[−1,1]`, 2 reps, linear, `C=1` | 0.567 | 0.567 | 0.663 |
| CIC-YNU-IoT | CV-selected: `[0,1]`, 1 rep, linear, `C=100` | 0.633 | 0.633 | 0.764 |
| EMBER2018 | Baseline: `[−1,1]`, 2 reps, linear, `C=1` | 0.767 | 0.767 | 0.741 |
| EMBER2018 | CV-selected: `[0,1]`, 1 rep, full, `C=10` | 0.767 | 0.767 | 0.777 |

## Dataset setup

I preserved the contributor notebooks at their committed filenames and put portability/correction work in separately named notebooks and runners. Those reproducible copies use repository-relative paths documented in [`data/README.md`](data/README.md). The EMBER 2018, CIC-YNU-IoT SAR, and Fabiana files are available and validated locally; Git ignores the data files. I still need to declare an EMBER2024 architecture/split subset before downloading that much larger corpus.

## Publication experiment: frozen protocol

### Completed campaign results

I completed all 54 planned simulation-mode conditions and retrieved all 21 IBM hardware jobs with no failed or retried attempt. The hardware campaign used 28,189 QPU seconds (469.8 minutes) on `ibm_rensselaer`; every primary estimate below averages three held-out 80/80 splits.

| Configuration | Exact balanced accuracy | Ideal Aer | Noisy Aer | Hardware |
|---|---:|---:|---:|---:|
| CLaMP native-10 | 0.842 | 0.842 | 0.850 | 0.846 |
| CLaMP PCA-4 | 0.750 | 0.742 | 0.754 | 0.746 |
| Fabiana native-8 | 0.975 | 0.975 | 0.975 | 0.958 |
| Fabiana PCA-4 | 0.938 | 0.933 | 0.938 | 0.929 |
| CIC PCA-4 | 0.692 | 0.704 | 0.696 | 0.704 |
| EMBER2018 PCA-4 | 0.617 | 0.625 | 0.625 | 0.613 |

My main findings are:

- Hardware-minus-exact balanced accuracy ranged from -0.0167 to +0.0125 across configurations. None of the six paired hardware-versus-exact McNemar contrasts was significant after Holm adjustment.
- Hardware kernel distortion increased with circuit width: mean test relative error was approximately 0.112 for the four-qubit configurations, 0.441 for Fabiana native-8, and 0.479 for CLaMP native-10. CZ count correlated with kernel distortion (`r=0.857`) but not with classification degradation (`r=-0.032`); these are descriptive, confounded associations across only 18 primary runs.
- CLaMP native-10 outperformed CLaMP PCA-4 on paired hardware predictions (Holm-adjusted `p=0.0179`). The three-split compression effect remains imprecise, so I do not interpret this as universal evidence against PCA.
- The complete results do not establish quantum advantage. On these small duplicate-safe splits, hardware largely preserved classifier performance even when the wider circuits' measured kernel matrices moved substantially farther from the exact kernels.

The completed paper assets are:

- [Consolidated result report](hardware_results/publication_3split_80x80_final_v3/analysis/publication_results.txt)
- [Figure 1: execution-noise ladder](hardware_results/publication_3split_80x80_final_v3/analysis/figure1_noise_ladder_balanced_accuracy.png)
- [Figure 2: kernel distortion](hardware_results/publication_3split_80x80_final_v3/analysis/figure2_kernel_distortion.png)
- [Figure 3: circuit cost versus degradation](hardware_results/publication_3split_80x80_final_v3/analysis/figure3_circuit_cost_vs_degradation.png)
- [Succinct LaTeX figure captions](hardware_results/publication_3split_80x80_final_v3/analysis/figure_captions.tex)
- [Combined LaTeX table index](hardware_results/publication_3split_80x80_final_v3/analysis/tables/all_tables.tex) and the nine captioned table fragments in the same directory
- Machine-readable run metrics, predictions, bootstrap intervals, paired contrasts, circuit resources, scaling estimates, and hardware-job traceability in `hardware_results/publication_3split_80x80_final_v3/analysis/`

The primary research question is:

> How do feature compression and NISQ hardware noise affect quantum-kernel geometry and malware-classification performance?

The publication experiment is implemented in `publication_hardware_experiment.py`. It is separate from the contributor notebooks and the completed pilot runner, so those files remain unchanged. The experiment compares six configurations:

| Configuration | Source representation | Quantum representation | Qubits |
|---|---|---|---:|
| `clamp_native10` | Allison's 10 selected CLaMP features | Native features | 10 |
| `clamp_pca4` | The same 10 CLaMP features and samples | Four training-fitted principal components | 4 |
| `fabiana_native8` | Fabiana's eight selected features | Native features | 8 |
| `fabiana_pca4` | The same eight Fabiana features and samples | Four training-fitted principal components | 4 |
| `cic_pca4` | The 314 numeric columns shared by ARM, MIPS, MIPSEL, and x86 SAR files | Four training-fitted principal components | 4 |
| `ember2018_pca4` | The 2,381 numeric EMBER2018 features | Four training-fitted principal components | 4 |

CLaMP and Fabiana are the controlled native-versus-PCA comparisons. Within each split, the native and PCA configurations use exactly the same source rows and labels. CIC tests whether the four-qubit result transfers across CPU architectures, and EMBER2018 supplies a second established malware corpus. The quantum-kernel family is intentionally held fixed; alternate feature-map searches are outside the main study.

### Sampling and preprocessing

- Use three deterministic held-out splits with training seeds 101, 204, and 303. Every condition has 80 training rows and 80 test rows, balanced at 40 benign and 40 malware samples. Seed 204 is the first ascending replacement for seed 202 that passed the transformed-overlap guard across every dataset/representation; seeds 202 and 203 were rejected before any model metric or hardware result was calculated.
- The three test sets within a dataset are mutually disjoint. All 240 reserved test rows are excluded from every training set; training rows may recur between splits.
- Hash the complete selected source representation, collapse same-label duplicates to one deterministic representative, and exclude conflicting-label groups before sampling. Preserve source IDs, row hashes, file checksums, and seeds.
- For CIC, match the benign and malware architecture counts inside every train and test split. Each class contains exactly 10 rows from each of ARM, MIPS, MIPSEL, and x86.
- Fit median imputation on training data only. Native representations then use training-fitted min-max scaling to `[0,1]`. PCA representations use training-fitted median imputation, standardization, PCA-4, and min-max scaling to `[0,1]`. Reuse each fitted transform for its test set and clip test values only at the final range-scaling step.
- Abort if a transformed test vector exactly matches a transformed training vector. Save every prepared matrix and fitted preprocessing parameter so all execution modes consume identical inputs.

### Fixed model and execution ladder

Every configuration uses Qiskit's `zz_feature_map` with two repetitions and linear entanglement, a fidelity kernel, and a precomputed-kernel `SVC(C=1)`. The shot-based modes use 512 shots per fidelity estimate. Error mitigation is disabled so measured changes represent the unmitigated execution environment.

Each of the 18 primary configuration/split conditions follows the same ladder:

1. exact statevector fidelity kernel;
2. 512-shot ideal Aer using the hardware-transpiled circuit;
3. 512-shot Aer using the initial backend noise model;
4. 512-shot execution on `ibm_rensselaer` using the same transpiled circuit and parameter bindings.

Linear and RBF SVCs are also evaluated on exactly the same transformed samples as classical sanity checks. The runner transpiles and records one representative fidelity circuit for each width (4, 8, and 10 qubits), including its physical layout, depth, operation counts, estimated duration, backend calibration snapshot, and software versions.

### Hardware workload and repeats

With 80 training and 80 test rows, one condition requires `80*79/2 + 80*80 = 9,560` fidelity evaluations, or 4,894,720 circuit executions at 512 shots. This is below IBM's documented 10-million-execution job limit. The complete hardware campaign contains:

- 18 primary runs: six configurations across three held-out splits;
- three repeat runs: split 0 of `clamp_pca4`, `fabiana_native8`, and `clamp_native10` is run once more, giving two measurements each at 4, 8, and 10 qubits;
- 21 hardware jobs total, 200,760 fidelity evaluations, and 102,789,120 circuit executions;
- two batches of at most 12 jobs, with the original anchors near the beginning and their repeats at the end to expose coarse time-dependent device drift.

Jobs are submitted sequentially within resumable eight-hour Runtime batches. A condition may be retried once after an error or cancellation; job IDs, attempts, timestamps, QPU usage, and per-attempt backend snapshots are retained. IBM documents both the [10-million-execution job limit](https://quantum.cloud.ibm.com/docs/en/guides/job-limits) and the three-hour maximum execution time for a single QPU job. The runner requests a two-hour per-job maximum, but actual QPU duration and queue behavior must still be monitored.

### Preregistered measurements and analysis

For every mode, report accuracy, balanced accuracy, decision-score ROC AUC, confusion matrix, and class recall. For shot-based kernels, also report train/test MAE, RMSE, Pearson correlation, relative Frobenius error versus exact, prediction disagreement, raw negative eigenvalues, and the explicit PSD projection diagnostics used before SVC fitting.

The primary estimand is paired hardware-minus-exact balanced-accuracy degradation. For CLaMP and Fabiana, also report the native/PCA difference-in-differences:

`(PCA hardware - PCA exact) - (native hardware - native exact)`.

Use all 240 non-overlapping test predictions per configuration for 10,000-replicate bootstrap intervals, resampling within class and split and averaging the three split metrics. This avoids treating decision scores from independently fitted SVCs as one globally calibrated ranking. Use exact paired McNemar tests for hardware versus exact and for native versus PCA hardware predictions, with Holm adjustment across the planned contrasts. Three-split intervals and the three anchor reruns are descriptive uncertainty checks, not evidence of quantum advantage.

### Publication figures and tables

The primary figure is a faceted line graph, not grouped bars. Each dataset panel shows the four ordered environments—exact, ideal Aer, noisy Aer, and hardware—on the x-axis and balanced accuracy on the y-axis. Thin paired fold trajectories expose within-sample changes; thicker means with 95% intervals summarize them. CLaMP and Fabiana panels contain separate native and PCA lines. CIC and EMBER2018 contain their PCA-4 line. A line graph makes the noise ladder and paired degradation visible in a way that three independent bars would hide. All figures are generated directly with Matplotlib in PNG, PDF, and SVG, and `analysis/figure_captions.tex` provides succinct paper-ready captions and labels.

Secondary figures report kernel correlation and relative Frobenius error across the same ladder, then plot transpiled CZ count against hardware-minus-exact balanced accuracy and kernel distortion. Confusion matrices can be supplementary because the pooled metrics and paired predictions retain more information.

A resource table is useful if it is labeled as a scaling estimate rather than evidence of feasibility. The analysis records measured 4-, 8-, and 10-qubit circuit resources and extrapolates kernel evaluations, circuit shots, CZ applications, and observed-time-based QPU duration for 80/80, 100/100, 500/500, 1,000/1,000, and 5,000/1,000 train/test scenarios. Gate depth and per-circuit gate counts do not grow with dataset size; the quadratic kernel workload does. Publication tables are emitted as captioned, labeled LaTeX fragments under `analysis/tables/`; CSV files remain only as machine-readable audit data.

### Running the frozen campaign

Inspect the deterministic workload without touching IBM Runtime:

```bash
.venv/bin/python -m unittest -v test_publication_hardware_experiment.py
.venv/bin/python publication_hardware_experiment.py plan
```

Prepare source pools and run only exact local baselines:

```bash
.venv/bin/python publication_hardware_experiment.py prepare
.venv/bin/python publication_hardware_experiment.py simulate --exact-only
```

The complete resumable simulation and hardware campaign requires an explicit QPU authorization flag:

```bash
.venv/bin/python publication_hardware_experiment.py all --authorize-qpu
```

`submit`, `status`, `wait`, `retrieve`, and `analyze` are also available as separate recovery commands. Importing the module, preparing data, or running tests cannot submit a job. Final-campaign outputs are written under `hardware_results/publication_3split_80x80_final_v3/`; `campaign.json` is the atomic source of truth for resuming. The earlier five-split 100/100 exact-only preparation remains archived under `hardware_results/publication_full_v1/`, and the aborted seed-202 preflight remains under `hardware_results/publication_3split_80x80_v2/`. The unattended execution instructions are in `FULL_HARDWARE_EXPERIMENT_PROMPT.md`.

This design supports a controlled empirical hardware-robustness claim. It does not establish quantum advantage, universal superiority of PCA, or competitive full-corpus malware detection.

## Artifacts

I saved the complete CLaMP run under `hardware_results/clamp_10q_r2_20train_10test/`:

- `submission.json`: job, circuit, and transpilation metadata
- `hardware_summary.json`: metrics, kernel comparisons, and QPU usage
- `baseline_summary.json`: exact, shot-based, and classical baselines
- `hardware_results.npz`: measured kernels, fidelities, scores, and predictions
- `prepared_data.npz`: exact sampled inputs, source indices, labels, and parameters
- `transpiled_fidelity_circuit.qpy`: submitted ISA circuit

The repaired and tuned PCA-4 runs are under `results/`:

- `cic_pca4_exact_seed42/`: Sean's 140/60 quantum-notebook split
- `cic_pca4_exact_80_20_seed42/`: the written 160/40 split
- `cic_pca4_kernel_sweep_seed42/`: training-only kernel search, selected exact kernel, and outer-test evaluation
- `ember2018_pca4_exact_seed42/`: memory-bounded reproduction of Sean's EMBER2018 run
- `ember2018_pca4_kernel_sweep_seed42/`: matched EMBER2018 training-only kernel search and outer-test evaluation
- baseline directories contain `summary.json` and `results.npz`; sweep directories contain `summary.json`, `cv_results.csv`, and `selected_kernel_results.npz`
- `experiment_results.txt`: consolidated plain-text outputs, matched classical checks, validation results, and hardware caveats

The original contributor notebooks remain at `QSVM big dataset-Copy1.ipynb`,
`Ember 2018 Quantum.ipynb`, and `fabiana.ipynb`. The executed portable copies
are `CIC PCA4 Repaired.ipynb`, `EMBER2018 PCA4 Reproducible.ipynb`, and
`Fabiana Quantum Reproducible.ipynb`. Fabiana's portable copy uses an exact
statevector fidelity kernel because her unchanged full shot-based Aer cell did
not complete within 1,800 seconds; all data selection and model dimensions are
otherwise retained, and the substitution is documented inside the notebook.

The memory-bounded dataset runners and generic sweep reproduce these experiments
without loading either full source dataframe:

```bash
.venv/bin/python cic_pca4_runner.py
.venv/bin/python cic_pca4_runner.py --test-size 0.20 \
  --output-dir results/cic_pca4_exact_80_20_seed42
.venv/bin/python pca4_quantum_kernel_sweep.py
.venv/bin/python ember2018_pca4_runner.py
.venv/bin/python pca4_quantum_kernel_sweep.py \
  --prepared-results results/ember2018_pca4_exact_seed42/results.npz \
  --output-dir results/ember2018_pca4_kernel_sweep_seed42 \
  --dataset-name EMBER2018
```

The frozen publication implementation is contained in:

- `publication_hardware_experiment.py`: duplicate-safe preparation, matched local baselines, guarded/resumable Runtime submission and retrieval, statistics, figures, resource tables, and consolidated text output;
- `test_publication_hardware_experiment.py`: synthetic offline protocol, fold, preprocessing, kernel, and QPU-authorization checks;
- `FULL_HARDWARE_EXPERIMENT_PROMPT.md`: the one-shot prompt that explicitly authorizes and monitors the complete 21-run QPU campaign.

The earlier 100/100 preparation and its 30 exact/classical baselines remain under `hardware_results/publication_full_v1/` as an archive and contain no hardware claim. The frozen final campaign uses the separately versioned `hardware_results/publication_3split_80x80_final_v3/`; its final readable result is `analysis/publication_results.txt` alongside detailed CSVs and publication figures.

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
- The four-qubit configurations are intentionally classically simulable; I use them to validate hardware behavior, not to claim classical intractability.
- The supplied CIC files are not identical to Sean's saved-run inputs: the local MIPSEL file has 516,679 rows rather than the saved notebook's 430,540, and the current family labels include `Rudedevil`. The repaired score is therefore a reproducible result for the current files, not a controlled one-variable correction of Sean's old run.
- The tuned CIC kernel fixed five baseline errors while introducing one new error, a net gain of four correct predictions on 60 rows. The paired exact McNemar p-value is 0.219, so this single-split gain is promising but not statistically significant. The outer test had also been evaluated in earlier project work and is not pristine at the project level; confirmation requires new group-aware seeds.
- EMBER tuning fixed three baseline errors and introduced three others, leaving accuracy unchanged. Its ROC AUC gain also needs confirmation on new seeds. Kernel selection did not consult either outer test in the sweep scripts, but both test sets had already appeared in saved project outputs and are not pristine at the project level.
- The kernel sweeps treat each four-component representation, fitted on all 140 outer-training rows, as fixed during inner CV. A publication-grade nested benchmark should refit imputation, scaling, and PCA inside every inner fold before reporting CV uncertainty.
- I still need to select a reproducible EMBER2024 subset; it is deliberately excluded from the frozen publication campaign.
- The frozen experiment handles Fabiana's selected-feature duplicates before splitting and uses the same 80/80 workload as every other configuration. All 21 hardware runs are complete, but the three splits still provide limited uncertainty for dataset-level and compression-effect claims.
- Even after completion, three 80/80 splits remain a controlled small-sample hardware benchmark. The larger-dataset resource table is explicitly an extrapolation, and any later extension to deeper maps or additional datasets should be reported as a separate follow-up rather than silently added to this preregistered comparison.
