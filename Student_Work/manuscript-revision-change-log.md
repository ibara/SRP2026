# Manuscript revision change log

This log explains the changes from the supplied draft to
`formatting-instructions-latex-2024-revised.tex`. The exhaustive, line-by-line
before/after record is `manuscript-revision-before-after.diff`: lines beginning
with `-` are the workspace draft and lines beginning with `+` are the revised
manuscript.

The LaTeX pasted in the request differs slightly from the copy already present
in the workspace. The pasted version includes `xcolor`, a `+1`/`0` label
sentence, red conclusion placeholders, an unfinished Future Work sentence, and
`\newpage`; the workspace copy had already removed some of those items. The
section notes below explicitly cover those pasted-draft passages as well as the
exact workspace diff.

## Preamble and title

Location: original lines 1--151; revised lines 1--40.

Before:

```latex
\title{A Quantum-Classical Machine Learning Comparison for Malware Detection}
```

After:

```latex
\title{Hardware-Measured Quantum Kernels for Malware Detection: Distortion and Classification Stability}
```

Reason: the new title foregrounds the paper's actual contribution rather than
suggesting that its main purpose is a broad search for quantum advantage.

The unused algorithm/listing example packages and the template-only
`bibentry` package were removed. The prompt's `xcolor` package was also omitted
because the red drafting annotations were resolved. `amsmath` was added for
the displayed execution ladder and equations, while `booktabs` was retained
for the tables. All required AAAI package and page settings, and all author and
affiliation text, were retained. The manual `\newpage` from the pasted draft
was removed because the AAAI template forbids manual page breaks.

## Abstract

Location: original lines 152--154; revised lines 42--44.

Before (robustness claim):

```latex
While we find that quantum advantage is not demonstrated, we believe that hardware robustness is: though the IBM hardware distorts kernels, especially wider native encodings, downstream classification remains stable.
```

After:

```latex
Paired confidence intervals included zero, and no hardware-versus-exact prediction comparison was significant after Holm adjustment. Because no practical-equivalence margin was prespecified, these results show empirical classification stability under the tested conditions, not formal equivalence.
```

Before (PCA claim):

```latex
Feature compression using principal component analysis (PCA) mitigates the distortion without appreciably harming accuracy on the IBM hardware.
```

After:

```latex
Principal component analysis (PCA) reduced circuit cost and hardware distortion but caused dataset-dependent accuracy losses, including 10.0 points for CLaMP.
```

Also added to the abstract were the full four-environment design, the reason for
including both exact and ideal Aer, the principal kernel-error values, the
hardware-minus-exact balanced-accuracy range, and the finding that matched
classical models remained competitive. The generic opening and grammatical
errors were rewritten without adding a quantum-advantage claim.

## Introduction

Location: original lines 156--168; revised lines 46--63.

Before:

```latex
Malicious binaries, more commonly called malware, is one such threat...
```

After:

```latex
Malicious binaries, more commonly called malware, are a persistent threat.
```

Before:

```latex
Using IBM's definition of quantum advantage, we sought to determine if quantum was at least one of more accurate, faster, or more energy efficient...
```

After:

```latex
We do not claim that hardware and exact execution are statistically equivalent, that quantum models outperform optimized classical models, or that the experiment demonstrates a speed or energy advantage.
```

New text explicitly introduces the execution ladder

```latex
\text{exact statevector}
\rightarrow \text{ideal shot-based Aer}
\rightarrow \text{backend-noise Aer}
\rightarrow \text{IBM hardware}
```

and explains that exact-to-ideal comparison isolates finite-shot estimation of
the transpiled circuit. The research questions and two contributions across
six dataset--representation configurations were made explicit. The classical
comparison is now framed as context from matched, untuned baselines, and
“robustness” is narrowed to observed stability in the tested workflow.

## Background

Location: original lines 170--178; revised lines 65--73.

Before:

```latex
Previous work compares classical, hybrid, and purely quantum models...
Others work begins the work of comparing quantum models...
A recurring theme across many these malware studies...
```

After:

```latex
Within malware detection, prior work has compared classical, hybrid, and quantum models for Android malware.
Other work compared a quantum multilayer perceptron with a quantum convolutional neural network...
Several malware studies either do not identify a real-hardware execution environment ... or explicitly report simulator-only experiments...
```

The literature claims and citations were retained but rewritten for grammar and
more cautious scope. “Aid in DDoS attacks” was clarified as DDoS traffic
analysis. A short explanation of the quantum-kernel/SVC division of labor was
added to connect the background directly to the experimental design.

## Methodology: study design and datasets

Location: original lines 180--208; revised lines 75--106.

Before in the pasted draft:

```latex
For our tests, we choose to employ the support vector machine (SVM) machine learning model as a binary classification method for identifying malicious malware data (indicated by a +1; benign is 0).
```

After:

```latex
Labels 0 and 1 denote benign and malware, respectively.
```

This corrects the project-wide convention: benign is `0`, malware is `1`, and
EMBER2018 raw label `-1` is excluded as unlabeled.

The dataset table was expanded from source/selected-row counts to include the
observation type, predictor count, candidate-pool meaning, and exact native/PCA
configurations. New dataset paragraphs record:

- all 10 selected CLaMP native fields and all 8 selected MBR native fields;
- the class counts after CLaMP and MBR deduplication;
- CIC's 314 shared fields, architecture-balanced sampling, and label rule;
- EMBER2018's 2,381 features, exclusion of raw label `-1`, and 600/600 sample;
- duplicate-vector hashing, conflict handling, and retained provenance fields.

These details come from the frozen campaign and prepared-source-pool metadata.

## Methodology: repeated holdouts and preprocessing

Location: original lines 210--212; revised lines 108--112.

Before:

```latex
For each dataset, principal component analysis (PCA) was performed to reduce dimensionality ... thus, we PCA-select data with 10, 8, and 4 features.
...
Duplicate rows were removed, and any invalid values were replaced with median imputation. Inputs were scaled and standardized.
```

After:

```latex
All preprocessing transformations were fitted only on the corresponding 80-row training partition and then applied to its test partition.
```

The replacement states the exact order: training-fitted median imputation;
native min--max scaling; or, for PCA, training-fitted standardization,
full-SVD four-component PCA, and min--max scaling, followed by clipping of test
coordinates. It corrects the earlier implication that the 8- and 10-feature
representations were PCA outputs: those are selected native representations,
whereas every PCA condition has four components.

The new text also records the master and training seeds, 80/80 class-balanced
partitions, globally reserved and mutually disjoint test rows, possible overlap
among training sets, CIC architecture stratification, matched source IDs for
native/PCA comparisons, explained-variance fractions, and finite/overlap
validation. It explicitly calls the design “repeated holdouts” rather than
independent replicates.

## Methodology: quantum kernel and execution environments

Location: original lines 214--260; revised lines 114--151.

Before:

```latex
By using a quantum kernel, we can map n classical input features into an exponentially larger quantum feature space.
```

After:

```latex
Here, each of d input coordinates was assigned to one qubit, mapping the classical input to a quantum state in a 2^d-dimensional Hilbert space; this dimensionality alone does not imply a computational advantage.
```

Before:

```latex
We tested four environments: (1) exact statevector evaluation, (2) noiseless Qiskit Aer simulation, (3) noisy Aer simulation, and (4) ibm_rensselaer hardware execution for 512 shots each (excluding exact evaluation).
```

After: the same four modes are retained, but the revised text specifies that
ideal/noisy/hardware use 512 shots, explains the role of each stage, and states
that the three shot-based modes use the same fixed transpiled instruction-set
circuit.

New reproducibility details include:

- optimization level 3 and transpiler seed 42;
- backend coupling constraints and native `rz`, `sx`, `x`, and `cz` operations;
- the exact logical-to-physical layouts at 4, 8, and 10 qubits;
- `NoiseModel.from_backend`, the August 26, 2026 serialized snapshot, and
  active-wire gate/readout channels;
- omission of the custom delay-relaxation pass because the frozen circuits
  contained no delay instructions;
- dense active-wire relabeling for Aer;
- disabled mitigation, dynamical decoupling, and twirling;
- 3,160 training plus 6,400 test kernel evaluations per condition and the
  resulting 4,894,720 shots.

Before (Table 2 caption):

```latex
Duration is the estimated QPU execution time based on transpilation.
```

After:

```latex
Duration is the estimated duration of one transpiled fidelity circuit based on backend instruction timings, not total job or QPU usage.
```

## Methodology: PSD correction and error definition

Location: original lines 235--237; revised lines 153--164.

Before:

```latex
corrections were applied to enforce symmetry and positive semidefiniteness
```

After:

```latex
Finite-shot training kernels were first symmetrized as K_s=(K+K^T)/2 and assigned a unit diagonal. We then eigendecomposed K_s, clipped every negative eigenvalue to zero, reconstructed the matrix, and normalized it to a unit diagonal.
```

The revision also states that exact training kernels received the same
conditioning and rectangular test kernels were not projected. It identifies
the precomputed SVC and fixed `C=1`, adds the matched linear/RBF settings, and
states that no hyperparameter was chosen from test or hardware results.

Before:

```latex
Potential resulting kernel distortion from this process was measured with Frobenius error and Pearson correlation.
```

After:

```latex
E_F(m)=\frac{\|K_m-K_e\|_F}{\|K_e\|_F}.
```

The surrounding text defines `K_e`, says that Pearson correlation uses
flattened test-kernel entries, and confirms that the unprojected rectangular
test kernels are compared.

## Methodology: statistical analysis

Location: no dedicated original subsection; revised lines 166--172.

Added:

```latex
We used two-sided exact McNemar tests on predictions paired by held-out source identifier. The prespecified family contained six hardware-versus-exact contrasts and two hardware native-versus-PCA contrasts, each based on 240 held-out predictions. Holm adjustment controlled the familywise error rate across all eight tests at alpha=0.05.
```

The new subsection also defines the 10,000-replicate within-split/class
bootstrap, the paired resampling used for hardware-minus-exact differences, the
fixed bootstrap seed, and the conditional interpretation required by overlapping
training samples. It explicitly states that no practical-equivalence margin was
prespecified and that the analysis is not an equivalence test. Exact software
versions are included.

## Results: predictive performance

Location: original lines 262--286; revised lines 174--206.

Before: Table 1 reported separate exact/hardware intervals and a point estimate
for hardware-minus-exact balanced accuracy.

After: the same table retains those results and adds paired 95% intervals for
each difference:

```text
CLaMP native-10:  +0.4 [-2.1, +2.9] percentage points
CLaMP PCA-4:      -0.4 [-3.8, +2.9]
MBR native-8:     -1.7 [-3.8,  0.0]
MBR PCA-4:        -0.8 [-2.1,  0.0]
CIC PCA-4:        +1.3 [-0.4, +3.3]
EMBER2018 PCA-4:  -0.4 [-2.5, +1.7]
```

The text now reports that all paired intervals include zero, all six
hardware-versus-exact Holm-adjusted McNemar values equal 1.0, and the models
disagree on 2--19 of 240 predictions. It immediately states why this is
evidence of small observed changes but not proof of equivalence. The complete
four-stage performance range is also reported.

## Results: kernel distortion, circuit resources, and PCA

Location: original lines 288--298; revised lines 208--221.

Before:

```latex
Across the 18 primary hardware runs, CZ count correlated with relative kernel error (r=0.857, Spearman rho=0.812)...
```

After:

```latex
These values do not isolate a CZ-gate effect. CZ count assumed only three values (22, 54, and 70), the three splits were clustered within configurations, and qubit count, depth, total circuit size, and CZ count increased together.
```

The revised interpretation is an association between overall circuit resource
requirements and kernel error, not a causal CZ-gate effect. The exact-to-ideal,
noisy-Aer, and hardware error ranges are now given so the four-environment
design is visible in the Results rather than only in the figure.

The PCA result remains, but it is consistently called a dataset-dependent
resource--prediction tradeoff. CLaMP's 10.0-point loss and adjusted `p=0.0179`
and MBR's 2.9-point loss and adjusted `p=0.458` are retained.

## Results: classical baselines and hardware workload

Location: original lines 300--310; revised lines 223--234.

The classical comparison is now explicitly descriptive and outside the
prespecified hypothesis family. It records where RBF matched/exceeded hardware
and where the quantum kernel was descriptively higher, while retaining the
fixed-`C` limitation.

Before:

```latex
The 21 hardware conditions required approximately 103 million shots and approximately 28,000 seconds of QPU usage.
```

After:

```latex
The 18 primary hardware conditions and three prespecified repeat conditions comprised 200,760 kernel evaluations and 102,789,120 circuit shots. All 21 hardware jobs completed without failure or resubmission and totaled 28,189 seconds of IBM-reported QPU usage.
```

The next sentence explicitly distinguishes aggregate QPU usage from the
single-circuit duration estimate and from unmeasured wall-clock and energy use.

## Discussion

Location: original lines 312--322; revised lines 236--252.

Before:

```latex
These findings support empirical robustness for the tested samples, feature map, shot count, backend, and calibration period.
```

After:

```latex
We therefore interpret the evidence as observed classification stability for these samples, circuits, and hardware runs, not as proof that noisy and exact quantum kernels are generally equivalent.
```

The revised Discussion separates four interpretations: kernel distortion versus
decision stability, the limits of the uncertainty analysis, confounding in the
circuit-resource correlations, and the PCA resource--representation tradeoff.
It also sharpens the classical-baseline framing and explains why latency and
energy advantage cannot be assessed.

A visible `Limitations` subsection was added. It gathers the small sample size,
three repeated holdouts, balanced rather than operational prevalence, two
native/PCA comparisons, one feature map/repetition count/shot count/backend,
one short hardware campaign, frozen noisy-Aer snapshot, disabled mitigation,
untuned baselines, coupled resource variables, and absent comparable
latency/energy measurements. It also records the post hoc CIC hash finding and
the limits of MBR provenance, full-corpus inference, family shift, and temporal
shift.

## Conclusion and Future Work

Location: original lines 324--328 and the unfinished prompt text; revised lines
254--262.

Before in the pasted draft:

```latex
The deviance caused by device noise was only a few points.
...
However, we showed  An idea that emerged from this research is
```

After:

```latex
The central contribution is instead a reproducible real-hardware observation: substantial numerical kernel distortion need not produce a comparably large change in downstream malware classification.
```

The conclusion now restates the scope, paired uncertainty, PCA tradeoff,
competitive classical results, and absence of demonstrated quantum advantage
without claiming that small circuits are generally robust.

The incomplete Future Work text was replaced with concrete extensions:
group-disjoint and temporal sampling, nested tuning, multiple hardware and
calibration windows, alternate feature maps/shots/mitigation, margin and
eigenspectrum analysis, comparable latency/energy measurement, and a
budget-matched QSVM/QNN comparison.

## Acknowledgments and bibliography

Location: original lines 330--335; revised lines 264--270.

Before:

```latex
We would also like to thank RPI, whose on-campus IBM Quantum System One this research was conducted on.
```

After:

```latex
We also thank Rensselaer Polytechnic Institute for access to its on-campus IBM Quantum System One.
```

The bibliography command now includes `paper_additional_references.bib`, which
contains the McNemar and Holm references used by the new statistical-methods
subsection.
