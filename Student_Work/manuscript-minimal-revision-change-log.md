# Conservative manuscript revision: change log

The user-provided LaTeX in the conversation is the authoritative original. The
similarly named file already present in the workspace differed from that text,
most notably in its Conclusion. The revised file now follows the user-provided
Conclusion structure. The title, author list, affiliations, section order,
three figures, and central argument were retained.

`manuscript-minimal-revision-before-after.diff` records every changed line
against the on-disk workspace copy. The entries below use the user-provided
original when the two source versions differ.

## Preamble

Before:

```latex
% REMOVE THIS: bibentry
% This is only needed to show inline citations in the guidelines document. You should not need it and can safely delete it.
\usepackage{bibentry}
% END REMOVE bibentry
```

After: removed, as directed by the AAAI template comment. No formatting package
or document-class setting was otherwise changed.

## Abstract

Before:

```latex
While we find that quantum advantage is not demonstrated, we believe that hardware robustness is: though the IBM hardware distorts kernels, especially wider native encodings, downstream classification remains stable . Feature compression using principle component analysis (PCA) mitigates the distortion without appreciably harming accuracy on the IBM hardware.
```

After:

```latex
Although we do not demonstrate quantum advantage, hardware-minus-exact balanced accuracy remained within two percentage points under the tested conditions despite substantial kernel distortion for the wider circuits. This is evidence of empirical stability, not formal statistical equivalence. Feature compression using principal component analysis (PCA) substantially reduced hardware-induced kernel distortion, although its effect on predictive accuracy was dataset dependent.
```

This removes the contradiction with the significant CLaMP PCA accuracy loss and
narrows the hardware claim to the tested conditions. The execution environments
were named consistently, and grammar and the spelling of “principal” were
corrected.

## Introduction

Before:

```latex
Malicious binaries, more commonly called malware, is one such threat...
```

After:

```latex
Malicious binaries, more commonly called malware, are one such threat.
```

The remaining edits correct grammar and clarify the existing contribution. One
methodological justification was added:

```latex
Including both exact statevector evaluation and ideal shot-based Aer separates finite-shot effects from modeled and measured device noise.
```

The broad hardware-robustness claim was changed to “empirical classification
stability under the specific conditions tested,” and the text now states that
speed and energy advantage were not directly evaluated.

## Background

Before:

```latex
Others work begins the work of comparing quantum models for malware classification...
```

After:

```latex
Other work compared quantum models for malware classification...
```

This section received grammar and terminology corrections only. Its paragraph
order, cited studies, and literature-review argument were retained.

## Methodology: classifier and labels

Before:

```latex
For our tests, we choose to employ the support vector machine (SVM) machine learning model as a binary classification method for identifying malicious malware data (indicated by a +1; benign is 0). However, instead of a classical precomputed kernel, we feed the support vector classifier (SVC) a quantum kernel.
```

After:

```latex
For our tests, we use a support vector machine (SVM) for binary classification, with malicious samples labeled 1 and benign samples labeled 0. Rather than a classical kernel, we provide the support vector classifier (SVC) with a precomputed quantum kernel and evaluate its behavior under simulated and real NISQ hardware conditions.
```

This follows the actual project encoding: 1 is malicious and 0 is benign.

## Methodology: datasets

Before:

```latex
Dataset & Source rows & Selected rows & Configuration(s) \\
CLaMP & 5,184 & 4,040 & Native-10, PCA-4 \\
MBR & 1,156 & 895 & Native-8, PCA-4 \\
CIC-YNU-IoTMal & 2,121,949 & 1,280 & PCA-4 \\
EMBER2018 & 799,912 & 1,200 & PCA-4 \\
```

After:

```latex
Dataset & Configuration(s) \\
CLaMP & Native-10, PCA-4 \\
MBR & Native-8, PCA-4 \\
CIC-YNU-IoTMal & PCA-4 \\
EMBER2018 & PCA-4 \\
```

All source-row, selected-row, duplicate-count, class-count, architecture-count,
feature-name, and physical-layout bookkeeping is omitted from the paper. The
table now does only one useful job: map each dataset to its tested
representation. Dataset type is summarized in two sentences below the table.

## Methodology: preprocessing and data splitting

Before:

```latex
Duplicate rows were removed, and any invalid values were replaced with median imputation. Inputs were scaled and standardized.
```

After:

```latex
All preprocessing operations were fitted using the training partition of each split and then applied to its test partition.
```

The revised paragraph then specifies median imputation, native min--max scaling,
and the PCA sequence (standardization, four components, and min--max scaling).
It also identifies the experiments as three repeated holdouts, explains that
test samples did not enter training within a split, and notes that training
samples can overlap across splits. The sample and split counts remain because
they are necessary to interpret the uncertainty analysis.

## Methodology: quantum kernel and execution environments

Before:

```latex
An (n)-qubit system spans a (2^n)-dimensional Hilbert space. By using a quantum kernel, we can map (n) classical input features into an exponentially larger quantum feature space. In particular, we use one qubit per input feature.
```

After:

```latex
We encode each feature with one qubit; thus, for PCA-4, we use four qubits to encode the four selected features.
```

This retains the mapping while avoiding language that could be read as claiming
an automatic exponential computational advantage.

Before:

```latex
corrections were applied to enforce symmetry and positive semidefiniteness
```

After:

```latex
we first applied $K\leftarrow(K+K^T)/2$ and set the diagonal to one. We then eigendecomposed the kernel, clipped negative eigenvalues to zero, reconstructed the matrix, and renormalized its diagonal to one.
```

The relative Frobenius error is now defined explicitly as

```latex
E_F=\frac{\lVert K_m-K_e\rVert_F}{\lVert K_e\rVert_F}.
```

The four-environment paragraph now explains why exact statevector and ideal Aer
are both included and identifies noisy Aer as backend-derived from the saved
hardware-campaign calibration, including gate/readout errors, basis gates, and
coupling map. It also states which mitigation methods were not applied. No
software-version list or physical-qubit layout was added.

## Methodology: circuit-duration caption

Before:

```latex
Duration is the estimated QPU execution time based on transpilation.
```

After:

```latex
Duration is the estimated single-circuit duration based on backend instruction timings.
```

This distinguishes a circuit estimate from aggregate IBM-reported QPU usage.

## Methodology: statistical analysis

Before: statistical tests and the Holm correction appeared in Results without a
methods description.

After: a short `Classification and Statistical Analysis` subsection specifies:

- the precomputed-kernel SVC and fixed, untuned classical baselines;
- balanced accuracy as primary and ROC AUC as secondary;
- class-stratified bootstrap resampling within class and split;
- paired resampling for hardware-minus-exact intervals;
- exact McNemar tests on source-paired predictions;
- one Holm-corrected family containing six hardware/exact and two native/PCA
  contrasts at `alpha = 0.05`; and
- why nonsignificance is not interpreted as equivalence.

## Results

Before:

```latex
Across the six configurations, hardware-minus-exact balanced accuracy ranged from $-1.67$ to $+1.25$ percentage points.
```

After:

```latex
Across the six configurations, hardware-minus-exact balanced accuracy remained within two percentage points.
```

Exact estimates remain in the table instead of being repeated throughout the
prose. The table’s delta column now gives the paired 95% interval requested by
the reviews. Every interval includes zero; the text explicitly says this does
not establish equivalence.

The CZ result changed from an unqualified correlation statement to an
association qualified by its clustered design and three resource levels. The
revised text states that width, depth, circuit size, and CZ count co-vary, so CZ
gates cannot be isolated as the cause.

The PCA paragraph retains the accuracy changes and corrected p-values because
they support the dataset-dependent tradeoff, but removes circuit-resource values
already shown in the circuit table. The figure caption likewise no longer
repeats bootstrap and test-set counts already specified in Methods.

Before:

```latex
The 21 hardware conditions required 102,789,120 shots and 28,189 seconds of IBM-reported QPU usage. Linear extrapolation estimated 58.3--59.0 QPU hours...
```

After:

```latex
The hardware experiments required approximately 103 million shots and 28,000 seconds of IBM-reported aggregate QPU usage.
```

The unmeasured workload projection was removed.

## Discussion

Before:

```latex
These findings support empirical robustness...
```

After:

```latex
These findings show empirical stability for the tested samples, feature map, shot count, backend, and calibration period. They do not establish formal equivalence between exact and hardware execution.
```

The CZ paragraph now explicitly rejects a causal interpretation. Repeated exact
PCA values were replaced by “a substantial loss for CLaMP and a smaller loss for
MBR,” since the values remain in Results.

The long limitations text and separate heading were replaced by one three-sentence
paragraph. It retains the limitations most relevant to interpreting the claims:
small repeated holdouts, one feature-map/hardware setting, two native/PCA
comparisons, confounded circuit resources, untuned classical baselines, and no
comparable wall-clock or energy measurements.

## Conclusion and Future Work

Before, the user-provided Conclusion opened with:

```latex
Ultimately, we have shown through this study the potential of quantum kernels, under NISQ conditions, for use in malware detection. The core principle we evaluate is whether, after applying PCA and performing feature reduction, quantum kernels can preserve learnability.
```

That opening and the original three-paragraph structure are now retained. The
revised first two paragraphs are:

```latex
Ultimately, we have shown through this study the potential of quantum kernels, under NISQ conditions, for use in malware detection. The core principle we evaluated was whether, after applying PCA and performing feature reduction, quantum kernels could preserve learnability. The PCA-reduced representations retained predictive signal across the evaluated datasets, although predictive performance was dataset dependent. The deviation caused by device noise was small in downstream balanced accuracy compared with the numerical distortion of the kernels under the tested conditions.

In malware detection, PCA can substantially reduce the number of features without resulting in a complete loss of learning. However, PCA was not cost-free: it reduced circuit requirements and hardware kernel distortion but caused a substantial predictive loss for CLaMP and a smaller loss for MBR.
```

This removes the repeated feature counts, the red drafting text, and the
unsupported claim that compressed kernels are robust *because* they use small
circuits. The third original summary paragraph remains, with “as circuit
resource requirements increased” replacing the potentially causal CZ wording
and with the conclusion limited to the tested conditions.

The incomplete Future Work text was replaced with the author-provided direction:

```latex
Future evaluations should use larger samples and further vary the tested configurations. Both the classical and quantum models may be improved through model tuning. Another useful comparison would be between this quantum support vector machine (QSVM) approach and a trainable quantum neural network (QNN) to test whether a learned representation improves predictive performance and exhibits the same sensitivity to hardware error.

Another idea that emerged from this study is to use quantum--high-performance computing (quantum-HPC) methods to evaluate quantum-kernel scalability. For example, kernel evaluation could be parallelized across groups of quantum resources based on the size of the feature space. Such methods remained outside the scope of this study.
```

Only small scientific and grammatical adjustments were made: “learning step”
became “kernel evaluation,” and the ambiguous phrase “incurs the same errors”
became “exhibits the same sensitivity to hardware error.”

## Acknowledgments and bibliography

Before:

```latex
We would also like to thank RPI, whose on-campus IBM Quantum System One this research was conducted on.
```

After:

```latex
We also thank RPI, where this research was conducted on its on-campus IBM Quantum System One.
```

The supplementary bibliography file was added to the `\bibliography{}` command
because it contains the Qiskit, dataset, McNemar, and Holm entries used by the
revision. The `\newpage` immediately before Acknowledgments was removed because
the AAAI template explicitly prohibits manual page breaks.
