# Balanced revision summary

The authoritative baseline for this revision is the original LaTeX supplied by
the author in the conversation. The new manuscript is
`formatting-instructions-latex-2024-balanced-revision.tex`.

The title, authors, affiliations, overall section order, four-environment study,
tables, figures, and main narrative were retained. The balanced version restores
more of the original phrasing than the minimal revision while keeping the changes
needed to answer the reviews.

## Preamble

Before: `xcolor`, the template-only `bibentry` block, and a manual `\newpage`
were present.

After: those items are removed because there is no colored drafting text,
`bibentry` is explicitly marked for removal, and the AAAI template prohibits
manual page breaks. All required AAAI formatting commands remain unchanged.

## Abstract

Before:

```latex
Feature compression using principal component analysis (PCA) mitigates the distortion without appreciably harming accuracy on the IBM hardware.
```

After:

```latex
Feature compression using principal component analysis (PCA) substantially reduced hardware-induced kernel distortion, although its effect on predictive accuracy was dataset dependent.
```

The hardware claim is also limited to empirical stability under the tested
conditions and is explicitly distinguished from formal equivalence.

## Introduction

The original six-paragraph structure and most of its phrasing are retained.
Grammar was corrected, the four quantum environments were named more precisely,
and this sentence was added to explain the exact/ideal distinction:

```latex
Including both exact statevector evaluation and noiseless, shot-based Aer allows us to distinguish finite-shot effects from modeled and measured device noise.
```

The speed and energy questions remain in the motivation, followed by a sentence
stating that neither was directly measured.

## Background

The citations, paragraph order, and claims are unchanged. Edits are limited to
grammar, terminology, and agreement, such as:

```latex
Others work begins the work...
```

becoming

```latex
Other work compared quantum models...
```

## Methodology

Before:

```latex
malicious malware data (indicated by a +1; benign is 0)
```

After:

```latex
malicious samples labeled 1 and benign samples labeled 0
```

The dataset table retains only dataset names and evaluated configurations;
source-row and deduplication counts were removed.

The preprocessing paragraph now states that imputation, scaling, and PCA were
fitted on each training partition and then applied to its test partition. It
also identifies the design as three repeated holdouts rather than independent
replicates.

The original Hilbert-space explanation is retained in a more precise form:

```latex
We map $n$ classical input features to quantum states in this Hilbert space, using one qubit per input feature.
```

This replaces “an exponentially larger quantum feature space,” which could be
read as implying an automatic computational advantage.

Before:

```latex
corrections were applied to enforce symmetry and positive semidefiniteness
```

After: the text gives the actual symmetrization, eigenvalue clipping,
reconstruction, and diagonal normalization procedure. It also defines

```latex
E_F=\frac{\left\|K_m-K_e\right\|_F}{\left\|K_e\right\|_F}.
```

The noisy-Aer paragraph identifies a backend-derived noise model from the saved
calibration immediately preceding the hardware campaign, along with the
backend coupling map and native basis gates. The circuit-table caption now calls
duration an estimated single-circuit duration rather than QPU execution time.

A short statistical subsection states the classifier settings, bootstrap
pairing, exact McNemar test, Holm-corrected hypothesis family, significance
threshold, and the absence of a predefined equivalence margin.

## Results

Before, separate exact and hardware intervals were reported without uncertainty
for their difference. After, Table 2 adds a paired 95% interval to each
hardware-minus-exact delta. The prose summarizes the result as “within two
percentage points,” leaving exact values in the table.

The CZ statement is now explicitly descriptive because CZ count had only three
levels and co-varied with circuit width, depth, and size. The PCA result is
described as a dataset-dependent tradeoff rather than a uniformly harmless
compression method.

The exact shot total is rounded in prose, and the unsupported workload
extrapolation is omitted. Aggregate IBM-reported QPU usage is distinguished from
the single-circuit duration in the resource table.

## Discussion

The original four-paragraph progression is retained: hardware stability, kernel
distortion, PCA, and classical comparison. The wording now avoids interpreting
nonsignificance as equivalence or CZ association as causation.

The prior long limitations block is replaced by one unheaded, three-sentence
paragraph covering sample size and repeated holdouts, limited hardware and model
settings, confounded resource measures, untuned classical models, and missing
time/energy measurements.

## Conclusion

The original three-paragraph conclusion structure is retained.

Before:

```latex
The deviance caused by device noise was only {\color{red}a few points.}
```

After:

```latex
The deviation caused by device noise was small in downstream balanced accuracy compared with the numerical distortion of the kernels under the tested conditions.
```

The repeated feature-count examples and red drafting markers are removed. The
PCA paragraph now acknowledges the substantial CLaMP loss and smaller MBR loss
instead of claiming compression was harmless or inherently robust.

## Future Work

The unfinished text is replaced with the author-provided directions: larger and
more varied experiments, model tuning, a QSVM/QNN comparison, and quantum-HPC
parallelization of quantum-kernel evaluation.

## Acknowledgments and bibliography

The acknowledgment sentence was corrected grammatically. The supplementary
bibliography is included because it supplies the added Qiskit, dataset,
McNemar, and Holm references.
