# Student work and experiment archive

Start with [the main notebook](../Malware_Classification.ipynb) for the supported
local Aer workflow. This folder preserves the earlier work:

- Original contributor notebooks and result notes.
- `hardware_results_experiment.py`, the CIC/EMBER runners, their tests, and `publication_figures.py`.
- `hardware_results/`: the complete frozen campaign, raw outputs, analyses and figures.
- Manuscript drafts, revision notes and LaTeX files.
- [The previous README](README_before_reorganization.md) and [original dataset manifest](dataset_manifest_original.md).

The Python runners' dataset and result paths now resolve relative to their files.
Other archived notebooks and documents retain their historical paths and environments;
they may need path edits before rerunning. Historical result files are preserved unchanged.
`requirements.txt` here is the original dependency list for the archived work;
the repository-root requirements are for the core notebook only.

The main notebook reuses the runner's deduplication, deterministic split selection,
train-fitted preprocessing, ZZ fidelity circuit, batched Aer sampling, kernel
projection and SVM evaluation. It runs one chosen PCA configuration without
requiring the archive or IBM hardware access.
