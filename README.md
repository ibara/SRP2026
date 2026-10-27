# Quantum ML for malware classification


1. Clone or download this repo. In its root folder, we recommend creating a Python environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   # Windows PowerShell: .venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```
   This installs Qiskit, Aer, Jupyter and the data-processing packages.
2. To reproduce on our datasets, download [CLaMP](https://github.com/urwithajit9/ClaMP/blob/master/dataset/ClaMP_Raw-5184.csv)
   into `Malware_Datasets/ClaMP_Raw-5184.csv`, or follow the
   [dataset instructions](Malware_Datasets/README.md) for MBR or your own CSV.
3. Open the notebook:
   ```bash
   python -m notebook QML_Malware_Classification.ipynb
   ```
   Choose **Run All**. Defaults are the CLaMP dataset, **4 qubits**, 512 shots, and 80 training / 80 test rows.
   Change `DATASET` or `N_QUBITS` in the settings cell, then restart and run all cells.

- [Malware_Classification.ipynb](Malware_Classification.ipynb): the complete local pipeline; no IBM account needed.
- [Malware_Datasets/](Malware_Datasets/README.md): download locations and dataset notes; data files stay local.
- [results/](results/README.md): the original README's core metrics and new notebook metrics.
- [Student_Work/](Student_Work/README.md): original notebooks, scripts, detailed hardware results and manuscript work.

The notebook runs one Aer split. Our hardware results average three splits.

If you're interested in running this on real hardware yourself, see [IBM Quantum's getting-started guide](https://quantum.cloud.ibm.com/docs/en/guides/hello-world).
