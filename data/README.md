# Dataset manifest

I keep downloaded data under this directory and intentionally exclude it from
Git. Portable experiment copies and runners use the repository-relative paths
below; the contributor notebooks at their original filenames remain unchanged.

| Dataset | Source | Local target | Status | Used by |
|---|---|---|---|---|
| EMBER 2018 v2 flattened features | [Kaggle](https://www.kaggle.com/datasets/dhoogla/ember-2018-v2-features) | `ember2018/train_ember_2018_v2_features.parquet` | Downloaded and validated | `Ember 2018 Quantum.ipynb`, `Intro to Dataset 1(classical) (1).ipynb` |
| CIC-YNU-IoTMal 2026 SAR features | [UNB](https://www.unb.ca/cic/datasets/ynu-iot-2026.html) | `cic_ynu/{arm,mips,mipsel,x86}_sar.parquet` | Available locally and validated | `QSVM big dataset-Copy1.ipynb`, `Big Dataset Classical.ipynb` |
| EMBER2024 | [GitHub](https://github.com/FutureComputing4AI/EMBER2024) | `ember2024/*.jsonl` | Subset selection required | `QuantumSimEMBER2024.ipynb`, `classicalEMBER.ipynb` |
| Malware/benign recognition | [Kaggle](https://www.kaggle.com/datasets/aashishzzz/malware-detection-dataset-with-new-malware-info) | `fabiana/Malware_and_benign_recognition.csv` | Downloaded and validated | `MachineLearning.ipynb`, `Classical.ipynb`, `Fabiana_DataAnalysis.ipynb` |

The four CIC SAR files supplied for this project are available locally.
Obtaining them independently requires the source's registration form.
EMBER2024 is tens of GB when downloaded in full, so I need to reproduce the
contributor's exact file manifest or declare an explicit architecture/split
subset before downloading it.

## Local validation

I validated the existing local files on 2026-08-22 and the supplied CIC files
on 2026-08-24:

| File | Shape | SHA-256 |
|---|---:|---|
| `ember2018/train_ember_2018_v2_features.parquet` | 799,912 rows x 2,382 columns | `75a69e70a10a7de0902b0c4b96f0959ff6d81b6c468bf9e8cff22e994699c393` |
| `fabiana/Malware_and_benign_recognition.csv` | 1,156 rows x 27 columns | `a0923ac3953115d70937c094304469ef15f169958fb092e6265abe10379a5a8b` |
| `cic_ynu/arm_sar.parquet` | 645,518 rows x 463 columns | `20d42b2785af2442ddddd2d997d2b65e58e9f84c73735429a04217b6c69a532f` |
| `cic_ynu/mips_sar.parquet` | 430,540 rows x 394 columns | `497c1d7665db48f74b178a072c4ee0613e004cb3683ba7c770357dfc9fb18281` |
| `cic_ynu/mipsel_sar.parquet` | 516,679 rows x 394 columns | `bd6eafebd08f5422df901169570d29b6fbec209e6b0f710368cfe797758aa67b` |
| `cic_ynu/x86_sar.parquet` | 529,212 rows x 411 columns | `30de5c903d154f509e3732900704fbc1c37748bed3e6f99f55114d3b03733540` |

## Kaggle download

The public downloads worked without authentication. If Kaggle later requests
authentication, I can run `kaggle auth login`. From the repository root:

```bash
mkdir -p data/ember2018
kaggle datasets download dhoogla/ember-2018-v2-features \
  --file train_ember_2018_v2_features.parquet \
  -p data/ember2018 --unzip

mkdir -p data/fabiana
kaggle datasets download aashishzzz/malware-detection-dataset-with-new-malware-info \
  --file Malware_and_benign_recognition.csv \
  -p data/fabiana --unzip
```

If the Kaggle client leaves the EMBER file as `.parquet.zip`, I extract it with
`python -m zipfile -e <zip-file> data/ember2018` and delete the ZIP only after
validating the parquet.

For the paper, I will record the dataset version, license, download date,
checksum, and all preprocessing or sampling.
