# VAEBAC — Variational Autoencoder-Based Amyloid Classifier

VAEBAC is a deep learning model for the prediction of amyloidogenic proteins from primary sequence. It combines dual LSTM branches for one-hot and physicochemical feature encoding with a Variational Autoencoder (VAE) to learn a compressed latent representation, which is then used for binary classification.

A live web server is available at **[vaebac.com](https://www.vaebac.com)** for browser-based predictions without any local setup.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Availability](#data-availability)
- [Usage](#usage)
  - [Prediction](#prediction)
  - [Evaluation](#evaluation)
- [Output Format](#output-format)
- [Model Architecture](#model-architecture)
- [Web Server](#web-server)
- [Authors](#authors)
- [License](#license)

---

## Overview

Amyloid proteins are associated with a wide range of neurodegenerative and systemic diseases. VAEBAC predicts the amyloidogenic potential of a protein sequence by jointly learning sequence and physicochemical features through a VAE-based architecture. 
---

---

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.11
- Biopython
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- Seaborn
- SHAP *(evaluation script only)*

---

## Installation

```bash
git clone https://github.com/uzairamu/VAEBAC.git
cd VAEBAC
pip install torch numpy pandas scikit-learn biopython matplotlib seaborn shap
```

---

## Data Availability

The following files are required to run the scripts and are available for download from Zenodo:

| File | Description |
|------|-------------|
| `raw_model_1.pth` | Trained model weights (prediction) |
| `vaebac_final_best_model.pth` | Trained model weights (evaluation) |
| `amino_acid_properties_updated.pkl` | Per-residue physicochemical property dictionary |
| `encoder.pkl` | Fitted one-hot encoder |

**Zenodo DOI:** `https://zenodo.org/records/18872451` 

Download and place all files in the root directory of the repository before running any scripts.

---

## Usage

### Prediction

`predict.py` takes a FASTA file as input and outputs a TSV file with per-sequence predictions and physicochemical descriptors.

```bash
python predict.py --input example_sequences.fasta --output results.tsv
```

**All arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | *(required)* | Path to input FASTA file |
| `--output` | `results.tsv` | Path to output TSV file |
| `--model` | `raw_model_1.pth` | Path to model weights |
| `--props` | `amino_acid_properties_updated.pkl` | Path to amino acid properties file |
| `--encoder` | `encoder.pkl` | Path to one-hot encoder |
| `--threshold` | `0.879` | Decision threshold for classification |

**Example with custom arguments:**

```bash
python predict.py \
    --input my_proteins.fasta \
    --output my_results.tsv \
    --threshold 0.879
```

---

### Evaluation

`vaebac_evaluation.py` evaluates VAEBAC on an independent test set provided as separate positive and negative FASTA files. It produces classification metrics, an ROC curve, a confusion matrix, and a SHAP summary plot.

```bash
python vaebac_evaluation.py
```

By default the script expects the following files in the working directory:

| File | Description |
|------|-------------|
| `positive_dataset.fasta` | Amyloidogenic sequences (label = 1) |
| `negative_dataset.fasta` | Non-amyloidogenic sequences (label = 0) |
| `vaebac_final_best_model.pth` | Model weights |
| `amino_acid_properties_updated.pkl` | Amino acid property dictionary |

Output files generated:

- `VAEBAC_ROC_curve.png` — ROC curve with AUC
- `confusion_matrix_vaebac.png` — Confusion matrix heatmap
- `VAEBAC_SHAP_summary.png` — SHAP feature importance plot

---

## Output Format

`predict.py` produces a tab-separated file with the following columns:

| Column | Description |
|--------|-------------|
| `ID` | Sequence identifier from the FASTA header |
| `Sequence` | Cleaned input sequence |
| `Class` | `Amyloidogenic` or `Non-Amyloidogenic` |
| `Probability` | Sigmoid probability score (0–1) |
| `Hydrophobicity (Avg)` | Mean hydrophobicity across residues |
| `Beta-sheet Propensity (Avg)` | Mean beta-sheet propensity across residues |
| `Solvent Accessibility (Avg)` | Mean solvent accessibility across residues |




For questions or issues, please open a [GitHub Issue](https://github.com/uzairamu/VAEBAC/issues) or contact us at `gh0443@myamu.ac.in`.

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
