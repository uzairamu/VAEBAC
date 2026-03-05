# -*- coding: utf-8 -*-
"""
vaebac_evaluation.py

Evaluation script for VAEBAC (Variational Autoencoder-Based Amyloid Classifier).
Loads a pre-trained VAEBAC model and evaluates it on an independent test set
provided as FASTA files. Outputs classification metrics, an ROC curve, a
confusion matrix, and a SHAP summary plot.

Requirements:
    pip install torch numpy pandas scikit-learn biopython shap matplotlib seaborn

Usage:
    python vaebac_evaluation.py
"""

import pickle

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from Bio import SeqIO
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, matthews_corrcoef
)
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POSITIVE_FASTA  = "positive_dataset.fasta"
NEGATIVE_FASTA  = "negative_dataset.fasta"
PROPERTIES_PKL  = "https://zenodo.org/records/18872116/files/amino_acid_properties_updated.pkl?download=1"
MODEL_PATH      = "https://zenodo.org/api/records/18872116/draft/files/raw_model_1.pth/content"

MAX_SEQ_LEN     = 1000    # sequences are truncated to this length
TOTAL_LEN       = 10746   # padded length used during training
THRESHOLD       = 0.879   # decision threshold determined on the validation set
BATCH_SIZE      = 1

PROTEIN_CODES = np.array([
    'A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
    'X', 'B', 'Z'
])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_fasta_sequences(fasta_path):
    """Return a list of raw sequence strings from a FASTA file."""
    return [str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")]


def clean_sequence(seq, valid_codes, max_len):
    """Replace unknown amino acids with 'X' and truncate to *max_len*."""
    valid = set(valid_codes)
    seq = ''.join(aa if aa in valid else 'X' for aa in seq)
    return seq[:max_len]


def build_onehot_encoder(protein_codes):
    encoder = OneHotEncoder()
    encoder.fit(protein_codes.reshape(-1, 1))
    return encoder


def encode_sequences(sequences, encoder):
    """One-hot encode each sequence into a list of float tensors."""
    embeddings = []
    for seq in sequences:
        arr = np.array(list(seq))
        embed = torch.tensor(encoder.transform(arr.reshape(-1, 1)).toarray())
        embeddings.append(embed)
    return embeddings


def build_physicochemical_features(sequences, properties_dict):
    """Build per-residue physicochemical feature tensors."""
    feats = []
    for seq in sequences:
        feats_protein = [np.array(properties_dict[aa]) for aa in seq]
        feats.append(torch.tensor(feats_protein))
    return feats


def get_sequence_lengths(sequences):
    return [torch.tensor(len(seq), dtype=torch.int64) for seq in sequences]


class ProteinDataset(Dataset):
    def __init__(self, onehot, feats, lengths, labels):
        self.onehot  = onehot
        self.feats   = feats
        self.lengths = lengths
        self.labels  = labels

    def __len__(self):
        return len(self.onehot)

    def __getitem__(self, idx):
        return (
            self.onehot[idx],
            self.feats[idx],
            self.lengths[idx],
            self.labels[idx],
        )


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class VAEBAC(nn.Module):
    """
    Variational Autoencoder-Based Amyloid Classifier (VAEBAC).

    Two LSTM branches process one-hot and physicochemical features
    independently. Their outputs are concatenated, pooled, and passed
    through a VAE encoder to obtain a latent mean vector (mu), which
    is used for binary classification.
    """

    def __init__(self):
        super().__init__()

        # Sequence encoder (one-hot input, 23-dim)
        self.lstm        = nn.LSTM(23, 16, 1)
        self.fc          = nn.Conv1d(16, 12, kernel_size=1)

        # Physicochemical feature encoder (4-dim)
        self.lstm_feats  = nn.LSTM(4, 16, 1)

        # VAE encoder
        self.encode_1    = nn.Linear(24, 28)
        self.mu          = nn.Linear(28, 512)
        self.log_var     = nn.Linear(28, 512)

        # VAE decoder (retained for architectural completeness)
        self.decoding_1  = nn.Sequential(
            nn.Linear(512, 28),
            nn.Linear(28, 24),
        )
        self.decoding_2  = nn.Sequential(
            nn.Conv1d(24, 12, kernel_size=1),
            nn.Conv1d(12, 16, kernel_size=1),
        )
        self.pad_decoder  = nn.Conv1d(1, 800, kernel_size=1)
        self.lstm_decode  = nn.LSTM(16, 23, 1)

        # Classifier head
        self.class_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.flat  = nn.Flatten()
        self.relu  = nn.ReLU()

    def encode(self, x):
        h       = self.encode_1(x)
        mu      = self.mu(h)
        log_var = self.log_var(h)
        return mu, log_var

    def forward(self, x, feats, lengths):
        # One-hot LSTM branch
        packed      = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _      = self.lstm(packed)
        out, _      = pad_packed_sequence(out, batch_first=True, total_length=TOTAL_LEN)
        out_seq     = self.relu(self.fc(out.permute(0, 2, 1)))

        # Physicochemical LSTM branch
        packed_f    = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_f, _    = self.lstm_feats(packed_f)
        out_f, _    = pad_packed_sequence(out_f, batch_first=True, total_length=TOTAL_LEN)
        out_feats   = self.relu(self.fc(out_f.permute(0, 2, 1)))

        # Concatenate, pool, flatten
        combined    = torch.cat((out_seq, out_feats), dim=1)
        pooled      = self.flat(self.pool(combined))

        # VAE encoding → classification
        mu, _       = self.encode(pooled)
        return self.class_layer(mu)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, loader, device):
    """Return sigmoid probabilities and ground-truth labels as numpy arrays."""
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            seq    = batch[0].to(torch.float32).to(device)
            feats  = batch[1].to(torch.float32).to(device)
            length = batch[2].to(torch.float32).to(device)
            label  = batch[3].unsqueeze(1).to(torch.float32).to(device)

            logits = model(seq, feats, length)
            probs  = torch.sigmoid(logits)

            all_probs.extend(probs.detach().cpu().numpy())
            all_labels.extend(label.detach().cpu().numpy())

    return np.array(all_probs), np.array(all_labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, probs, threshold):
    """Compute and print standard binary classification metrics."""
    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sn   = tp / (tp + fn)
    sp   = tn / (tn + fp)
    acc  = (tp + tn) / (tp + tn + fp + fn)
    ba   = (sn + sp) / 2
    mcc  = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn))
    pre  = tp / (tp + fp)
    gmean = np.sqrt(sn * sp)
    f1   = 2 * pre * sn / (pre + sn)
    auc_score = roc_auc_score(labels, probs)

    print("===== VAEBAC Evaluation Metrics =====")
    print(
        f"SN: {sn:.4f}  SP: {sp:.4f}  ACC: {acc:.4f}  BA: {ba:.4f}  "
        f"MCC: {mcc:.4f}  Pre: {pre:.4f}  AUC: {auc_score:.4f}  "
        f"Gmean: {gmean:.4f}  F1: {f1:.4f}"
    )

    return {
        "SN": sn, "SP": sp, "ACC": acc, "BA": ba, "MCC": mcc,
        "Pre": pre, "AUC": auc_score, "Gmean": gmean, "F1": f1,
        "CM": confusion_matrix(labels, preds),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_roc(labels, probs, save_path="VAEBAC_ROC_curve.png"):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"VAEBAC (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve – Independent Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()
    print(f"ROC curve saved to {save_path}")


def plot_confusion_matrix(cm, save_path="confusion_matrix_vaebac.png"):
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-amyloid", "Amyloid"],
        yticklabels=["Non-amyloid", "Amyloid"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix – Independent Test Set")
    plt.tight_layout()
    plt.savefig(save_path, dpi=400)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


def plot_shap_summary(model, latent_vectors, n_background=100, n_explain=200,
                      save_path="VAEBAC_SHAP_summary.png"):
    """Generate a SHAP summary plot for the classifier head."""
    background = torch.tensor(latent_vectors[:n_background], dtype=torch.float32).to(DEVICE)
    samples    = torch.tensor(latent_vectors[:n_explain],    dtype=torch.float32).to(DEVICE)

    explainer   = shap.GradientExplainer(model.class_layer, background)
    shap_values = explainer.shap_values(samples)
    shap_values = np.squeeze(np.array(shap_values), axis=2)

    shap.summary_plot(shap_values, samples.cpu().numpy(), show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"SHAP summary plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Using device: {DEVICE}")

    # -- Load amino acid properties --
    with open(PROPERTIES_PKL, "rb") as f:
        properties_dict = pickle.load(f)

    # -- Build one-hot encoder --
    encoder = build_onehot_encoder(PROTEIN_CODES)

    # -- Load and preprocess sequences --
    positive_seqs = [
        clean_sequence(s, PROTEIN_CODES, MAX_SEQ_LEN)
        for s in load_fasta_sequences(POSITIVE_FASTA)
    ]
    negative_seqs = [
        clean_sequence(s, PROTEIN_CODES, MAX_SEQ_LEN)
        for s in load_fasta_sequences(NEGATIVE_FASTA)
    ]

    sequences = positive_seqs + negative_seqs
    labels    = [1] * len(positive_seqs) + [0] * len(negative_seqs)

    print(f"Total sequences : {len(sequences)}")
    print(f"  Positives     : {len(positive_seqs)}")
    print(f"  Negatives     : {len(negative_seqs)}")

    # -- Build features --
    onehot_embeddings = encode_sequences(sequences, encoder)
    physchem_features = build_physicochemical_features(sequences, properties_dict)
    seq_lengths       = get_sequence_lengths(sequences)

    # -- Build dataset and loader --
    padded_onehot = pad_sequence(onehot_embeddings, batch_first=True)
    padded_feats  = pad_sequence(physchem_features, batch_first=True)
    label_tensor  = torch.tensor(labels, dtype=torch.float32)

    dataset    = ProteinDataset(padded_onehot, padded_feats, seq_lengths, label_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # -- Load model --
    model = VAEBAC().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Freeze VAE encoder parameters (evaluation only)
    for param in list(model.encode_1.parameters()) + \
                 list(model.mu.parameters()) + \
                 list(model.log_var.parameters()):
        param.requires_grad = False

    # -- Inference --
    probs, true_labels = run_inference(model, dataloader, DEVICE)

    # -- Metrics --
    results = compute_metrics(true_labels, probs, THRESHOLD)

    # -- Plots --
    plot_roc(true_labels, probs)
    plot_confusion_matrix(results["CM"])

    # -- SHAP analysis --
    # Extract latent vectors (mu) for SHAP explanation
    latent_vectors = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            seq    = batch[0].to(torch.float32).to(DEVICE)
            feats  = batch[1].to(torch.float32).to(DEVICE)
            length = batch[2].to(torch.float32).to(DEVICE)

            packed   = pack_padded_sequence(seq, length.cpu(), batch_first=True, enforce_sorted=False)
            out, _   = model.lstm(packed)
            out, _   = pad_packed_sequence(out, batch_first=True, total_length=TOTAL_LEN)
            out_seq  = model.relu(model.fc(out.permute(0, 2, 1)))

            packed_f  = pack_padded_sequence(feats, length.cpu(), batch_first=True, enforce_sorted=False)
            out_f, _  = model.lstm_feats(packed_f)
            out_f, _  = pad_packed_sequence(out_f, batch_first=True, total_length=TOTAL_LEN)
            out_feats = model.relu(model.fc(out_f.permute(0, 2, 1)))

            combined  = torch.cat((out_seq, out_feats), dim=1)
            pooled    = model.flat(model.pool(combined))
            mu, _     = model.encode(pooled)
            latent_vectors.append(mu.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)
    plot_shap_summary(model, latent_vectors)


if __name__ == "__main__":
    main()
