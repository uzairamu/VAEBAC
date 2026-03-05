# -*- coding: utf-8 -*-
"""
predict.py

Standalone prediction script for VAEBAC (Variational Autoencoder-Based
Amyloid Classifier). Reads protein sequences from a FASTA file, runs
inference, and writes results to a TSV file.

Requirements:
    pip install torch numpy pandas scikit-learn biopython

Usage:
    python predict.py --input sequences.fasta --output results.tsv
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Bio import SeqIO
from io import StringIO
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_PATH      = "raw_model_1.pth"
PROPERTIES_PKL  = "amino_acid_properties_updated.pkl"
ENCODER_PKL     = "encoder.pkl"
THRESHOLD       = 0.879
BATCH_SIZE      = 8

VALID_AAS = set(['A','R','N','D','C','E','Q','G','H','I','L','K',
                 'M','F','P','S','T','W','Y','V','X','B','Z'])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        self.lstm       = nn.LSTM(23, 16, 1)
        self.fc         = nn.Conv1d(16, 12, kernel_size=1)
        self.lstm_feats = nn.LSTM(4, 16, 1)

        self.encode_1   = nn.Linear(24, 28)
        self.mu         = nn.Linear(28, 512)
        self.log_var    = nn.Linear(28, 512)

        self.decoding_1 = nn.Sequential(
            nn.Linear(512, 28),
            nn.Linear(28, 24),
        )
        self.decoding_2 = nn.Sequential(
            nn.Conv1d(24, 12, kernel_size=1),
            nn.Conv1d(12, 16, kernel_size=1),
        )
        self.pad_decoder = nn.Conv1d(1, 800, kernel_size=1)
        self.lstm_decode = nn.LSTM(16, 23, 1)

        self.class_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flat = nn.Flatten()
        self.relu = nn.ReLU()

    def encode(self, x):
        h = self.encode_1(x)
        return self.mu(h), self.log_var(h)

    def forward(self, x, feats, lengths):
        packed      = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _      = self.lstm(packed)
        out, _      = pad_packed_sequence(out, batch_first=True)
        out_seq     = self.relu(self.fc(out.permute(0, 2, 1)))

        packed_f    = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out_f, _    = self.lstm_feats(packed_f)
        out_f, _    = pad_packed_sequence(out_f, batch_first=True)
        out_feats   = self.relu(self.fc(out_f.permute(0, 2, 1)))

        combined    = torch.cat((out_seq, out_feats), dim=1)
        pooled      = self.flat(self.pool(combined))
        mu, _       = self.encode(pooled)
        return self.class_layer(mu)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_sequences(fasta_path):
    """Parse a FASTA file and return cleaned sequences and their IDs."""
    sequences, ids = [], []
    for record in SeqIO.parse(fasta_path, "fasta"):
        cleaned = ''.join(aa if aa in VALID_AAS else 'X'
                          for aa in str(record.seq).upper())
        sequences.append(cleaned)
        ids.append(str(record.id))
    return sequences, ids


def encode_sequences(sequences, encoder):
    embeddings = []
    for seq in sequences:
        arr   = np.array(list(seq))
        embed = torch.tensor(encoder.transform(arr.reshape(-1, 1)).toarray())
        embeddings.append(embed)
    return embeddings


def build_physicochemical_features(sequences, properties_dict):
    feats = []
    for seq in sequences:
        fp = [np.array(properties_dict.get(aa, properties_dict['A'])) for aa in seq]
        feats.append(torch.tensor(fp))
    return feats


def collate_fn(batch):
    x, f, l = zip(*batch)
    return pad_sequence(x, batch_first=True), pad_sequence(f, batch_first=True), torch.stack(l)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, sequences, encoder, properties_dict, device, batch_size):
    onehot  = encode_sequences(sequences, encoder)
    feats   = build_physicochemical_features(sequences, properties_dict)
    lengths = [torch.tensor(len(s), dtype=torch.int64) for s in sequences]

    dataset = list(zip(onehot, feats, lengths))
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch_x, batch_feats, batch_len in loader:
            batch_x     = batch_x.to(torch.float32).to(device)
            batch_feats = batch_feats.to(torch.float32).to(device)
            batch_len   = batch_len.to(torch.int64).to(device)

            logits = model(batch_x, batch_feats, batch_len)
            probs  = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.extend(probs)

    return [float(p) for p in all_probs]


# ---------------------------------------------------------------------------
# Feature summary
# ---------------------------------------------------------------------------

def compute_physicochemical_summary(sequences, properties_dict):
    """Compute per-sequence average of hydrophobicity, beta-sheet propensity,
    and solvent accessibility."""
    hydro, beta, sasa = [], [], []
    for seq in sequences:
        fp = [properties_dict.get(aa, properties_dict['A']) for aa in seq]
        fp = np.array(fp)
        hydro.append(float(np.mean(fp[:, 0])))
        beta.append(float(np.mean(fp[:, 1])))
        sasa.append(float(np.mean(fp[:, 2])))
    return hydro, beta, sasa


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VAEBAC: predict amyloidogenicity from protein sequences."
    )
    parser.add_argument("--input",  required=True, help="Path to input FASTA file")
    parser.add_argument("--output", default="results.tsv", help="Path to output TSV file")
    parser.add_argument("--model",  default=MODEL_PATH,     help="Path to model weights (.pth)")
    parser.add_argument("--props",  default=PROPERTIES_PKL, help="Path to amino acid properties (.pkl)")
    parser.add_argument("--encoder",default=ENCODER_PKL,    help="Path to one-hot encoder (.pkl)")
    parser.add_argument("--threshold", type=float, default=THRESHOLD,
                        help="Decision threshold for amyloidogenicity (default: 0.879)")
    args = parser.parse_args()

    print(f"Device : {DEVICE}")

    # Load resources
    with open(args.props, "rb") as f:
        properties_dict = pickle.load(f)
    with open(args.encoder, "rb") as f:
        encoder = pickle.load(f)

    model = VAEBAC().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    # Load sequences
    sequences, ids = load_sequences(args.input)
    if not sequences:
        raise ValueError(f"No valid sequences found in {args.input}")
    print(f"Loaded {len(sequences)} sequence(s) from {args.input}")

    # Inference
    probs = run_inference(model, sequences, encoder, properties_dict, DEVICE, BATCH_SIZE)

    # Physicochemical summary
    hydro, beta, sasa = compute_physicochemical_summary(sequences, properties_dict)

    # Build results
    classes = ["Amyloidogenic" if p >= args.threshold else "Non-Amyloidogenic" for p in probs]

    result_df = pd.DataFrame({
        "ID"                          : ids,
        "Sequence"                    : sequences,
        "Class"                       : classes,
        "Probability"                 : probs,
        "Hydrophobicity (Avg)"        : hydro,
        "Beta-sheet Propensity (Avg)" : beta,
        "Solvent Accessibility (Avg)" : sasa,
    })

    result_df.to_csv(args.output, index=False, sep='\t')
    print(f"\nResults written to {args.output}")

    # Summary
    n_amy     = sum(1 for c in classes if c == "Amyloidogenic")
    n_non_amy = len(classes) - n_amy
    print(f"\n===== Prediction Summary =====")
    print(f"Total sequences   : {len(sequences)}")
    print(f"Amyloidogenic     : {n_amy}")
    print(f"Non-Amyloidogenic : {n_non_amy}")


if __name__ == "__main__":
    main()
