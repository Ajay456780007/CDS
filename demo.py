import math
import time
import matplotlib.pyplot as plt
import gzip
import seaborn as sns
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from urllib.parse import unquote
import numpy as np
from keras import Sequential, layers, Model
from collections import Counter

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, \
    mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional
from keras import layers
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import layers, models
from sklearn.model_selection import StratifiedShuffleSplit
from keras.utils import to_categorical
from keras import mixed_precision
import os

import torch
from torch import nn
from layer_HGNN import HGNN_conv
import torch.nn.functional as F



# === Paths ===
def read_data():
    import pandas as pd
    import numpy as np
    import os
    import gzip
    from urllib.parse import unquote
    from Bio import SeqIO
    from Bio.Seq import Seq
    from sklearn.preprocessing import StandardScaler

    dna_dir = "dataset/dataset1/dna_chromosomes/"
    gff3_dir = "dataset/dataset1/gff3_files/"

    # === Collect all FASTA and GFF3 files ===
    fasta_files = sorted([
        os.path.join(dna_dir, f) for f in os.listdir(dna_dir)
        if f.lower().endswith(".fa.gz")
    ])
    gff3_files = sorted([
        os.path.join(gff3_dir, f) for f in os.listdir(gff3_dir)
        if f.lower().endswith(".gff3")
    ])

    # === Parse GFF3 attributes ===
    def parse_attributes(attr_str):
        attr_dict = {}
        for pair in attr_str.strip().split(";"):
            if "=" in pair:
                key, value = pair.split("=", 1)
                attr_dict[key.strip()] = unquote(value.strip())
        return attr_dict

    # === Function to parse GFF3 and extract CDS entries for a given chromosome ===
    def parse_gff3_cds(gff3_file, chrom_id):
        cds_dict = {}
        with open(gff3_file, encoding="utf-8") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue
                if parts[2] != "CDS":
                    continue
                if parts[0] != chrom_id:
                    continue

                start = int(parts[3]) - 1  # GFF3 is 1-based; convert to 0-based
                end = int(parts[4])  # end is exclusive
                strand = parts[6]
                attrs = parse_attributes(parts[8])
                parent_id = attrs.get("Parent", "NA")

                if parent_id not in cds_dict:
                    cds_dict[parent_id] = {
                        "strand": strand,
                        "ranges": []
                    }
                cds_dict[parent_id]["ranges"].append((start, end))
        return cds_dict

    # === Extract CDS sequences with start/end positions ===
    cds_sequences = []

    for fasta_path, gff_path in zip(fasta_files, gff3_files):
        print(f"Processing: {os.path.basename(fasta_path)} with {os.path.basename(gff_path)}")

        # Read chromosome sequence
        with gzip.open(fasta_path, "rt", encoding="utf-8") if fasta_path.endswith(".gz") else open(fasta_path, "r",
                                                                                                   encoding="utf-8") as f:
            record = next(SeqIO.parse(f, "fasta"))

        chrom_seq = record.seq
        chrom_id = record.id

        # Parse CDS entries
        cds_dict = parse_gff3_cds(gff_path, chrom_id)
        print(f"Found {len(cds_dict)} CDS transcripts in {os.path.basename(gff_path)}")

        for parent_id, info in cds_dict.items():
            strand = info["strand"]
            regions = sorted(info["ranges"], key=lambda x: x[0])
            full_seq = "".join(str(chrom_seq[start:end]) for start, end in regions)
            if strand == "-":
                full_seq = str(Seq(full_seq).reverse_complement())

            transcript_start = min(start for start, end in regions)
            transcript_end = max(end for start, end in regions)

            cds_sequences.append({
                "transcript_id": parent_id,
                "chrom": chrom_id,
                "strand": strand,
                "start": transcript_start,
                "end": transcript_end,
                "sequence": full_seq
            })

    df_cds = pd.DataFrame(cds_sequences)
    df_cds.to_csv("dataset/dataset1/cds_sequences.csv", index=False)

    # === Load mapping and expression data ===
    alias_df = pd.read_csv("dataset/dataset1/alias/genes_to_alias_ids.tsv", sep="\t", header=None)
    tpm_df = pd.read_csv("dataset/dataset1/geo_file/abundance.tsv", sep="\t")

    alias_df.columns = ["e_id", "source", "d_id", "agpv4_id"]
    tpm_df["gene_id"] = tpm_df["target_id"].apply(lambda x: x.split("_T")[0])
    df_cds["clean_gene_id"] = df_cds["transcript_id"].apply(lambda x: x.replace("transcript:", "").split("_T")[0])

    df_merged = df_cds.merge(alias_df[["e_id", "d_id"]], left_on="clean_gene_id", right_on="e_id", how="left")

    avg_tpm = tpm_df.groupby("gene_id", as_index=False)["tpm"].mean().rename(columns={"tpm": "avg_tpm"})
    final_df = df_merged.merge(avg_tpm, left_on="d_id", right_on="gene_id", how="left")

    final_df = final_df.drop(columns=["clean_gene_id", "e_id", "gene_id"])
    final_df.dropna(subset=['avg_tpm'], inplace=True)

    # === Normalize start and end columns using z-score
    scaler = StandardScaler()
    final_df[["start_z", "end_z"]] = scaler.fit_transform(final_df[["start", "end"]])

    # === Encode transcript_id, strand, sequence
    final_df["transcript_index"] = pd.factorize(final_df["transcript_id"])[0]
    final_df["strand_numeric"] = final_df["strand"].map({'+': 1, '-': 0})

    def encode_sequence(seq):
        mapping = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        return [mapping.get(base.upper(), 4) for base in seq]

    final_df["sequence_encoded"] = final_df["sequence"].apply(encode_sequence)

    # === Convert avg_tpm to expression labels
    def compute_labels(tpm_array):
        mean_tpm = np.mean(tpm_array)
        low = mean_tpm / 2
        high = mean_tpm * 1.5
        return np.array([0 if t < low else 1 if t < high else 2 for t in tpm_array], dtype=np.int32)

    final_df["expression_label"] = compute_labels(final_df["avg_tpm"].values)

    # === Pad or truncate sequences
    FIXED_LEN = 6000
    PAD_VALUE = 4

    def pad_or_truncate(seq):
        return seq[:FIXED_LEN] if len(seq) > FIXED_LEN else seq + [PAD_VALUE] * (FIXED_LEN - len(seq))

    final_df['sequence_encoded'] = final_df['sequence_encoded'].apply(pad_or_truncate)

    # === Final feature selection and saving
    # Save X and y
    np.save('dataset/dataset1/sequence_encoded.npy', np.array(final_df['sequence_encoded'].tolist(), dtype=np.int32))
    np.save('dataset/dataset1/expression_label.npy', final_df['expression_label'].values.astype(np.int32))

    # Save other features (excluding sequence, label, and transcript index)
    other_features = final_df.drop(columns=['sequence_encoded', 'expression_label', 'transcript_index','transcript_id',"sequence","avg_tpm","start","end","strand","d_id"])
    np.save('dataset/dataset1/other_features.npy', other_features.values)

    # === Logging info ===
    print("Unique label values:", final_df["expression_label"].unique())
    print("Transcript ID sample:", final_df["transcript_id"].head())
    print("other_features shape:", other_features.shape)
    print("sequence_encoded shape:", np.load('dataset/dataset1/sequence_encoded.npy').shape)
    print("expression_label shape:", np.load('dataset/dataset1/expression_label.npy').shape)


# read_data()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# === Load Only First 50 Samples Directly ===
sequence_data = np.load('dataset/dataset1/sequence_encoded.npy')[:200, :6000]
expression_labels = np.load("dataset/dataset1/expression_label.npy")[:200].astype(int)
node_features = np.load('dataset/dataset1/other_features.npy',allow_pickle=True)[:200]  # shape (2000, 4)

node_features = np.array(list(node_features)).astype(np.float32)

print("Node features shape:",node_features.shape)
# # === Create Shared Hypergraph Matrix ===
# def create_random_hypergraph(num_nodes, num_hyperedges, connection_prob=0.1):
#     return np.random.rand(num_nodes, num_hyperedges) < connection_prob
#
# shared_G = create_random_hypergraph(50, 50).astype(np.float32)  # (50, 50)
# shared_G_batch = tf.convert_to_tensor(shared_G[np.newaxis, ...])  # (1, 50, 50)
#
# # === Train/Test Split (same for transformer and HGNN) ===
# indices = np.arange(50)
# train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=expression_labels, random_state=42)
#
# x_train_dna = sequence_data[train_idx]
# x_test_dna = sequence_data[test_idx]
# x_train_node = node_features[train_idx]
# x_test_node = node_features[test_idx]
# y_train = expression_labels[train_idx]
# y_test = expression_labels[test_idx]
#
# from sklearn.model_selection import train_test_split
#

x=sequence_data #shape(100,6000)
y=expression_labels #shape(100)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# import numpy as np
# from collections import Counter
# from sklearn.metrics import accuracy_score
#
# class KNN:
#     def __init__(self, k):
#         self.k = k
#         print(f"KNN initialized with k = {self.k}")
#
#     def fit(self, X_train, y_train):
#         if self.k > len(X_train):
#             raise ValueError("k cannot be greater than the number of training samples")
#         self.x_train = np.array(X_train)
#         self.y_train = np.array(y_train).flatten()
#
#     def calculate_euclidean(self, sample1, sample2):
#         return np.linalg.norm(sample1.astype(np.float32) - sample2.astype(np.float32))
#
#     def nearest_neighbors(self, test_sample):
#         distances = [
#             (self.y_train[i], self.calculate_euclidean(self.x_train[i], test_sample))
#             for i in range(len(self.x_train))
#         ]
#         distances.sort(key=lambda x: x[1])  # Sort by distance
#         neighbors = [distances[i][0] for i in range(self.k)]
#         return neighbors
#
#     def majority_vote(self, neighbors):
#         count = Counter(neighbors)
#         return sorted(count.items(), key=lambda x: (-x[1], x[0]))[0][0]
#
#     def predict(self, test_set):
#         predictions = []
#         for test_sample in test_set:
#             neighbors = self.nearest_neighbors(test_sample)
#             prediction = self.majority_vote(neighbors)
#             predictions.append(prediction)
#         return predictions
# #KNN Model Building
# # === Apply KNN to your dataset ===
# # Make sure x_train, x_test, y_train, y_test are already defined
# #
# # model3 = KNN(k=5)
# # model3.fit(x_train, y_train)
# # predictions = model3.predict(x_test)
# #
# # accuracy = accuracy_score(y_test, predictions)
# # print(f"Accuracy for KNN: {accuracy:.4f}")
#
#
#
#
#
# def create_BiLSTM(input_shape, num_classes):
#     model = Sequential()
#     model.add(Bidirectional(LSTM(units=64,
#                                  return_sequences=False,
#                                  activation='tanh'),
#                             input_shape=input_shape))
#     model.add(Dense(units=num_classes, activation='softmax'))
#
#     model.compile(loss='sparse_categorical_crossentropy',
#                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                   metrics=["accuracy"])
#     return model
#
# #BiLSTM Model Buildimg
#
# # # Number of classes in your classification
# # num_classes = 3  # e.g., low = 0, medium = 1, high = 2
#
#
# # model2 = create_BiLSTM(x_train_bilstm.shape[1:], num_classes)
# # model2.fit(x_train_bilstm, y_train, epochs=10, batch_size=32, validation_split=0.1)
# #
# #
# # loss_bilstm, acc_bilstm = model2.evaluate(x_test_bilstm, y_test)
# # print("The loss of BiLSTM:", loss_bilstm)
# # print("The accuracy of BiLSTM:", acc_bilstm)
#
#
#
# def compute_metrics(y_true, y_pred, average='macro'):
#     cm = confusion_matrix(y_true, y_pred)
#     tp = np.diag(cm)
#     fn = cm.sum(axis=1) - tp
#     fp = cm.sum(axis=0) - tp
#     tn = cm.sum() - (tp + fn + fp)
#     specificity = np.mean(tn / (tn + fp)) if np.all(tn + fp) else 0.0
#
#     return {
#         "confusion_matrix": cm,
#         "accuracy": accuracy_score(y_true, y_pred),
#         "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
#         "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
#         "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
#         "specificity": specificity,
#         "mae": mean_absolute_error(y_true, y_pred),
#         "mse": mean_squared_error(y_true, y_pred)
#     }
#
# # === Main Evaluation Loop ===
# results = {"ProposedModel": [], "KNN": [], "BiLSTM": []}
# metrics = {"ProposedModel": [], "KNN": [], "BiLSTM": []}
# training_percentage = [40, 50, 60, 70, 80, 90]
#
# #
# x_all = np.concatenate([x_train, x_test], axis=0)
# y_all = np.concatenate([y_train, y_test], axis=0)
#
# for percent in training_percentage:
#     print(f"\n=== Training with {percent}% of total data ===")
#     indices = np.arange(len(x_all))
#     np.random.shuffle(indices)
#     num_train = int(len(x_all) * percent / 100)
#
#
#     train_idx = indices[:num_train]
#     test_idx = indices[num_train:]
#
#
#     def get_generators(x_all, y_all, node_features_reduced, hg_adj, train_idx, test_idx, batch_size=32):
#         x_train = x_all[train_idx]
#         y_train = y_all[train_idx]
#         x_test = x_all[test_idx]
#         y_test = y_all[test_idx]
#
#         node_features_train = node_features_reduced[train_idx]
#         node_features_test = node_features_reduced[test_idx]
#         hg_adj_train = hg_adj[train_idx]
#         hg_adj_test = hg_adj[test_idx]
#
#         # Create data generators
#         train_gen = HybridDataGenerator(x_train, node_features_train, hg_adj_train, y_train,
#                                         batch_size=batch_size, shuffle=True)
#         test_gen = HybridDataGenerator(x_test, node_features_test, hg_adj_test, y_test,
#                                        batch_size=batch_size, shuffle=False)
#
#         return train_gen, test_gen, y_test
#
#
#     print(
#         f"x_train: {x_train.shape}, node_features_train: {node_features_train.shape}, hg_adj_train: {hg_adj_train.shape}")
#     print(f"x_test: {x_test.shape}, node_features_test: {node_features_test.shape}, hg_adj_test: {hg_adj_test.shape}")
#
#     # --- Proposed Model ---
#     train_gen, test_gen, y_test_split = get_generators(x_all, y_all, node_features_reduced, hg_adj, train_idx, test_idx,
#                                                        batch_size=32)
#
#     # --- Proposed Model ---
#     combined_model = CombinedModel(diffusion_transformer, hgnn_embedding, fusion_dim=64, num_classes=3)
#     combined_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#                            loss='sparse_categorical_crossentropy',
#                            metrics=['accuracy'])
#
#     combined_model.fit(train_gen, validation_data=test_gen, epochs=10, verbose=1)
#
#     # Predict on test set using generator
#     y_pred_probs = combined_model.predict(test_gen)
#     y_pred = np.argmax(y_pred_probs, axis=1)
#     metric_vals = compute_metrics(y_test, y_pred)
#     results["ProposedModel"].append(metric_vals["accuracy"])
#     metrics["ProposedModel"].append(metric_vals)
#     print(f"ProposedModel Accuracy: {metric_vals['accuracy']:.4f}")
#
#     # #--- BiLSTM Model---
#     # x_train_bilstm = np.expand_dims(x_train, axis=-1).astype(np.float32)
#     # x_test_bilstm = np.expand_dims(x_test, axis=-1).astype(np.float32)
#     # model2 = create_BiLSTM(x_train_bilstm.shape[1:], num_classes=3)
#     # model2.fit(x_train_bilstm, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=0)
#     # y_pred = np.argmax(model2.predict(x_test_bilstm), axis=-1)
#     # metric_vals = compute_metrics(y_test, y_pred)
#     # results["BiLSTM"].append(metric_vals["accuracy"])
#     # metrics["BiLSTM"].append(metric_vals)
#     # print(f"BiLSTM Accuracy: {metric_vals['accuracy']:.4f}")
#     #
#     # # --- KNN ---
#     # knn_model = KNN(k=5)
#     # knn_model.fit(x_train, y_train)
#     # y_pred = knn_model.predict(x_test)
#     # metric_vals = compute_metrics(y_test, y_pred)
#     # results["KNN"].append(metric_vals["accuracy"])
#     # metrics["KNN"].append(metric_vals)
#     # print(f"KNN Accuracy: {metric_vals['accuracy']:.4f}")
#
# # === Save results ===
# np.save("model_accuracy_results.npy", results)
# np.save("model_detailed_metrics.npy", metrics)
#
# # === Accuracy Plot ===
# bar_width = 0.2
# x_range = np.arange(len(training_percentage))
# model_names = list(results.keys())
# plt.figure(figsize=(12, 6))
#
# for i, model_name in enumerate(model_names):
#     plt.bar(x_range + i * bar_width, results[model_name], width=bar_width, label=model_name)
#
# plt.xlabel("Training Percentage")
# plt.ylabel("Accuracy")
# plt.title("Model Accuracy vs Training Data Percentage")
# plt.xticks(x_range + bar_width, training_percentage)
# plt.legend()
# plt.tight_layout()
# plt.savefig("training_percentage_comparison_bar.png")
# plt.show()
#
# # === Confusion Matrices ===
# for model_name in model_names:
#     for i, percent in enumerate(training_percentage):
#         cm = metrics[model_name][i]["confusion_matrix"]
#         plt.figure(figsize=(6, 5))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.title(f"{model_name} Confusion Matrix ({percent}%)")
#         plt.xlabel("Predicted")
#         plt.ylabel("True")
#         plt.tight_layout()
#         plt.savefig(f"conf_matrix_{model_name}_{percent}.png")
#         plt.close()


x_hgnn = torch.tensor(node_features, dtype=torch.float32) #shape(1000,4)

y_hgnn = torch.tensor(y, dtype=torch.long) #shape(50,)
import numpy as np


def build_incidence_matrix_from_labels(y_hgnn):
    """
    Build incidence matrix H from TPM class labels.
    Each unique label defines a hyperedge.

    Parameters:
        y (Tensor): Tensor of shape [num_genes], with labels like 0,1,2

    Returns:
        H (ndarray): [num_genes, num_hyperedges]
    """
    y_np = y_hgnn.cpu().numpy() if hasattr(y_hgnn, 'cpu') else y_hgnn  # Convert to numpy if tensor
    num_genes = len(y_np) #shape(50,)
    classes = np.unique(y_np) #shape(3,)
    num_hyperedges = len(classes) # value 3

    H = np.zeros((num_genes, num_hyperedges), dtype=np.float32) # shape(50,3)

    for j, label in enumerate(classes):
        H[:, j] = (y_np == label).astype(float)  # mark genes in this class
    return H


def generate_G_from_H(H):
    H = torch.tensor(H)
    Dv = torch.diag(torch.sum(H, dim=1))  # Vertex degrees
    De = torch.diag(torch.sum(H, dim=0))  # Hyperedge degrees
    De_inv = torch.inverse(De)
    Dv_inv_sqrt = torch.inverse(torch.sqrt(Dv))
    HT = torch.transpose(H, 0, 1)
    G = Dv_inv_sqrt @ H @ De_inv @ HT @ Dv_inv_sqrt
    return G


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


H = build_incidence_matrix_from_labels(y) #shape(50,3)
G = generate_G_from_H(H) #shape(50,50)



in_ch=x_hgnn.shape[1] #4
n_class=len(torch.unique(y_hgnn)) #3
idx = torch.arange(len(y_hgnn)) #shape(50)
n_hid=128

model_HGNN=HGNN(in_ch,n_class,n_hid,dropout=0.5)
# optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
# criterion=nn.CrossEntropyLoss()

# from sklearn.model_selection import train_test_split
#
# idx = torch.arange(len(y))
# train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
#
# x_train, y_train = x[train_idx], y[train_idx]
# x_test, y_test = x[test_idx], y[test_idx]

# epochs = 50
# model.train()
# for epoch in range(epochs):
#     optimizer.zero_grad()
#     out = model(x, G)  # Entire graph used, but only train on train_idx
#     loss = criterion(out[train_idx], y_train)
#     loss.backward()
#     optimizer.step()
#
#
#     print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")
#
# model.eval()
# with torch.no_grad():
#     preds = model(x, G)
#     pred_classes = preds[test_idx].argmax(dim=1)
#     acc = (pred_classes == y_test).float().mean()
#     print(f"Test Accuracy: {acc:.4f}")
#


# diffusion transformer--

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math

# Positional Embedding (no parameters saved)
class PositionalEmbedding(nn.Module):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = torch.outer(torch.arange(x.size(1), device=device).float(), emb)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb.unsqueeze(0)  # shape (1, L, dim)


class LinformerAttention(nn.Module):
    def __init__(self, seq_len, dim, n_heads, k, bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.qw = nn.Linear(dim, dim, bias=bias)
        self.kw = nn.Linear(dim, dim, bias=bias)
        self.vw = nn.Linear(dim, dim, bias=bias)
        self.E = nn.Parameter(torch.randn(seq_len, k))
        self.F = nn.Parameter(torch.randn(seq_len, k))
        self.ow = nn.Linear(dim, dim, bias=bias)

    def forward(self, x):
        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)
        B, L, D = q.shape
        q = q.view(B, L, self.n_heads, -1).permute(0, 2, 1, 3)
        k = k.view(B, L, self.n_heads, -1).permute(0, 2, 3, 1)
        v = v.view(B, L, self.n_heads, -1).permute(0, 2, 3, 1)

        k = torch.matmul(k, self.E[:L, :])
        v = torch.matmul(v, self.F[:L, :]).permute(0, 1, 3, 2)

        qk = torch.matmul(q, k) * self.scale
        attn = torch.softmax(qk, dim=-1)

        v_attn = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(B, L, D)
        return self.ow(v_attn)


class TransformerBlock(nn.Module):
    def __init__(self, seq_len, dim, heads, mlp_dim, k):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = LinformerAttention(seq_len, dim, heads, k)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class DiT1D(nn.Module):
    def __init__(self, seq_len, dim, depth, heads, mlp_dim, k, input_dim, output_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, dim)
        self.pos_embedding = PositionalEmbedding(dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(seq_len, dim, heads, mlp_dim, k) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)



# Simulated input
sequence_data = sequence_data #shape(50,6000)
expression_labels = expression_labels #shape(50,)

sequence_data_tensor = torch.tensor(sequence_data, dtype=torch.long) #shape(50,6000)
expression_labels = torch.tensor(expression_labels, dtype=torch.long) #shape(50)

# One-hot encode DNA sequences
x_dna = F.one_hot(sequence_data_tensor, num_classes=5).float() #shape(50,6000,5)
#
# # Dataset class
# class DNADataset(Dataset):
#     def __init__(self, x_data, labels):
#         self.x_data = x_data
#         self.labels = labels
#
#     def __len__(self):
#         return len(self.x_data)
#
#     def __getitem__(self, idx):
#         return self.x_data[idx], self.labels[idx]
#
# # DataLoader
# dataset = DNADataset(x_dna, expression_labels)
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dit = DiT1D(seq_len=6000, dim=128, depth=2, heads=4, mlp_dim=256,
              k=64, input_dim=5, output_dim=3).to(device)
#
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# loss_fn = nn.CrossEntropyLoss()

# # Training Loop
# # Training Loop with Accuracy
# for epoch in range(15):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#
#     for batch_x, batch_y in dataloader:
#         batch_x = batch_x.to(device)
#         batch_y = batch_y.to(device)
#
#         pred = model(batch_x)
#
#         # Calculate loss
#         loss = loss_fn(pred, batch_y)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         # Track total loss
#         total_loss += loss.item()
#
#         # Calculate accuracy
#         predicted_labels = torch.argmax(pred, dim=1)
#         correct += (predicted_labels == batch_y).sum().item()
#         total += batch_y.size(0)
#
#     avg_loss = total_loss / len(dataloader)
#     accuracy = correct / total * 100
#
#     print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
#


import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridGeneExpressionModel(nn.Module):
    def __init__(self, hgnn, dit, hgnn_output_dim, dit_output_dim, hidden_dim, num_classes):
        super(HybridGeneExpressionModel, self).__init__()
        self.hgnn = hgnn
        self.dit = dit
        self.fc1 = nn.Linear(hgnn_output_dim + dit_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x_hgnn, G, x_dna):
        out_hgnn = self.hgnn(x_hgnn, G)  # shape: [N, n_class] or [N, hgnn_output_dim]
        out_dit = self.dit(x_dna)  # shape: [N, dit_output_dim]

        # Concatenate outputs along feature dimension
        combined = torch.cat([out_hgnn, out_dit], dim=1)  # shape: [N, hgnn_out + dit_out]

        hidden = F.relu(self.fc1(combined))
        output = self.fc2(hidden)  # shape: [N, num_classes]

        return output


model = HybridGeneExpressionModel(hgnn=model_HGNN,
                                  dit=model_dit,
                                  hgnn_output_dim=3,   # From HGNN final layer
                                  dit_output_dim=3,    # From DiT final layer
                                  hidden_dim=64,
                                  num_classes=3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


class HybridDataset(Dataset):
    def __init__(self, node_feats, adj_G, dna_seqs, labels):
        self.node_feats = node_feats
        self.G = adj_G
        self.dna_seqs = dna_seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.node_feats[idx], self.G, self.dna_seqs[idx], self.labels[idx]


train_dataset = HybridDataset(x, G, x_dna, expression_labels) #x=(50,6000), G=(50,50), x_dna=(50,6000,5) ,expression_labels=(50)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)


# Split data (make sure x_hgnn and x_dna are aligned)
from sklearn.model_selection import train_test_split

indices = np.arange(x_hgnn.shape[0]) #shape(1000,)
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42) #train_idx=(800,), test_idx=(200,)

x_hgnn_train = x_hgnn[train_idx] # shape (800,4)
x_dna_train = x_dna[train_idx]
y_train = y_hgnn[train_idx]

x_hgnn_test = x_hgnn[test_idx]
x_dna_test = x_dna[test_idx]
y_test = y_hgnn[test_idx]

print("Train labels:", torch.unique(y_train, return_counts=True))
print("y_train dtype:", y_train.dtype)

# Training Loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

model.train()
epochs = 15
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(x_hgnn, G, x_dna)  # [N, num_classes]
    loss = loss_fn(output, y_hgnn)
    loss.backward()
    optimizer.step()
    print("Model output shape:", output.shape)

    pred = output.argmax(dim=1)
    acc = (pred == y_hgnn).float().mean().item()
    print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {acc:.4f}")
