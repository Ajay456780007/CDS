from imblearn.over_sampling import SMOTE
from collections import Counter
import numpy as np
import os

# Load original data
x = np.load("dataset/dataset1/sequence_encoded.npy")  # shape: (65000, seq_len)
node_features = np.load("dataset/dataset1/other_features.npy",allow_pickle=True)  # shape: (65000, num_features)
y = np.load("dataset/dataset1/expression_label.npy")  # shape: (65000,)

# Combine sequences and other features
x_combined = np.concatenate([x, node_features], axis=1)
print('Original combined shape:', x_combined.shape)
print('Before SMOTE class distribution:', Counter(y))

# Apply SMOTE to combined input
smote = SMOTE(random_state=42, k_neighbors=2)
X_sm, y_sm = smote.fit_resample(x_combined, y)
print('After SMOTE class distribution:', Counter(y_sm))

# Split back: assume you know how many columns are sequence vs. features
seq_len = x.shape[1]
x_sm = X_sm[:, :seq_len]
features_sm = X_sm[:, seq_len:]

# Save all to disk
output_dir = "dataset/dataset1/smote_output1"
os.makedirs(output_dir, exist_ok=True)

np.save(os.path.join(output_dir, "sequence_encoded_smote.npy"), x_sm)
np.save(os.path.join(output_dir, "other_features_smote.npy"), features_sm)
np.save(os.path.join(output_dir, "expression_label_smote.npy"), y_sm)

print("SMOTE done and saved:")
print(f"- sequence_encoded_smote.npy: {x_sm.shape}")
print(f"- other_features_smote.npy:   {features_sm.shape}")
print(f"- expression_label_smote.npy: {y_sm.shape}")
