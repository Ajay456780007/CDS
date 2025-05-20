import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from demo import node_features

x=np.load("dataset/dataset1/smote_output/sequence_encoded_smote.npy")
node_features=np.load("dataset/dataset1/other_features.npy")
y=np.load("dataset/dataset1/smote_output/expression_label_smote.npy")
X_combined = np.concatenate([x,node_features], axis=1)

x=x[:5000]
y=y[:5000]

unique_classes, counts = np.unique(y, return_counts=True)
print("The unique values in y is:",unique_classes)
print("the number of counts of unique classes is:",counts)

