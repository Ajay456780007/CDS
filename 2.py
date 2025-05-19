import numpy as np
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split

# === Load Data ===
sequence_and_labels = np.load('dataset/dataset1/seq_and_label.npy')
other_features = np.load('dataset/dataset1/other_features.npy')

dna_sequences = sequence_and_labels[:, :-1].astype(np.int32)
expression_labels = sequence_and_labels[:, -1].astype(int)
node_features = other_features

print("Expression labels unique values:", np.unique(expression_labels))

# === Create Hypergraph (Shared) ===
def create_random_hypergraph(num_nodes, num_hyperedges, connection_prob=0.1):
    return (np.random.rand(num_nodes, num_hyperedges) < connection_prob).astype(np.float32)

shared_G = create_random_hypergraph(50, 50)  # shape (50, 50)

# === Split Data ===
train_idx, test_idx = train_test_split(np.arange(len(dna_sequences)), test_size=0.2,
                                      stratify=expression_labels, random_state=42)

train_data = {
    'dna': dna_sequences[train_idx],
    'node': node_features[train_idx],
    'label': expression_labels[train_idx]
}
test_data = {
    'dna': dna_sequences[test_idx],
    'node': node_features[test_idx],
    'label': expression_labels[test_idx]
}

# === Dataset Preparation ===
def prepare_dataset(dna, node, labels, batch_size=4, shuffle=True):
    # Repeat shared_G for each sample in the batch during mapping
    dataset = tf.data.Dataset.from_tensor_slices((dna, node, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(labels), seed=42)
    
    def map_fn(dna_seq, node_feat, label):
        # dna_seq: (seq_len,), node_feat: (node_feat_dim,)
        # Expand dims and tile node_feat to shape (50, node_feat_dim)
        node_feat_tiled = tf.tile(tf.expand_dims(node_feat, axis=0), [50, 1])
        # Tile shared_G to match batch dim later in model call, so just pass shared_G once
        return (dna_seq, node_feat_tiled, shared_G), label

    dataset = dataset.map(map_fn)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

train_ds = prepare_dataset(train_data['dna'], train_data['node'], train_data['label'], batch_size=4, shuffle=True)
test_ds = prepare_dataset(test_data['dna'], test_data['node'], test_data['label'], batch_size=4, shuffle=False)

# === Positional Encoding ===
def get_positional_encoding(seq_len, model_dim):
    angle_rads = np.arange(seq_len)[:, np.newaxis] / np.power(10000, (2 * (np.arange(model_dim)[np.newaxis, :] // 2)) / model_dim)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

# === Sinusoidal Embedding ===
class SinusoidalEmbedding(layers.Layer):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
    def call(self, x):
        half_dim = self.model_dim // 2
        freqs = tf.exp(tf.linspace(tf.math.log(1.0), tf.math.log(1000.0), half_dim))
        angles = 2.0 * np.pi * x * freqs
        return tf.concat([tf.sin(angles), tf.cos(angles)], axis=-1)[..., tf.newaxis]

# === Transformer Block ===
class TransformerBlock(layers.Layer):
    def __init__(self, model_dim, heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=heads, key_dim=model_dim // heads, dropout=rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dense(model_dim),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    def call(self, x, training=False):
        x = self.norm1(x + self.dropout1(self.att(x, x), training=training))
        return self.norm2(x + self.dropout2(self.ffn(x), training=training))

# === Diffusion Transformer ===
class DiffusionTransformer(tf.keras.Model):
    def __init__(self, seq_len=6000, model_dim=128, num_heads=4, ff_dim=256, depth=4, dropout_rate=0.1):
        super().__init__()
        self.embedding = layers.Embedding(input_dim=5, output_dim=model_dim)
        self.pos_encoding = get_positional_encoding(seq_len, model_dim)
        self.time_emb = SinusoidalEmbedding(model_dim)
        self.transformer_blocks = [TransformerBlock(model_dim, num_heads, ff_dim, dropout_rate) for _ in range(depth)]
        self.global_pool = layers.GlobalAveragePooling1D()
    def call(self, x, training=False):
        noise_var = tf.zeros((tf.shape(x)[0], 1))
        x = self.embedding(x)
        x += self.pos_encoding[:, :tf.shape(x)[1], :]
        noise_emb = self.time_emb(noise_var)
        noise_emb = tf.transpose(noise_emb, [0, 2, 1])
        noise_emb = tf.tile(noise_emb, [1, tf.shape(x)[1], 1])
        x += tf.cast(noise_emb, tf.float32)
        for block in self.transformer_blocks:
            x = block(x, training=training)
        return self.global_pool(x)

# === HGNN Components ===
class HGNNConv(layers.Layer):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = layers.Dense(output_dim, use_bias=False)
    def call(self, x, G):
        x = self.linear(x)
        return tf.matmul(tf.transpose(G, [0, 2, 1]), x)

class HGNNEmbedding(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.5):
        super().__init__()
        self.hgc1 = HGNNConv(input_dim, hidden_dim)
        self.hgc2 = HGNNConv(hidden_dim, hidden_dim)
        self.dropout = layers.Dropout(dropout_rate)
    def call(self, x, G, training=False):
        x = self.hgc1(x, G)
        x = self.dropout(x, training=training)
        x = self.hgc2(x, G)
        return tf.reduce_mean(x, axis=1)

# === Simple Fusion + Classification Model ===
class SimpleFusionModel(tf.keras.Model):
    def __init__(self, fusion_dim=128, num_classes=3, dropout_rate=0.2):
        super().__init__()
        self.dense1 = layers.Dense(fusion_dim, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    def call(self, x, training=False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        return self.output_layer(x)

# === Combined Model ===
class CombinedModel(tf.keras.Model):
    def __init__(self, diffusion_transformer, hgnn_embedding, fusion_model):
        super().__init__()
        self.diffusion_transformer = diffusion_transformer
        self.hgnn_embedding = hgnn_embedding
        self.fusion_model = fusion_model
    def call(self, inputs, training=False):
        dna_seq, node_feat, hypergraph = inputs
        dt_out = self.diffusion_transformer(dna_seq, training=training)  # shape (batch, model_dim)
        hg_out = self.hgnn_embedding(node_feat, hypergraph, training=training)  # shape (batch, hidden_dim)
        concat = tf.concat([dt_out, hg_out], axis=-1)
        return self.fusion_model(concat, training=training)

# === Instantiate Models ===
diffusion_transformer = DiffusionTransformer(seq_len=dna_sequences.shape[1])
hgnn_embedding = HGNNEmbedding(input_dim=node_features.shape[1])
fusion_model = SimpleFusionModel(fusion_dim=128, num_classes=len(np.unique(expression_labels)))

model = CombinedModel(diffusion_transformer, hgnn_embedding, fusion_model)

# === Compile ===
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# === Train ===
model.fit(train_ds, epochs=10)
loss, acc = model.evaluate(test_ds)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")
