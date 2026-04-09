import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def compute_all_class_weights(labels):
    num_groups = labels.shape[1]
    weights = []
    for i in range(num_groups):
        yy = labels[:, i].astype(float)
        unique_classes = np.unique(yy)
        cw = compute_class_weight("balanced", classes=unique_classes, y=yy)
        weight_dict = {c: w for c, w in zip(unique_classes, cw)}
        weights.append([weight_dict.get(0.0, 1.0), weight_dict.get(1.0, 1.0)])
    return np.array(weights, dtype=np.float32)


def get_group_losses_vectorized(y_true, y_pred, group_weights):
    """Parallel implementation using vectorization across all groups."""
    # 1. Compute BCE for all elements: shape (batch, num_groups)
    err = tf.keras.backend.binary_crossentropy(y_true, y_pred)

    # 2. Reshape weights for broadcasting: (1, num_groups)
    w0 = tf.transpose(group_weights[:, 0:1])
    w1 = tf.transpose(group_weights[:, 1:2])

    # 3. Apply weights in one parallel step
    weight_vector = y_true * w1 + (1 - y_true) * w0

    # 4. Mean across batch dimension: returns (num_groups,)
    return tf.reduce_mean(err * weight_vector, axis=0)


def multi_gap_vectorized(y_true, y_pred, group_weights, penalty_weight=0.05):
    """Broadcasting pairwise + Vectorized group losses"""
    group_losses = get_group_losses_vectorized(y_true, y_pred, group_weights)

    # 1. Overall Error: Sum of individual group means
    overall_err = tf.reduce_sum(group_losses)

    # 2. Pairwise Difference Error (v1 logic):
    # We use broadcasting to compute (L_i - L_j)^2 for all pairs in parallel.
    # This avoids a global mean bottleneck and treats each group pair independently.
    # expand_dims creates shape (1, num_groups) and (num_groups, 1) to trigger a [num_groups, num_groups] matrix.
    diffs = tf.expand_dims(group_losses, axis=0) - tf.expand_dims(group_losses, axis=1)

    diff_err = tf.reduce_sum(tf.square(diffs)) / 2.0  # Divide by 2 because we count both (i,j) and (j,i)
    return overall_err + penalty_weight * diff_err


# Configuration
N = 10000  # Number of samples
G = 7  # Number of groups
labels = np.random.choice([0, 1], size=(N, G))

# Compute generalized weights
weights = compute_all_class_weights(labels)
print("Computed weights for each group:")
for i, w in enumerate(weights):
    print(f"Group {i}: {w}")


# Use the multi_gap loss function
def get_model_loss(weights, penalty_weight=0.01):
    weights_tensor = tf.constant(weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        return multi_gap_vectorized(y_true, y_pred, weights_tensor, penalty_weight)

    return loss
