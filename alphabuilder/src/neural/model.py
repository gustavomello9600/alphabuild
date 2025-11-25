import tensorflow as tf
from tensorflow.keras import layers, models

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def create_vit_regressor(
    input_shape=(32, 64, 3), # (H, W, C) - Note: W=64, H=32 usually
    patch_size=4,
    num_patches=None, # Calculated automatically if None
    projection_dim=64,
    num_heads=4,
    transformer_layers=4,
    mlp_head_units=[2048, 1024],
):
    """
    Creates a Vision Transformer model for regression (Fitness prediction).
    """
    inputs = layers.Input(shape=input_shape)
    
    # Calculate num_patches if not provided
    if num_patches is None:
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    # Create patches
    patches = Patches(patch_size)(inputs)
    
    # Encode patches
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer Blocks
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = layers.Dense(units=projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(units=projection_dim, activation=tf.nn.gelu)(x3)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # MLP Head
    features = representation
    for units in mlp_head_units:
        features = layers.Dense(units, activation=tf.nn.gelu)(features)
        features = layers.Dropout(0.5)(features)

    # Output Layer (Scalar Regression)
    # Sigmoid activation because fitness is in [0, 1] range (Kane Eq 1)
    # But usually it's small, so linear or sigmoid? 
    # Kane fitness is 1 / (Mass + Penalty). Max is 1/Mass_min.
    # If mass is normalized, fitness is roughly in [0, ~10].
    # Let's use linear (None) or Relu to ensure positive.
    outputs = layers.Dense(1, activation="relu")(features)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
