import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

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

def get_sinusoidal_embeddings(num_positions, projection_dim):
    """
    Generates fixed sinusoidal positional embeddings.
    """
    positions = np.arange(num_positions)[:, np.newaxis]
    div_term = np.exp(np.arange(0, projection_dim, 2) * -(np.log(10000.0) / projection_dim))
    
    embeddings = np.zeros((num_positions, projection_dim))
    embeddings[:, 0::2] = np.sin(positions * div_term)
    embeddings[:, 1::2] = np.cos(positions * div_term)
    
    return tf.constant(embeddings, dtype=tf.float32)

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        # Fixed Sinusoidal Embeddings (Crucial Context #2)
        self.position_embedding = get_sinusoidal_embeddings(num_patches + 1, projection_dim) # +1 for CLS token

    def call(self, patch):
        # patch shape: (batch, num_patches, patch_dim)
        encoded = self.projection(patch)
        
        # Add CLS token (Crucial Context #3)
        batch_size = tf.shape(patch)[0]
        cls_token = tf.zeros((batch_size, 1, encoded.shape[-1])) # Learnable CLS token could be better, but zero is standard start
        encoded = tf.concat([cls_token, encoded], axis=1)
        
        # Add positional embeddings
        encoded = encoded + self.position_embedding
        return encoded

def create_vit_regressor(
    input_shape=(32, 64, 3), # (H, W, C)
    patch_size=4,
    num_patches=None, 
    projection_dim=64,
    num_heads=4,
    transformer_layers=4,
    mlp_head_units=[2048, 1024],
):
    """
    Creates a Vision Transformer model for Max Displacement prediction.
    Complies with 'contexto_crucial.md'.
    """
    inputs = layers.Input(shape=input_shape)
    
    if num_patches is None:
        num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)

    # Create patches
    patches = Patches(patch_size)(inputs)
    
    # Encode patches (includes CLS token and Sinusoidal Embeddings)
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Transformer Blocks
    for _ in range(transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = layers.Dense(units=projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dense(units=projection_dim, activation=tf.nn.gelu)(x3)
        encoded_patches = layers.Add()([x3, x2])

    # Aggregation: Use CLS Token (Index 0) (Crucial Context #3)
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    cls_token_output = representation[:, 0, :] 
    
    features = layers.Dropout(0.5)(cls_token_output)

    # MLP Head
    for units in mlp_head_units:
        features = layers.Dense(units, activation=tf.nn.gelu)(features)
        features = layers.Dropout(0.5)(features)

    # Output Layer: Max Displacement (Crucial Context #1)
    # Using softplus to ensure positive output, as displacement is magnitude
    outputs = layers.Dense(1, activation="softplus", name="max_displacement_output")(features)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
