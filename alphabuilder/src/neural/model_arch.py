import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Constants matching data_loader
MAX_DEPTH = 16
MAX_HEIGHT = 64
MAX_WIDTH = 128
CHANNELS = 3

def get_sinusoidal_embeddings(num_positions, projection_dim):
    """
    Generates sinusoidal positional embeddings.
    """
    positions = np.arange(num_positions)[:, np.newaxis]
    div_term = np.exp(np.arange(0, projection_dim, 2) * -(np.log(10000.0) / projection_dim))
    
    embeddings = np.zeros((num_positions, projection_dim))
    embeddings[:, 0::2] = np.sin(positions * div_term)
    embeddings[:, 1::2] = np.cos(positions * div_term)
    
    return tf.constant(embeddings, dtype=tf.float32)

class SinusoidalPositionalEmbedding(layers.Layer):
    """
    Fixed Sinusoidal Positional Embedding Layer.
    Better for generalization than learned embeddings.
    """
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.pos_embedding = get_sinusoidal_embeddings(num_patches, projection_dim)

    def call(self, x):
        return x + self.pos_embedding

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
        })
        return config

class ClassToken(layers.Layer):
    """
    Appends a learnable CLS token to the input sequence.
    """
    def __init__(self, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.w_init = tf.random_normal_initializer()

    def build(self, input_shape):
        self.cls_token = tf.Variable(
            initial_value=self.w_init(shape=(1, 1, self.projection_dim), dtype="float32"),
            trainable=True,
            name="cls_token"
        )

    def call(self, x):
        batch_size = tf.shape(x)[0]
        # Broadcast cls_token to batch size
        cls_token_broadcast = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_token_broadcast, x], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"projection_dim": self.projection_dim})
        return config

def build_3d_vit(
    input_shape: tuple = (MAX_DEPTH, MAX_HEIGHT, MAX_WIDTH, CHANNELS),
    patch_size: tuple = (2, 8, 8),
    num_heads: int = 4,
    transformer_layers: int = 4,
    projection_dim: int = 64,
    mlp_head_units: list = [2048, 1024],
) -> keras.Model:
    """
    Builds a 3D Vision Transformer model optimized for Max Displacement prediction.
    
    Changes from v1:
    - Sinusoidal Positional Embeddings (Generalization)
    - CLS Token (Representation)
    - Output: Max Displacement (Scalar, > 0)
    """
    
    inputs = layers.Input(shape=input_shape)
    
    # --- 1. Patch Creation (using Conv3D) ---
    patches = layers.Conv3D(
        filters=projection_dim,
        kernel_size=patch_size,
        strides=patch_size,
        padding="VALID",
        name="patch_projection"
    )(inputs)
    
    # Reshape to sequence
    d_patches = input_shape[0] // patch_size[0]
    h_patches = input_shape[1] // patch_size[1]
    w_patches = input_shape[2] // patch_size[2]
    num_patches = d_patches * h_patches * w_patches
    
    encoded_patches = layers.Reshape((num_patches, projection_dim))(patches)
    
    # --- 2. CLS Token & Positional Embedding ---
    # Add CLS token *before* positional embedding (standard ViT)
    # Note: If we add CLS first, we need num_patches + 1 embeddings.
    encoded_patches = ClassToken(projection_dim)(encoded_patches)
    
    # Add Positional Embeddings (Fixed Sinusoidal)
    # We need embeddings for num_patches + 1 (CLS)
    encoded_patches = SinusoidalPositionalEmbedding(num_patches + 1, projection_dim)(encoded_patches)
    
    # --- 3. Transformer Blocks ---
    for i in range(transformer_layers):
        # Layer Normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        
        # Multi-Head Attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        
        # Skip Connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer Normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        
        # MLP
        x3 = layers.Dense(projection_dim * 2, activation=tf.nn.gelu)(x3)
        x3 = layers.Dropout(0.1)(x3)
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)
        
        # Skip Connection 2
        encoded_patches = layers.Add()([x3, x2])
        
    # --- 4. Head (Use CLS Token) ---
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    # Extract CLS token (index 0)
    # Shape: (Batch, projection_dim)
    cls_token_output = layers.Lambda(lambda x: x[:, 0])(representation)
    
    # MLP Head for Regression
    features = cls_token_output
    for units in mlp_head_units:
        features = layers.Dense(units, activation=tf.nn.gelu)(features)
        features = layers.Dropout(0.2)(features)
        
    # Output: Max Displacement (Scalar)
    # We use Softplus to ensure positivity, as displacement >= 0
    outputs = layers.Dense(1, activation='softplus', name="max_displacement_output")(features)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="AlphaBuilder_ViT_3D_v2")
    return model

if __name__ == "__main__":
    # Smoke Test
    print("Running Smoke Test for Model Arch v2...")
    
    model = build_3d_vit()
    model.summary()
    
    # Mock Input
    mock_input = tf.random.normal((2, MAX_DEPTH, MAX_HEIGHT, MAX_WIDTH, CHANNELS))
    
    # Forward Pass
    output = model(mock_input)
    
    print(f"Input Shape: {mock_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    assert output.shape == (2, 1), f"Expected (2, 1), got {output.shape}"
    assert np.all(output.numpy() >= 0), "Output must be non-negative (Softplus)"
    
    print("Model Arch v2 Smoke Test Passed!")
