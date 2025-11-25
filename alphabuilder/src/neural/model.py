import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np


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

@tf.keras.utils.register_keras_serializable()
class UniversalPatchEncoder(layers.Layer):
    def __init__(self, patch_size=4, projection_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.projection_dim = projection_dim

    def build(self, input_shape):
        # Input shape can be (B, H, W, C) or (B, D, H, W, C)
        # Channel dim is always last
        channels = input_shape[-1]
        if channels is None:
             raise ValueError("Channel dimension must be defined.")

        # 2D Kernel: (h, w, cin, cout)
        self.kernel_2d = self.add_weight(
            name="kernel_2d",
            shape=(self.patch_size, self.patch_size, channels, self.projection_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        # 3D Kernel: (d, h, w, cin, cout)
        self.kernel_3d = self.add_weight(
            name="kernel_3d",
            shape=(self.patch_size, self.patch_size, self.patch_size, channels, self.projection_dim),
            initializer="glorot_uniform",
            trainable=True
        )
        
        self.bias = self.add_weight(
            name="bias",
            shape=(self.projection_dim,),
            initializer="zeros",
            trainable=True
        )
        
        # CLS Token
        self.cls_token = self.add_weight(
            name="cls_token",
            shape=(1, 1, self.projection_dim),
            initializer="zeros",
            trainable=True
        )
        
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update({
            "patch_size": self.patch_size,
            "projection_dim": self.projection_dim,
        })
        return config

    def _get_sinusoidal_embeddings_2d(self, height, width):
        """Generate 2D sinusoidal embeddings."""
        d_model = self.projection_dim
        y_pos = tf.range(height, dtype=tf.float32)
        x_pos = tf.range(width, dtype=tf.float32)
        d_half = d_model // 2
        
        def get_1d_embedding(pos, d):
            i = tf.range(d, dtype=tf.float32)
            angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d, tf.float32))
            angle_rads = pos[:, tf.newaxis] * angle_rates[tf.newaxis, :]
            sines = tf.math.sin(angle_rads[:, 0::2])
            cosines = tf.math.cos(angle_rads[:, 1::2])
            return tf.concat([sines, cosines], axis=-1)

        y_emb = get_1d_embedding(y_pos, d_half)
        x_emb = get_1d_embedding(x_pos, d_half)
        
        y_emb_grid = tf.tile(y_emb[:, tf.newaxis, :], [1, width, 1])
        x_emb_grid = tf.tile(x_emb[tf.newaxis, :, :], [height, 1, 1])
        
        embeddings = tf.concat([y_emb_grid, x_emb_grid], axis=-1)
        return tf.reshape(embeddings, [1, height * width, d_model])

    def _get_sinusoidal_embeddings_3d(self, depth, height, width):
        """Generate 3D sinusoidal embeddings."""
        d_model = self.projection_dim
        d_third = d_model // 3
        
        def get_1d_embedding(pos, d):
            i = tf.range(d, dtype=tf.float32)
            angle_rates = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(d, tf.float32))
            angle_rads = pos[:, tf.newaxis] * angle_rates[tf.newaxis, :]
            sines = tf.math.sin(angle_rads[:, 0::2])
            cosines = tf.math.cos(angle_rads[:, 1::2])
            return tf.concat([sines, cosines], axis=-1)

        z_emb = get_1d_embedding(tf.range(depth, dtype=tf.float32), d_third)
        y_emb = get_1d_embedding(tf.range(height, dtype=tf.float32), d_third)
        x_emb = get_1d_embedding(tf.range(width, dtype=tf.float32), d_model - 2*d_third)
        
        z_grid = tf.tile(z_emb[:, tf.newaxis, tf.newaxis, :], [1, height, width, 1])
        y_grid = tf.tile(y_emb[tf.newaxis, :, tf.newaxis, :], [depth, 1, width, 1])
        x_grid = tf.tile(x_emb[tf.newaxis, tf.newaxis, :, :], [depth, height, 1, 1])
        
        embeddings = tf.concat([z_grid, y_grid, x_grid], axis=-1)
        return tf.reshape(embeddings, [1, depth * height * width, d_model])

    def call(self, images):
        # Input is always 5D: (Batch, Depth, Height, Width, Channels)
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        depth = input_shape[1]
        
        # Check if input is "flat" (2D treated as 3D with depth=1)
        is_flat = tf.equal(depth, 1)
        
        # Prepare safe input for 3D branch to satisfy XLA shape inference
        # When input is 2D (Depth=1), Conv3D with Depth=4 would fail static analysis.
        # So we feed a dummy tensor of Depth=4 to the 3D branch in that case.
        def _get_safe_3d():
            # Tile to depth 4: (B, 1, H, W, C) -> (B, 4, H, W, C)
            return tf.tile(images, [1, 4, 1, 1, 1])
            
        safe_3d_input = tf.cond(is_flat, _get_safe_3d, lambda: images)
        
        def apply_2d_sim():
            # Use 2D kernel reshaped as (1, P, P, C, Out)
            # Strides: (1, 1, P, P, 1) -> No stride in depth
            k2d = tf.expand_dims(self.kernel_2d, axis=0)
            return tf.nn.conv3d(
                images, 
                k2d, 
                strides=[1, 1, self.patch_size, self.patch_size, 1], 
                padding="VALID"
            )
            
        def apply_3d():
            # Use 3D kernel (P, P, P, C, Out)
            # Use safe_3d_input which is guaranteed to have Depth >= 4
            return tf.nn.conv3d(
                safe_3d_input, 
                self.kernel_3d, 
                strides=[1, self.patch_size, self.patch_size, self.patch_size, 1], 
                padding="VALID"
            )
            
        x = tf.cond(is_flat, apply_2d_sim, apply_3d)
        x = tf.nn.bias_add(x, self.bias)
        
        # Recalculate shapes after conv
        d_prime = tf.shape(x)[1]
        h_prime = tf.shape(x)[2]
        w_prime = tf.shape(x)[3]
        
        # Flatten patches
        x = tf.reshape(x, [batch_size, d_prime * h_prime * w_prime, self.projection_dim])
        
        # Add 3D positional embeddings
        # For 2D (d_prime=1), this works if the embedding function handles it correctly
        pos_emb = self._get_sinusoidal_embeddings_3d(d_prime, h_prime, w_prime)
        
        encoded = x + pos_emb
        
        cls_broadcast = tf.tile(self.cls_token, [batch_size, 1, 1])
        return tf.concat([cls_broadcast, encoded], axis=1)


def create_vit_regressor(
    input_shape=(None, None, 3), # Flexible shape
    patch_size=4,
    projection_dim=64,
    num_heads=4,
    transformer_layers=4,
    mlp_head_units=[2048, 1024],
):
    """
    Creates a Universal ViT model (2D/3D, Variable Resolution).
    """
    inputs = layers.Input(shape=input_shape)
    
    # Encode patches (Universal: handles 2D/3D and variable sizes)
    encoded_patches = UniversalPatchEncoder(patch_size, projection_dim)(inputs)

    # Transformer Blocks
    for _ in range(transformer_layers):
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
        x3 = layers.Dense(projection_dim)(x3)
        x3 = layers.Dropout(0.1)(x3)
        # Skip Connection 2
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
