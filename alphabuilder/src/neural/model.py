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
        input_shape = tf.shape(images)
        batch_size = input_shape[0]
        rank = tf.rank(images)
        is_3d = tf.equal(rank, 5)
        
        # Normalize input to 5D to satisfy XLA static shape requirements
        # We use tf.reshape with a dynamically constructed shape vector
        # This avoids tf.cond returning tensors of different ranks
        
        shape = tf.shape(images)
        
        def _shape_5d():
            return shape
            
        def _shape_4d_to_5d():
            # Insert dim 1 at axis 1: (B, H, W, C) -> (B, 1, H, W, C)
            return tf.concat([shape[:1], [1], shape[1:]], axis=0)
            
        target_shape = tf.cond(is_3d, _shape_5d, _shape_4d_to_5d)
        x_5d = tf.reshape(images, target_shape)
        
        def process_2d():
            # Squeeze back to 4D for 2D processing
            # x_5d is (B, 1, H, W, C) -> (B, H, W, C)
            x_4d = tf.squeeze(x_5d, axis=1)
            
            x = tf.nn.conv2d(
                x_4d, 
                self.kernel_2d, 
                strides=[1, self.patch_size, self.patch_size, 1], 
                padding="VALID"
            )
            x = tf.nn.bias_add(x, self.bias)
            
            h_prime = tf.shape(x)[1]
            w_prime = tf.shape(x)[2]
            
            x = tf.reshape(x, [batch_size, h_prime * w_prime, self.projection_dim])
            pos_emb = self._get_sinusoidal_embeddings_2d(h_prime, w_prime)
            return x + pos_emb
            
        def process_3d():
            # Use x_5d directly
            x = tf.nn.conv3d(
                x_5d, 
                self.kernel_3d, 
                strides=[1, self.patch_size, self.patch_size, self.patch_size, 1], 
                padding="VALID"
            )
            x = tf.nn.bias_add(x, self.bias)
            
            d_prime = tf.shape(x)[1]
            h_prime = tf.shape(x)[2]
            w_prime = tf.shape(x)[3]
            
            x = tf.reshape(x, [batch_size, d_prime * h_prime * w_prime, self.projection_dim])
            pos_emb = self._get_sinusoidal_embeddings_3d(d_prime, h_prime, w_prime)
            return x + pos_emb

        encoded = tf.cond(is_3d, process_3d, process_2d)
        
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
