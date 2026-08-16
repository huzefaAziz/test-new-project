import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model
from typing import Tuple, List, Optional, Dict
import math

# ============================================================================
# Helper Functions
# ============================================================================

def reparameterize(mean, logvar):
    """Reparameterization trick for sampling from Gaussian distribution."""
    eps = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * logvar) * eps

def gaussian_kl(mean1, logvar1, mean2, logvar2):
    """KL divergence between two Gaussian distributions."""
    kl = 0.5 * (
        logvar2 - logvar1 + 
        (tf.exp(logvar1) + (mean1 - mean2)**2) / tf.exp(logvar2) - 1.0
    )
    return tf.reduce_sum(kl, axis=-1)

def standard_gaussian_kl(mean, logvar):
    """KL divergence between Gaussian and standard normal."""
    return gaussian_kl(mean, logvar, 0.0, 0.0)

def diagonal_gaussian_log_prob(x, mean, logvar):
    """Log probability of x under diagonal Gaussian."""
    var = tf.exp(logvar)
    log_prob = -0.5 * (tf.math.log(2.0 * np.pi) + logvar + (x - mean)**2 / var)
    return tf.reduce_sum(log_prob, axis=-1)


# ============================================================================
# Core Variational Components
# ============================================================================

class VariationalEncoder(layers.Layer):
    """Variational encoder that outputs mean and log variance."""
    
    def __init__(self, latent_dim, hidden_dims=[256, 128], name="variational_encoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        
        # Build MLP for encoding
        self.hidden_layers = []
        for i, dim in enumerate(hidden_dims):
            self.hidden_layers.append(layers.Dense(dim, activation='relu', name=f'enc_dense_{i}'))
        
        # Output layers for mean and log variance
        self.mean_dense = layers.Dense(latent_dim, name='mean_dense')
        self.logvar_dense = layers.Dense(latent_dim, name='logvar_dense')
    
    def call(self, inputs, training=True):
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        
        mean = self.mean_dense(x)
        logvar = self.logvar_dense(x)
        # Clamp logvar for numerical stability
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)
        
        return mean, logvar


class VariationalDecoder(layers.Layer):
    """Decoder that reconstructs from latent representation."""
    
    def __init__(self, output_dim, hidden_dims=[128, 256], name="variational_decoder", **kwargs):
        super().__init__(name=name, **kwargs)
        self.output_dim = output_dim
        
        self.hidden_layers = []
        for i, dim in enumerate(hidden_dims):
            self.hidden_layers.append(layers.Dense(dim, activation='relu', name=f'dec_dense_{i}'))
        
        self.output_dense = layers.Dense(output_dim, name='output_dense')
    
    def call(self, z, training=True):
        x = z
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_dense(x)


class ConditionalPrior(layers.Layer):
    """Conditional prior p(sy | sx, z) - JEPA predictor."""
    
    def __init__(self, latent_dim, hidden_dims=[256, 128], name="conditional_prior", **kwargs):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        
        self.hidden_layers = []
        for i, dim in enumerate(hidden_dims):
            self.hidden_layers.append(layers.Dense(dim, activation='relu', name=f'prior_dense_{i}'))
        
        self.mean_dense = layers.Dense(latent_dim, name='prior_mean')
        self.logvar_dense = layers.Dense(latent_dim, name='prior_logvar')
    
    def call(self, sx, z, training=True):
        # Concatenate context and auxiliary latents
        combined = tf.concat([sx, z], axis=-1)
        x = combined
        for layer in self.hidden_layers:
            x = layer(x)
        
        mean = self.mean_dense(x)
        logvar = self.logvar_dense(x)
        logvar = tf.clip_by_value(logvar, -10.0, 10.0)
        
        return mean, logvar


# ============================================================================
# Main Var-JEPA Model
# ============================================================================

class VarJEPA(Model):
    """
    Variational Joint-Embedding Predictive Architecture.
    
    Implements the full ELBO objective with:
    - Context encoder: q(sx | x)
    - Target posterior: q(sy | sx, z, y)
    - Auxiliary encoder: q(z | sx)
    - Conditional prior: p(sy | sx, z)
    - Context decoder: p(x | sx)
    - Target decoder: p(y | sy)
    """
    
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        aux_latent_dim: int = 16,
        hidden_dims: List[int] = [256, 128],
        alpha_rec: float = 1.0,
        alpha_gen: float = 1.0,
        alpha_kl_sx: float = 1.0,
        alpha_kl_z: float = 1.0,
        alpha_kl_sy: float = 1.0,
        reconstruction_std: float = 1.0,
        name: str = "var_jepa",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.latent_dim = latent_dim
        self.aux_latent_dim = aux_latent_dim
        self.input_dim = input_dim
        
        # Loss weights
        self.alpha_rec = alpha_rec
        self.alpha_gen = alpha_gen
        self.alpha_kl_sx = alpha_kl_sx
        self.alpha_kl_z = alpha_kl_z
        self.alpha_kl_sy = alpha_kl_sy
        self.reconstruction_std = reconstruction_std
        
        # Context encoder: q(sx | x)
        self.context_encoder = VariationalEncoder(
            latent_dim, 
            hidden_dims=hidden_dims,
            name='context_encoder'
        )
        
        # Auxiliary encoder: q(z | sx)
        self.aux_encoder = VariationalEncoder(
            aux_latent_dim,
            hidden_dims=[64, 32],
            name='aux_encoder'
        )
        
        # Target posterior: q(sy | sx, z, y)
        self.target_posterior = VariationalEncoder(
            latent_dim,
            hidden_dims=hidden_dims,
            name='target_posterior'
        )
        
        # Conditional prior (predictor): p(sy | sx, z)
        self.predictor = ConditionalPrior(
            latent_dim,
            hidden_dims=hidden_dims,
            name='predictor'
        )
        
        # Context decoder: p(x | sx)
        self.context_decoder = VariationalDecoder(
            input_dim,
            hidden_dims=list(reversed(hidden_dims)),
            name='context_decoder'
        )
        
        # Target decoder: p(y | sy)
        self.target_decoder = VariationalDecoder(
            input_dim,
            hidden_dims=list(reversed(hidden_dims)),
            name='target_decoder'
        )
        
        # For tracking losses
        self._losses_dict = {}
    
    def encode_context(self, x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode context observation to latent distribution."""
        return self.context_encoder(x)
    
    def encode_aux(self, sx: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode context latent to auxiliary latent."""
        return self.aux_encoder(sx)
    
    def encode_target(self, sx: tf.Tensor, z: tf.Tensor, y: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode target observation conditioned on context and aux."""
        combined = tf.concat([sx, z, y], axis=-1)
        return self.target_posterior(combined)
    
    def predict_target(self, sx: tf.Tensor, z: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict target latent from context and aux."""
        return self.predictor(sx, z)
    
    def decode_context(self, sx: tf.Tensor) -> tf.Tensor:
        """Reconstruct context from latent."""
        return self.context_decoder(sx)
    
    def decode_target(self, sy: tf.Tensor) -> tf.Tensor:
        """Reconstruct target from latent."""
        return self.target_decoder(sy)
    
    def sample_latents(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Sample all latents using reparameterization trick.
        
        Returns:
            sx, z, sy, sx_mean, sx_logvar, z_mean, z_logvar, sy_mean, sy_logvar
        """
        # Encode context
        sx_mean, sx_logvar = self.encode_context(x)
        sx = reparameterize(sx_mean, sx_logvar) if training else sx_mean
        
        # Encode auxiliary
        z_mean, z_logvar = self.encode_aux(sx)
        z = reparameterize(z_mean, z_logvar) if training else z_mean
        
        # Encode target posterior
        sy_mean, sy_logvar = self.encode_target(sx, z, y)
        sy = reparameterize(sy_mean, sy_logvar) if training else sy_mean
        
        return sx, z, sy, sx_mean, sx_logvar, z_mean, z_logvar, sy_mean, sy_logvar
    
    def compute_elbo_loss(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        training: bool = True
    ) -> Tuple[tf.Tensor, dict]:
        """
        Compute the complete ELBO loss.
        
        Returns:
            total_loss: The weighted negative ELBO
            loss_dict: Dictionary of individual loss components
        """
        # Sample latents
        sx, z, sy, sx_mean, sx_logvar, z_mean, z_logvar, sy_mean, sy_logvar = \
            self.sample_latents(x, y, training)
        
        # 1. Context reconstruction loss: E[log p(x | sx)]
        x_recon = self.decode_context(sx)
        recon_loss = -diagonal_gaussian_log_prob(
            x, x_recon, 
            tf.math.log(self.reconstruction_std**2 + 1e-8) * tf.ones_like(x)
        )
        L_rec = tf.reduce_mean(recon_loss)
        
        # 2. Target generation loss: E[log p(y | sy)]
        y_recon = self.decode_target(sy)
        gen_loss = -diagonal_gaussian_log_prob(
            y, y_recon,
            tf.math.log(self.reconstruction_std**2 + 1e-8) * tf.ones_like(y)
        )
        L_gen = tf.reduce_mean(gen_loss)
        
        # 3. KL divergence for context latent: KL(q(sx|x) || p(sx))
        L_kl_sx = tf.reduce_mean(standard_gaussian_kl(sx_mean, sx_logvar))
        
        # 4. KL divergence for auxiliary latent: KL(q(z|sx) || p(z))
        L_kl_z = tf.reduce_mean(standard_gaussian_kl(z_mean, z_logvar))
        
        # 5. KL divergence for target posterior: KL(q(sy|sx,z,y) || p(sy|sx,z))
        # Get conditional prior parameters
        prior_mean, prior_logvar = self.predict_target(sx, z)
        L_kl_sy = tf.reduce_mean(gaussian_kl(sy_mean, sy_logvar, prior_mean, prior_logvar))
        
        # Total negative ELBO
        total_loss = (
            self.alpha_rec * L_rec +
            self.alpha_gen * L_gen +
            self.alpha_kl_sx * L_kl_sx +
            self.alpha_kl_z * L_kl_z +
            self.alpha_kl_sy * L_kl_sy
        )
        
        loss_dict = {
            'total': total_loss,
            'L_rec': L_rec,
            'L_gen': L_gen,
            'L_kl_sx': L_kl_sx,
            'L_kl_z': L_kl_z,
            'L_kl_sy': L_kl_sy,
        }
        
        self._losses_dict = loss_dict
        return total_loss, loss_dict
    
    def call(self, inputs: tf.Tensor, training: bool = True) -> dict:
        """
        Forward pass. For compatibility with standard Keras API.
        """
        # If inputs is a tuple/list, treat as (x, y)
        if isinstance(inputs, (tuple, list)):
            x, y = inputs
        else:
            # If single input, use it as both x and y (for self-supervised learning)
            x = inputs
            y = inputs
        
        sx, z, sy, sx_mean, sx_logvar, z_mean, z_logvar, sy_mean, sy_logvar = \
            self.sample_latents(x, y, training)
        
        x_recon = self.decode_context(sx)
        y_recon = self.decode_target(sy)
        
        return {
            'sx': sx,
            'z': z,
            'sy': sy,
            'sx_mean': sx_mean,
            'sx_logvar': sx_logvar,
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'sy_mean': sy_mean,
            'sy_logvar': sy_logvar,
            'x_recon': x_recon,
            'y_recon': y_recon,
        }
    
    def train_step(self, data):
        """Custom training step with ELBO loss."""
        x, y = data
        
        with tf.GradientTape() as tape:
            total_loss, loss_dict = self.compute_elbo_loss(x, y, training=True)
        
        # Compute gradients and update
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # Return metrics
        metrics = {f'loss_{k}': v for k, v in loss_dict.items()}
        metrics['loss'] = total_loss
        return metrics
    
    def test_step(self, data):
        """Test step for evaluation."""
        x, y = data
        total_loss, loss_dict = self.compute_elbo_loss(x, y, training=False)
        
        metrics = {f'val_loss_{k}': v for k, v in loss_dict.items()}
        metrics['val_loss'] = total_loss
        return metrics
    
    def get_embeddings(
        self, 
        x: tf.Tensor, 
        y: Optional[tf.Tensor] = None,
        use_target: bool = False
    ) -> tf.Tensor:
        """
        Get deterministic embeddings for downstream tasks.
        Uses posterior means (deterministic) rather than sampling.
        
        Args:
            x: Context observations
            y: Target observations (optional)
            use_target: If True, use target posterior mean; otherwise use context encoder
        
        Returns:
            Embeddings tensor
        """
        if y is None or not use_target:
            # Use context encoder only (for pure representation learning)
            sx_mean, _ = self.encode_context(x)
            return sx_mean
        else:
            # Use target posterior mean (conditioned on context and target)
            sx_mean, _ = self.encode_context(x)
            z_mean, _ = self.encode_aux(sx_mean)
            sy_mean, _ = self.encode_target(sx_mean, z_mean, y)
            return sy_mean
    
    def get_uncertainty(
        self,
        x: tf.Tensor,
        y: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """
        Get per-sample uncertainty estimates from latent covariances.
        
        Returns:
            Uncertainty scores (batch_size,)
        """
        if y is None:
            # Use context encoder uncertainty
            _, sx_logvar = self.encode_context(x)
            uncertainty = tf.reduce_sum(tf.exp(sx_logvar), axis=-1)
        else:
            # Use target posterior uncertainty
            sx_mean, sx_logvar = self.encode_context(x)
            z_mean, z_logvar = self.encode_aux(sx_mean)
            _, sy_logvar = self.encode_target(sx_mean, z_mean, y)
            uncertainty = tf.reduce_sum(tf.exp(sy_logvar), axis=-1)
        
        return uncertainty


# ============================================================================
# Variational JEPA with Transformer Encoder (Var-T-JEPA for Tabular Data)
# ============================================================================

class TransformerEncoder(layers.Layer):
    """Transformer encoder for tabular data with feature tokenization."""
    
    def __init__(
        self,
        num_features: int,
        latent_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout_rate: float = 0.1,
        name: str = "transformer_encoder",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.num_features = num_features
        self.latent_dim = latent_dim
        
        # Feature embeddings for each feature
        self.feature_embedding = layers.Dense(latent_dim, name='feature_embedding')
        
        # Positional embeddings
        self.positional_embedding = self.add_weight(
            name='positional_embedding',
            shape=(num_features + 1, latent_dim),  # +1 for CLS token
            initializer='random_normal',
            trainable=True
        )
        
        # Transformer layers
        self.attention_layers = []
        self.ff_layers_1 = []
        self.ff_layers_2 = []
        self.ln_layers = []
        
        for i in range(num_layers):
            self.attention_layers.append(
                layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=latent_dim // num_heads,
                    dropout=dropout_rate,
                    name=f'transformer_attn_{i}'
                )
            )
            self.ff_layers_1.append(
                layers.Dense(ff_dim, activation='relu', name=f'transformer_ff_{i}_1')
            )
            self.ff_layers_2.append(
                layers.Dense(latent_dim, name=f'transformer_ff_{i}_2')
            )
            self.ln_layers.append(
                layers.LayerNormalization(epsilon=1e-6, name=f'transformer_ln_{i}')
            )
        
        self.dropout = layers.Dropout(dropout_rate)
    
    def call(self, inputs, training=True):
        # inputs: (batch, num_features) or (batch, num_features, feature_dim)
        if len(inputs.shape) == 2:
            # If inputs are flat features, project them
            x = self.feature_embedding(tf.expand_dims(inputs, axis=-1))
        else:
            x = self.feature_embedding(inputs)
        
        # Add CLS token
        batch_size = tf.shape(x)[0]
        cls_token = tf.zeros((batch_size, 1, self.latent_dim))
        x = tf.concat([cls_token, x], axis=1)
        
        # Add positional embeddings
        x = x + self.positional_embedding[tf.newaxis, :, :]
        x = self.dropout(x, training=training)
        
        # Transformer layers
        for i in range(len(self.attention_layers)):
            # Self-attention
            attn_out = self.attention_layers[i](x, x, training=training)
            x = x + attn_out
            
            # FFN
            ffn_out = self.ff_layers_1[i](x)
            ffn_out = self.ff_layers_2[i](ffn_out)
            x = x + ffn_out
            
            # Layer norm
            x = self.ln_layers[i](x)
        
        return x


class VarTJEPA(Model):
    """
    Var-T-JEPA: Variational JEPA for Tabular Data.
    
    Implements the full Var-JEPA framework for heterogeneous tabular data
    with feature-level masking and transformer-based encoders.
    """
    
    def __init__(
        self,
        num_features: int,
        latent_dim: int = 64,
        aux_latent_dim: int = 16,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_dim: int = 256,
        dropout_rate: float = 0.1,
        alpha_rec: float = 1.0,
        alpha_gen: float = 1.0,
        alpha_kl_sx: float = 1.0,
        alpha_kl_z: float = 1.0,
        alpha_kl_sy: float = 1.0,
        reconstruction_std: float = 1.0,
        context_ratio: float = 0.5,
        target_ratio: float = 0.3,
        name: str = "var_t_jepa",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.num_features = num_features
        self.latent_dim = latent_dim
        self.aux_latent_dim = aux_latent_dim
        self.context_ratio = context_ratio
        self.target_ratio = target_ratio
        
        # Loss weights
        self.alpha_rec = alpha_rec
        self.alpha_gen = alpha_gen
        self.alpha_kl_sx = alpha_kl_sx
        self.alpha_kl_z = alpha_kl_z
        self.alpha_kl_sy = alpha_kl_sy
        self.reconstruction_std = reconstruction_std
        
        # Feature embedding
        self.feature_embedding = layers.Dense(latent_dim, name='feature_embedding')
        
        # Context encoder (transformer)
        self.context_transformer = TransformerEncoder(
            num_features=num_features,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name='context_transformer'
        )
        
        # Target encoder (transformer)
        self.target_transformer = TransformerEncoder(
            num_features=num_features,
            latent_dim=latent_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            name='target_transformer'
        )
        
        # Predictor (conditional prior)
        self.predictor = ConditionalPrior(
            latent_dim,
            hidden_dims=[ff_dim, ff_dim // 2],
            name='predictor'
        )
        
        # Variational encoders for each latent
        self.context_encoder = VariationalEncoder(
            latent_dim,
            hidden_dims=[ff_dim // 2],
            name='context_encoder'
        )
        
        self.aux_encoder = VariationalEncoder(
            aux_latent_dim,
            hidden_dims=[ff_dim // 4],
            name='aux_encoder'
        )
        
        self.target_posterior = VariationalEncoder(
            latent_dim,
            hidden_dims=[ff_dim // 2],
            name='target_posterior'
        )
        
        # Decoders
        self.context_decoder = VariationalDecoder(
            num_features,
            hidden_dims=[ff_dim // 2, ff_dim],
            name='context_decoder'
        )
        
        self.target_decoder = VariationalDecoder(
            num_features,
            hidden_dims=[ff_dim // 2, ff_dim],
            name='target_decoder'
        )
        
        self._losses_dict = {}
    
    def create_masks(
        self,
        batch_size: int
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Create random context and target masks for feature-level masking.
        """
        num_context = int(self.num_features * self.context_ratio)
        num_target = int(self.num_features * self.target_ratio)
        
        # Ensure masks don't overlap
        max_total = num_context + num_target
        assert max_total <= self.num_features, f"Masks would overlap: {num_context} + {num_target} > {self.num_features}"
        
        # Create masks for each batch element
        context_masks = []
        target_masks = []
        
        for _ in range(batch_size):
            # Randomly select indices for context and target
            all_indices = np.random.permutation(self.num_features)
            context_indices = all_indices[:num_context]
            target_indices = all_indices[num_context:num_context + num_target]
            
            context_mask = np.zeros(self.num_features, dtype=np.float32)
            target_mask = np.zeros(self.num_features, dtype=np.float32)
            
            context_mask[context_indices] = 1.0
            target_mask[target_indices] = 1.0
            
            context_masks.append(context_mask)
            target_masks.append(target_mask)
        
        return tf.convert_to_tensor(context_masks), tf.convert_to_tensor(target_masks)
    
    def encode_context_with_mask(
        self,
        x: tf.Tensor,
        context_mask: tf.Tensor,
        training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode masked context features."""
        # Apply mask
        x_masked = x * context_mask
        
        # Project features (add feature dimension if needed)
        if len(x_masked.shape) == 2:
            x_embedded = self.feature_embedding(tf.expand_dims(x_masked, axis=-1))
        else:
            x_embedded = self.feature_embedding(x_masked)
        
        # Apply transformer
        x_transformed = self.context_transformer(x_embedded, training=training)
        
        # Extract CLS token representation
        cls_rep = x_transformed[:, 0, :]  # CLS token
        
        # Encode to variational distribution
        mean, logvar = self.context_encoder(cls_rep)
        return mean, logvar
    
    def encode_target_with_conditioning(
        self,
        sx: tf.Tensor,
        z: tf.Tensor,
        y: tf.Tensor,
        target_mask: tf.Tensor,
        training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Encode target features conditioned on context and auxiliary latents."""
        # Apply mask
        y_masked = y * target_mask
        
        # Project features
        if len(y_masked.shape) == 2:
            y_embedded = self.feature_embedding(tf.expand_dims(y_masked, axis=-1))
        else:
            y_embedded = self.feature_embedding(y_masked)
        
        # Apply transformer
        y_transformed = self.target_transformer(y_embedded, training=training)
        
        # Extract CLS token representation
        cls_rep = y_transformed[:, 0, :]
        
        # Concatenate with conditioning latents
        combined = tf.concat([cls_rep, sx, z], axis=-1)
        
        # Encode to variational distribution
        mean, logvar = self.target_posterior(combined)
        return mean, logvar
    
    def predict_target_latent(
        self,
        sx: tf.Tensor,
        z: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Predict target latent distribution from context and auxiliary."""
        return self.predictor(sx, z)
    
    def compute_elbo_loss(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        context_mask: Optional[tf.Tensor] = None,
        target_mask: Optional[tf.Tensor] = None,
        training: bool = True
    ) -> Tuple[tf.Tensor, dict]:
        """Compute ELBO loss with feature-level masking."""
        batch_size = tf.shape(x)[0]
        
        # Create masks if not provided
        if context_mask is None or target_mask is None:
            context_mask, target_mask = self.create_masks(
                batch_size.numpy() if isinstance(batch_size, tf.Tensor) else batch_size
            )
        
        # 1. Encode context
        sx_mean, sx_logvar = self.encode_context_with_mask(x, context_mask, training=training)
        sx = reparameterize(sx_mean, sx_logvar) if training else sx_mean
        
        # 2. Encode auxiliary latent
        z_mean, z_logvar = self.aux_encoder(sx)
        z = reparameterize(z_mean, z_logvar) if training else z_mean
        
        # 3. Encode target posterior
        sy_mean, sy_logvar = self.encode_target_with_conditioning(
            sx, z, y, target_mask, training=training
        )
        sy = reparameterize(sy_mean, sy_logvar) if training else sy_mean
        
        # 4. Predict target latent (conditional prior)
        prior_mean, prior_logvar = self.predict_target_latent(sx, z)
        
        # 5. Decode context and target
        x_recon = self.context_decoder(sx)
        y_recon = self.target_decoder(sy)
        
        # Compute losses
        # Context reconstruction (only on masked features)
        mask_ctx = tf.expand_dims(context_mask, axis=-1)
        recon_loss = -diagonal_gaussian_log_prob(
            x * mask_ctx,
            x_recon * mask_ctx,
            tf.math.log(self.reconstruction_std**2 + 1e-8) * tf.ones_like(x)
        )
        # Average over masked features
        L_rec = tf.reduce_sum(recon_loss * tf.squeeze(mask_ctx, axis=-1), axis=-1)
        L_rec = tf.reduce_mean(L_rec) / (tf.reduce_mean(context_mask) + 1e-8)
        
        # Target generation (only on masked features)
        mask_trg = tf.expand_dims(target_mask, axis=-1)
        gen_loss = -diagonal_gaussian_log_prob(
            y * mask_trg,
            y_recon * mask_trg,
            tf.math.log(self.reconstruction_std**2 + 1e-8) * tf.ones_like(y)
        )
        L_gen = tf.reduce_sum(gen_loss * tf.squeeze(mask_trg, axis=-1), axis=-1)
        L_gen = tf.reduce_mean(L_gen) / (tf.reduce_mean(target_mask) + 1e-8)
        
        # KL divergences
        L_kl_sx = tf.reduce_mean(standard_gaussian_kl(sx_mean, sx_logvar))
        L_kl_z = tf.reduce_mean(standard_gaussian_kl(z_mean, z_logvar))
        L_kl_sy = tf.reduce_mean(gaussian_kl(sy_mean, sy_logvar, prior_mean, prior_logvar))
        
        # Total loss
        total_loss = (
            self.alpha_rec * L_rec +
            self.alpha_gen * L_gen +
            self.alpha_kl_sx * L_kl_sx +
            self.alpha_kl_z * L_kl_z +
            self.alpha_kl_sy * L_kl_sy
        )
        
        loss_dict = {
            'total': total_loss,
            'L_rec': L_rec,
            'L_gen': L_gen,
            'L_kl_sx': L_kl_sx,
            'L_kl_z': L_kl_z,
            'L_kl_sy': L_kl_sy,
        }
        
        self._losses_dict = loss_dict
        return total_loss, loss_dict
    
    def call(self, inputs, training: bool = True) -> dict:
        """Forward pass."""
        if isinstance(inputs, (tuple, list)):
            x, y = inputs
        else:
            x = inputs
            y = inputs
        
        batch_size = tf.shape(x)[0]
        context_mask, target_mask = self.create_masks(
            batch_size.numpy() if isinstance(batch_size, tf.Tensor) else batch_size
        )
        
        sx_mean, sx_logvar = self.encode_context_with_mask(x, context_mask, training=training)
        sx = reparameterize(sx_mean, sx_logvar) if training else sx_mean
        
        z_mean, z_logvar = self.aux_encoder(sx)
        z = reparameterize(z_mean, z_logvar) if training else z_mean
        
        sy_mean, sy_logvar = self.encode_target_with_conditioning(
            sx, z, y, target_mask, training=training
        )
        sy = reparameterize(sy_mean, sy_logvar) if training else sy_mean
        
        x_recon = self.context_decoder(sx)
        y_recon = self.target_decoder(sy)
        
        return {
            'sx': sx,
            'sx_mean': sx_mean,
            'sx_logvar': sx_logvar,
            'z': z,
            'z_mean': z_mean,
            'z_logvar': z_logvar,
            'sy': sy,
            'sy_mean': sy_mean,
            'sy_logvar': sy_logvar,
            'x_recon': x_recon,
            'y_recon': y_recon,
            'context_mask': context_mask,
            'target_mask': target_mask,
        }
    
    def train_step(self, data):
        """Custom training step."""
        x, y = data
        total_loss, loss_dict = self.compute_elbo_loss(x, y, training=True)
        
        with tf.GradientTape() as tape:
            total_loss, loss_dict = self.compute_elbo_loss(x, y, training=True)
        
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        metrics = {f'loss_{k}': v for k, v in loss_dict.items()}
        metrics['loss'] = total_loss
        return metrics
    
    def test_step(self, data):
        """Test step."""
        x, y = data
        total_loss, loss_dict = self.compute_elbo_loss(x, y, training=False)
        
        metrics = {f'val_loss_{k}': v for k, v in loss_dict.items()}
        metrics['val_loss'] = total_loss
        return metrics
    
    def get_embeddings(
        self, 
        x: tf.Tensor, 
        y: Optional[tf.Tensor] = None,
        use_target: bool = False
    ) -> tf.Tensor:
        """
        Get deterministic embeddings for downstream tasks.
        Uses posterior means (deterministic) rather than sampling.
        """
        if y is None or not use_target:
            # Use context encoder only
            batch_size = tf.shape(x)[0]
            context_mask = tf.ones((batch_size, self.num_features))
            sx_mean, _ = self.encode_context_with_mask(x, context_mask, training=False)
            return sx_mean
        else:
            # Use target posterior mean
            batch_size = tf.shape(x)[0]
            context_mask = tf.ones((batch_size, self.num_features))
            target_mask = tf.ones((batch_size, self.num_features))
            
            sx_mean, _ = self.encode_context_with_mask(x, context_mask, training=False)
            z_mean, _ = self.aux_encoder(sx_mean)
            sy_mean, _ = self.encode_target_with_conditioning(
                sx_mean, z_mean, y, target_mask, training=False
            )
            return sy_mean
    
    def get_uncertainty(
        self,
        x: tf.Tensor,
        y: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        """Get per-sample uncertainty estimates."""
        if y is None:
            batch_size = tf.shape(x)[0]
            context_mask = tf.ones((batch_size, self.num_features))
            _, sx_logvar = self.encode_context_with_mask(x, context_mask, training=False)
            uncertainty = tf.reduce_sum(tf.exp(sx_logvar), axis=-1)
        else:
            batch_size = tf.shape(x)[0]
            context_mask = tf.ones((batch_size, self.num_features))
            target_mask = tf.ones((batch_size, self.num_features))
            
            sx_mean, _ = self.encode_context_with_mask(x, context_mask, training=False)
            z_mean, _ = self.aux_encoder(sx_mean)
            _, sy_logvar = self.encode_target_with_conditioning(
                sx_mean, z_mean, y, target_mask, training=False
            )
            uncertainty = tf.reduce_sum(tf.exp(sy_logvar), axis=-1)
        
        return uncertainty


# ============================================================================
# SIGReg (Sketched Isotropic Gaussian Regularization)
# ============================================================================

class SIGReg(layers.Layer):
    """
    Sketched Isotropic Gaussian Regularization.
    
    Encourages aggregated embedding distribution to match N(0, I)
    using random projections and Epps-Pulley test.
    """
    
    def __init__(
        self,
        latent_dim: int,
        num_projections: int = 64,
        name: str = "sigreg",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.latent_dim = latent_dim
        self.num_projections = num_projections
        
        # Random projection directions (fixed)
        self.projections = self.add_weight(
            name='projections',
            shape=(latent_dim, num_projections),
            initializer='random_normal',
            trainable=False
        )
        # Normalize projections
        self.projections.assign(self.projections / tf.norm(self.projections, axis=0, keepdims=True))
    
    def epps_pulley_statistic(self, projections: tf.Tensor) -> tf.Tensor:
        """Compute Epps-Pulley test statistic for normality."""
        batch_size = tf.cast(tf.shape(projections)[0], tf.float32)
        
        # Center the data
        centered = projections - tf.reduce_mean(projections, axis=0, keepdims=True)
        
        # Compute sample moments
        m2 = tf.reduce_mean(centered**2, axis=0)
        m4 = tf.reduce_mean(centered**4, axis=0)
        
        # Expected moments for standard normal
        expected_m2 = 1.0
        expected_m4 = 3.0
        
        # Test statistic (simplified Epps-Pulley approximation)
        stat = tf.reduce_mean((m2 - expected_m2)**2 + (m4 - expected_m4)**2)
        return stat
    
    def call(self, embeddings: tf.Tensor) -> tf.Tensor:
        """Compute SIGReg loss for a batch of embeddings."""
        # Project embeddings
        projections = tf.matmul(embeddings, self.projections)
        
        # Compute Epps-Pulley statistic for each projection
        stat = self.epps_pulley_statistic(projections)
        
        return stat


# ============================================================================
# Var-JEPA with SIGReg (Combined Objective)
# ============================================================================

class VarJEPAWithSIGReg(Model):
    """
    Var-JEPA with SIGReg regularization.
    Combines the ELBO objective with SIGReg for additional
    aggregated distribution regularization.
    """
    
    def __init__(
        self,
        var_jepa: VarJEPA,
        lambda_sx: float = 10.0,
        lambda_sy: float = 10.0,
        num_projections: int = 64,
        name: str = "var_jepa_sigreg",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        
        self.var_jepa = var_jepa
        self.lambda_sx = lambda_sx
        self.lambda_sy = lambda_sy
        
        # SIGReg modules for sx and sy
        self.sigreg_sx = SIGReg(
            var_jepa.latent_dim,
            num_projections=num_projections,
            name='sigreg_sx'
        )
        self.sigreg_sy = SIGReg(
            var_jepa.latent_dim,
            num_projections=num_projections,
            name='sigreg_sy'
        )
    
    def compute_loss(
        self,
        x: tf.Tensor,
        y: tf.Tensor,
        training: bool = True
    ) -> Tuple[tf.Tensor, dict]:
        """Compute combined ELBO + SIGReg loss."""
        # Forward pass to get latents
        outputs = self.var_jepa.call((x, y), training=training)
        
        # Compute ELBO loss
        elbo_loss, elbo_dict = self.var_jepa.compute_elbo_loss(x, y, training=training)
        
        # Compute SIGReg losses
        sigreg_sx = self.sigreg_sx(outputs['sx'])
        sigreg_sy = self.sigreg_sy(outputs['sy'])
        
        # Combined loss
        total_loss = elbo_loss + self.lambda_sx * sigreg_sx + self.lambda_sy * sigreg_sy
        
        loss_dict = {
            'total': total_loss,
            'elbo': elbo_loss,
            'sigreg_sx': sigreg_sx,
            'sigreg_sy': sigreg_sy,
            **elbo_dict
        }
        
        return total_loss, loss_dict
    
    def call(self, inputs, training: bool = True) -> dict:
        return self.var_jepa.call(inputs, training=training)
    
    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            total_loss, loss_dict = self.compute_loss(x, y, training=True)
        
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        metrics = {f'loss_{k}': v for k, v in loss_dict.items()}
        metrics['loss'] = total_loss
        return metrics
    
    def test_step(self, data):
        x, y = data
        total_loss, loss_dict = self.compute_loss(x, y, training=False)
        
        metrics = {f'val_loss_{k}': v for k, v in loss_dict.items()}
        metrics['val_loss'] = total_loss
        return metrics
    
    def get_embeddings(
        self, 
        x: tf.Tensor, 
        y: Optional[tf.Tensor] = None,
        use_target: bool = False
    ) -> tf.Tensor:
        return self.var_jepa.get_embeddings(x, y, use_target)
    
    def get_uncertainty(
        self,
        x: tf.Tensor,
        y: Optional[tf.Tensor] = None
    ) -> tf.Tensor:
        return self.var_jepa.get_uncertainty(x, y)


# ============================================================================
# Utility Functions
# ============================================================================

def create_synthetic_data(
    num_samples: int = 10000,
    input_dim: int = 20,
    latent_dim: int = 5,
    num_classes: int = 2
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Create synthetic data with mixture structure."""
    # Generate latents with mixture distribution
    mixture_idx = np.random.binomial(1, 0.5, num_samples)
    
    # Context latents: mixture of two Gaussians
    sx = np.zeros((num_samples, latent_dim))
    for i in range(num_samples):
        if mixture_idx[i] == 0:
            sx[i] = np.random.normal(0, 1.0, latent_dim)
        else:
            sx[i] = np.random.normal(2.0, 1.0, latent_dim)
    
    # Auxiliary latents
    z = np.random.normal(0, 1.0, (num_samples, latent_dim))
    
    # Target latents: correlated with sx and z
    sy = sx + 0.5 * z + np.random.normal(0, 0.3, (num_samples, latent_dim))
    
    # Observations: nonlinear mapping from latents
    w1 = np.random.randn(latent_dim, input_dim)
    w2 = np.random.randn(latent_dim, input_dim)
    x = np.tanh(sx @ w1) + 0.1 * np.random.randn(num_samples, input_dim)
    y = np.tanh(sy @ w2) + 0.1 * np.random.randn(num_samples, input_dim)
    
    # Labels: mixture component for downstream task
    labels = mixture_idx
    
    return (
        tf.convert_to_tensor(x, dtype=tf.float32),
        tf.convert_to_tensor(y, dtype=tf.float32),
        tf.convert_to_tensor(labels, dtype=tf.float32)
    )


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example of how to use Var-JEPA in TensorFlow."""
    
    print("Generating synthetic data...")
    x, y, labels = create_synthetic_data(num_samples=5000, input_dim=20, latent_dim=8)
    
    # Split into train and test
    train_size = 4000
    x_train, x_test = x[:train_size], x[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    labels_train, labels_test = labels[:train_size], labels[train_size:]
    
    # Create Var-JEPA model
    model = VarJEPA(
        input_dim=20,
        latent_dim=8,
        aux_latent_dim=4,
        hidden_dims=[128, 64],
        alpha_rec=1.0,
        alpha_gen=1.0,
        alpha_kl_sx=1.0,
        alpha_kl_z=1.0,
        alpha_kl_sy=1.0,
        reconstruction_std=0.5
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    )
    
    # Create dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.batch(64).shuffle(1000).prefetch(tf.data.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
    
    # Train model
    print("\nTraining Var-JEPA...")
    history = model.fit(
        train_dataset,
        epochs=10,
        validation_data=test_dataset,
        verbose=1
    )
    
    # Get embeddings for downstream task
    print("\nGetting embeddings...")
    embeddings = model.get_embeddings(x_train)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Get uncertainty estimates
    uncertainty = model.get_uncertainty(x_train)
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Uncertainty range: [{tf.reduce_min(uncertainty):.4f}, {tf.reduce_max(uncertainty):.4f}]")
    
    return model, history


if __name__ == "__main__":
    print("Var-JEPA Implementation in TensorFlow")
    print("=" * 50)
    
    # Run example
    model, history = example_usage()
    print("\nModel summary:")
    model.summary()
