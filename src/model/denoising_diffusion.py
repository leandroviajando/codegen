import flax.linen as nn
import jax
import jax.numpy as jnp

from src.constants import SEQUENCE_LENGTH, VOCAB_SIZE
from src.model.transformer_block import TransformerBlock


def forward_diffusion(
    x0: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray, betas: jnp.ndarray
) -> jnp.ndarray:
    """Forward diffusion process."""
    alpha = 1.0 - betas
    alpha_bar = jnp.cumprod(alpha)
    alpha_bar_t = alpha_bar[t]

    mean = jnp.sqrt(alpha_bar_t)[:, None, None] * x0
    var = 1.0 - alpha_bar_t[:, None, None]
    return mean + jnp.sqrt(var) * noise


class DenoisingDiffusionModel(nn.Module):
    """Transformer-based denoising diffusion model for code generation.

    Learns to denoise data that has been corrupted by the forward diffusion process."""

    dim_model: int
    num_layers: int
    num_heads: int
    dim_ffnn: int
    vocab_size: int = VOCAB_SIZE
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, t, training: bool = True):
        # Time embedding
        t = t[:, None]  # (batch_size, 1)
        t_embedding = nn.Dense(self.dim_model)(jnp.concatenate([t, t**2], axis=-1))
        t_embedding = jax.numpy.sin(t_embedding)

        token_embedding = nn.Embed(
            num_embeddings=self.vocab_size, features=self.dim_model
        )(x)

        # Add time embedding to each position
        x = token_embedding + t_embedding[:, None, :]

        # Position embedding
        position_ids = jnp.arange(x.shape[1])[None, :]
        position_embedding = nn.Embed(
            num_embeddings=SEQUENCE_LENGTH, features=self.dim_model
        )(position_ids)
        x = x + position_embedding

        # Transformer layers
        for _ in range(self.num_layers):
            x = TransformerBlock(
                dim_model=self.dim_model,
                num_heads=self.num_heads,
                dim_ffnn=self.dim_ffnn,
                dropout_rate=self.dropout_rate,
            )(x, training=training)

        # Project back to vocabulary size
        return nn.Dense(self.vocab_size)(x)
