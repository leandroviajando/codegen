from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]
from flax.training import train_state

from src.constants import BATCH_SIZE, LEARNING_RATE, SEQUENCE_LENGTH, VOCAB_SIZE
from src.model.denoising_diffusion import DenoisingDiffusionModel, forward_diffusion


def init_train_state(
    rng: Any, model: DenoisingDiffusionModel
) -> train_state.TrainState:
    """Creates initial training state."""
    params_rng, dropout_rng = jax.random.split(rng)

    variables = model.init(
        {"params": params_rng, "dropout": dropout_rng},
        jnp.ones((BATCH_SIZE, SEQUENCE_LENGTH), dtype=jnp.int32),
        jnp.ones((BATCH_SIZE,), dtype=jnp.float32),
        training=True,
    )

    params = variables["params"]
    tx = optax.adamw(learning_rate=LEARNING_RATE, weight_decay=0.01)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, static_argnums=(0,))
def train_step(
    model: DenoisingDiffusionModel,
    state: train_state.TrainState,
    batch: jnp.ndarray,
    rng: Any,
    betas: jnp.ndarray,
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    """Single training step."""
    rng, noise_rng, time_rng, dropout_rng = jax.random.split(rng, 4)

    def loss_fn(params):
        # Sample random timesteps
        t = jax.random.randint(time_rng, (batch.shape[0],), 0, len(betas))

        # Convert batch to one-hot encoding for diffusion
        batch_one_hot = jax.nn.one_hot(batch, VOCAB_SIZE)

        # Generate noise
        noise = jax.random.normal(noise_rng, batch_one_hot.shape, dtype=jnp.float32)

        # Add noise to one-hot representation
        noisy_batch_one_hot = forward_diffusion(batch_one_hot, t, noise, betas)

        # Get most likely token from noisy distribution (convert back to integers)
        noisy_batch = jnp.argmax(noisy_batch_one_hot, axis=-1)

        # Predict original tokens
        pred = model.apply(
            {"params": params},
            noisy_batch,
            t / len(betas),
            training=True,
            rngs={"dropout": dropout_rng},
        )

        # Calculate loss against original batch
        loss = optax.softmax_cross_entropy_with_integer_labels(pred, batch)
        return jnp.mean(loss)

    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss
