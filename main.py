import jax
import jax.numpy as jnp
import numpy as np

from src.constants import BATCH_SIZE, NUM_EPOCHS
from src.data.dataset import create_dataset
from src.model.denoising_diffusion import DenoisingDiffusionModel
from src.training.trainer import init_train_state, train_step


def main():
    print("JAX devices:", jax.devices())
    print("Using device:", jax.devices()[0])

    dataset, tokenizer = create_dataset()
    model = DenoisingDiffusionModel(
        dim_model=512, num_layers=6, num_heads=8, dim_ffnn=2048
    )

    rng = jax.random.PRNGKey(seed=42)
    state = init_train_state(rng, model)

    # Create noise schedule
    timesteps = 1000
    betas = jnp.linspace(0.0001, 0.02, timesteps)

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        num_batches = len(dataset) // BATCH_SIZE

        for i in range(num_batches):
            batch_idx = np.random.randint(0, len(dataset), size=BATCH_SIZE)
            batch = jnp.array(dataset[batch_idx]["input_ids"], dtype=jnp.int32)

            rng, step_rng = jax.random.split(rng)
            state, loss = train_step(model, state, batch, step_rng, betas)
            total_loss += loss

            if i % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}/{num_batches}, Loss: {loss:.4f}"
                )

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
