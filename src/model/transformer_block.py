import flax.linen as nn


class TransformerBlock(nn.Module):
    dim_model: int
    num_heads: int
    dim_ffnn: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, training: bool = True):
        self_attention_output = nn.MultiHeadAttention(
            num_heads=self.num_heads,
            qkv_features=self.dim_model,
            dropout_rate=self.dropout_rate,
            deterministic=not training,
        )(x, x)
        x = nn.LayerNorm()(x + self_attention_output)

        ffnn_output = nn.Sequential(
            [
                nn.Dense(self.dim_ffnn),
                nn.gelu,
                nn.Dropout(rate=self.dropout_rate, deterministic=not training),
                nn.Dense(self.dim_model),
                nn.Dropout(rate=self.dropout_rate, deterministic=not training),
            ]
        )(x)

        return nn.LayerNorm()(x + ffnn_output)
