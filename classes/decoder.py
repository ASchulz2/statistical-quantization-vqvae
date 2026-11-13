from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ============================================================
# Settings
# ============================================================

@dataclass
class DecoderSettings:

    latent_dim: int = 64

    hidden_dims: tuple[int, ...] = (
        256,
        512,
    )

    output_dim: int = 784

    activation: type[nn.Module] = nn.ReLU

    output_activation: type[nn.Module] = nn.Sigmoid

    dtype: torch.dtype = torch.float32

    device: str = "cpu"


# ============================================================
# Decoder
# ============================================================

class Decoder(nn.Module):

    """
    Feed-forward decoder for the VQ-VAE.

    Maps

        z_q
        ↓
        x_hat

    using a configurable multilayer perceptron.
    """

    def __init__(
        self,
        settings: DecoderSettings,
    ):

        super().__init__()

        self.settings = settings

        self.device = torch.device(
            settings.device
        )

        layers: list[nn.Module] = []

        in_features = settings.latent_dim

        # ----------------------------------------------------
        # Hidden layers
        # ----------------------------------------------------

        for hidden in settings.hidden_dims:

            layers.append(

                nn.Linear(

                    in_features,

                    hidden,

                    dtype=settings.dtype,

                )

            )

            layers.append(

                settings.activation()

            )

            in_features = hidden

        # ----------------------------------------------------
        # Output layer
        # ----------------------------------------------------

        layers.append(

            nn.Linear(

                in_features,

                settings.output_dim,

                dtype=settings.dtype,

            )

        )

        layers.append(

            settings.output_activation()

        )

        self.network = nn.Sequential(
            *layers
        )

        self._initialize_weights()

        self.to(self.device)


    # --------------------------------------------------------
    # Weight initialization
    # --------------------------------------------------------

    def _initialize_weights(self):

        """
        Initialize decoder weights.

        Hidden layers:
            Kaiming initialization.

        Output layer:
            Xavier initialization,
            which works well with Sigmoid outputs.
        """

        linear_layers = [

            module

            for module in self.network

            if isinstance(module, nn.Linear)

        ]

        for layer in linear_layers[:-1]:

            nn.init.kaiming_uniform_(

                layer.weight,

                nonlinearity="relu",

            )

            if layer.bias is not None:

                nn.init.zeros_(

                    layer.bias

                )

        output = linear_layers[-1]

        nn.init.xavier_uniform_(

            output.weight

        )

        if output.bias is not None:

            nn.init.zeros_(

                output.bias

            )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:

        z = z.to(

            device=self.device,

            dtype=self.settings.dtype,

        )

        return self.network(z)