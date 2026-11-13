from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


# ============================================================
# Settings
# ============================================================

@dataclass
class EncoderSettings:

    input_dim: int = 784

    hidden_dims: tuple[int, ...] = (
        512,
        256,
    )

    latent_dim: int = 64

    activation: type[nn.Module] = nn.ReLU

    dtype: torch.dtype = torch.float32

    device: str = "cpu"


# ============================================================
# Encoder
# ============================================================

class Encoder(nn.Module):

    """
    Feed-forward encoder for the VQ-VAE.

    Maps

        x
        ↓
        z_e

    using a configurable multilayer perceptron.
    """

    def __init__(
        self,
        settings: EncoderSettings,
    ):

        super().__init__()

        self.settings = settings

        self.device = torch.device(
            settings.device
        )

        layers: list[nn.Module] = []

        in_features = settings.input_dim

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
        # Latent layer
        # ----------------------------------------------------

        layers.append(

            nn.Linear(

                in_features,

                settings.latent_dim,

                dtype=settings.dtype,

            )

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
        Initialize all linear layers.

        Hidden layers use Kaiming initialization,
        which is well suited for ReLU activations.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.kaiming_uniform_(

                    module.weight,

                    nonlinearity="relu",

                )

                if module.bias is not None:

                    nn.init.zeros_(

                        module.bias

                    )


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = x.to(

            device=self.device,

            dtype=self.settings.dtype,

        )

        return self.network(x)