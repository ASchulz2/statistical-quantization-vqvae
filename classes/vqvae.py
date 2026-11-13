from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from classes.encoder import (
    Encoder,
    EncoderSettings,
)

from classes.decoder import (
    Decoder,
    DecoderSettings,
)

from classes.vector_quantizer_torch import (
    VectorQuantizerTorch,
    VectorQuantizerTorchSettings,
    VectorQuantizerOutput,
)


# ============================================================
# Settings
# ============================================================

@dataclass
class VQVAEOutput:
    """
    Output of the complete VQ-VAE.
    """

    reconstruction: torch.Tensor

    z_e: torch.Tensor

    z_q: torch.Tensor

    indices: torch.Tensor

    vq_loss: torch.Tensor

    commitment_loss: torch.Tensor

    variance_loss: torch.Tensor

    perplexity: torch.Tensor

    utilization: torch.Tensor


@dataclass
class VQVAESettings:

    encoder: EncoderSettings

    decoder: DecoderSettings

    quantizer: VectorQuantizerTorchSettings


# ============================================================
# Model
# ============================================================

class VQVAE(nn.Module):

    """
    Vector Quantized Variational Autoencoder.

    Pipeline

        x
            ↓
        Encoder
            ↓
        z_e
            ↓
        VectorQuantizer
            ↓
        z_q
            ↓
        Decoder
            ↓
        reconstruction
    """

    def __init__(
        self,
        settings: VQVAESettings,
    ):

        super().__init__()

        self.settings = settings

        self.encoder = Encoder(
            settings.encoder
        )

        self.quantizer = VectorQuantizerTorch(
            settings.quantizer
        )

        self.decoder = Decoder(
            settings.decoder
        )

    # --------------------------------------------------------
    # Convenience methods
    # --------------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.encoder(x)


    def quantize(
        self,
        z_e: torch.Tensor,
    ) -> VectorQuantizerOutput:

        return self.quantizer(z_e)


    def decode(
        self,
        z_q: torch.Tensor,
    ) -> torch.Tensor:

        return self.decoder(z_q)


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> VQVAEOutput:

        #
        # Encoder
        #

        z_e = self.encoder(x)

        #
        # Quantization
        #

        quantized = self.quantize(z_e)

        #
        # Decoder
        #

        reconstruction = self.decode(
            quantized.z_q
        )

        #
        # Assemble output
        #

        return VQVAEOutput(

            reconstruction=reconstruction,

            z_e=z_e,

            z_q=quantized.z_q,

            indices=quantized.indices,

            vq_loss=quantized.vq_loss,

            commitment_loss=quantized.commitment_loss,

            variance_loss=quantized.variance_loss,

            perplexity=quantized.batch_perplexity,

            utilization=quantized.batch_utilization,

        )