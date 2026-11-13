from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from classes.vqvae import (
    VQVAEOutput,
)


# ============================================================
# Settings
# ============================================================

@dataclass
class VQVAEObjectiveSettings:

    #
    # Commitment weight
    #

    beta: float = 0.25


    #
    # Whether to include the classical VQ loss
    #
    # Gradient VQ-VAE:
    #
    #     L = Lrec + Lvq + beta Lcommit
    #
    #
    # EMA-VQ:
    #
    #     L = Lrec + beta Lcommit
    #

    use_vq_loss: bool = True

    use_variance_loss: bool = False

    #
    # Reconstruction criterion
    #

    use_mse: bool = True

    use_bce: bool = False



    #
    # Additional regularizers
    #

    gamma_rate: float = 0.0

    gamma_distortion: float = 0.0

    gamma_dead: float = 0.0

    gamma_variance: float = 0.0



# ============================================================
# Output
# ============================================================

@dataclass
class VQVAEObjectiveOutput:

    loss: torch.Tensor

    reconstruction_loss: torch.Tensor

    vq_loss: torch.Tensor

    commitment_loss: torch.Tensor

    variance_loss: torch.Tensor

    distortion: torch.Tensor

    rmse: torch.Tensor

    psnr: torch.Tensor

    perplexity: torch.Tensor

    utilization: torch.Tensor



# ============================================================
# Objective
# ============================================================

class VQVAEObjective(nn.Module):

    """
    VQ-VAE objective.

    Classical gradient VQ-VAE:

        L =
            L_rec
          + L_vq
          + beta L_commit


    EMA Vector Quantization:

        L =
            L_rec
          + beta L_commit


    where the codebook update is handled
    separately through EMA statistics.
    """

    def __init__(
        self,
        settings: VQVAEObjectiveSettings,
    ):

        super().__init__()

        self.settings = settings



    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        output: VQVAEOutput,
    ) -> VQVAEObjectiveOutput:



        #
        # Reconstruction loss
        #

        if self.settings.use_bce:

            reconstruction_loss = F.binary_cross_entropy(

                output.reconstruction,

                x,

            )

        else:

            reconstruction_loss = F.mse_loss(

                output.reconstruction,

                x,

            )



        #
        # Codebook loss
        #

        if self.settings.use_vq_loss:

            vq_loss = output.vq_loss

        else:

            #
            # EMA-VQ:
            #
            # no gradient update of codebook
            #

            vq_loss = torch.zeros_like(
                output.vq_loss
            )



        #
        # Total objective
        #

        loss = (

            reconstruction_loss

            +

            vq_loss

            +

            self.settings.beta
            *
            output.commitment_loss

        )

        if self.settings.use_variance_loss:
            loss += self.settings.gamma_variance * output.variance_loss



        #
        # Distortion statistics
        #

        distortion = F.mse_loss(

            output.reconstruction,

            x,

        )


        rmse = torch.sqrt(

            distortion

        )


        psnr = (

            -10.0

            *

            torch.log10(

                distortion + 1e-10

            )

        )



        return VQVAEObjectiveOutput(

            loss=loss,

            reconstruction_loss=reconstruction_loss,

            vq_loss=vq_loss,

            commitment_loss=output.commitment_loss,

            variance_loss = output.variance_loss,

            distortion=distortion,

            rmse=rmse,

            psnr=psnr,

            perplexity=output.perplexity,

            utilization=output.utilization,

        )