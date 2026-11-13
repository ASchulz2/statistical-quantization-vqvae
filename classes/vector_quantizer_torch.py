from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# ============================================================
# Settings
# ============================================================

@dataclass
class VectorQuantizerTorchSettings:

    num_embeddings: int = 32

    embedding_dim: int = 64

    commitment_cost: float = 0.25

    dtype: torch.dtype = torch.float32

    device: str = "cpu"

    use_ema: bool = False

    ema_decay: float = 0.99


    #
    # Statistical codebook
    #

    use_variance: bool = False

    sample_latent: bool = True

    variance_init: float = 0.01



# ============================================================
# Output
# ============================================================

@dataclass
class VectorQuantizerOutput:

    z_q: torch.Tensor

    indices: torch.Tensor

    vq_loss: torch.Tensor

    variance_loss: torch.Tensor

    commitment_loss: torch.Tensor

    batch_entropy: torch.Tensor

    batch_rate: torch.Tensor

    batch_perplexity: torch.Tensor

    batch_utilization: torch.Tensor



# ============================================================
# Vector Quantizer
# ============================================================

class VectorQuantizerTorch(nn.Module):


    def __init__(
        self,
        settings: VectorQuantizerTorchSettings,
    ):

        super().__init__()

        self.settings = settings


        self.embedding_mean = nn.Embedding(
            settings.num_embeddings,
            settings.embedding_dim,
            dtype=settings.dtype,
            device=settings.device,
        )

        self.embedding_logvar = nn.Embedding(
            settings.num_embeddings,
            settings.embedding_dim,
            dtype=settings.dtype,
            device=settings.device,
        )


        nn.init.uniform_(
            self.embedding_mean.weight,
            -1.0,
            1.0,
        )


        nn.init.constant_(
            self.embedding_logvar.weight,
            np.log(settings.variance_init),
        )


        #
        # EMA statistics
        #

        self.register_buffer(
            "ema_cluster_size",
            torch.zeros(
                settings.num_embeddings,
                dtype=settings.dtype,
                device=settings.device,
            )
        )


        self.register_buffer(
            "ema_weight",
            torch.zeros(
                settings.num_embeddings,
                settings.embedding_dim,
                dtype=settings.dtype,
                device=settings.device,
            )
        )

        self.register_buffer(
            "ema_squared_weight",
            torch.zeros(
                settings.num_embeddings,
                settings.embedding_dim,
                dtype=settings.dtype,
                device=settings.device,
            ),
        )


        if settings.use_ema:

            self.embedding_mean.weight.requires_grad_(False)
            self.embedding_logvar.weight.requires_grad_(False)



    # ========================================================
    # Assignment
    # ========================================================

    @torch.no_grad()
    def assign(
        self,
        z_e: torch.Tensor,
    ) -> torch.Tensor:


        original_shape = z_e.shape


        flat = z_e.reshape(

            -1,

            self.settings.embedding_dim,

        )


        distances = (

            torch.sum(
                flat ** 2,
                dim=1,
                keepdim=True,
            )

            +

            torch.sum(
                self.embedding_mean.weight ** 2,
                dim=1,
            )

            -

            2.0
            *
            flat
            @
            self.embedding_mean.weight.t()

        )


        indices = torch.argmin(

            distances,

            dim=1,

        )


        return indices.reshape(

            original_shape[:-1]

        )



    # ========================================================
    # Quantization
    # ========================================================

    def quantize(
        self,
        indices: torch.Tensor,
    ):
        mu = self.embedding_mean(indices)

        if self.settings.use_variance:

            logvar = self.embedding_logvar(indices)

            std = torch.exp(
                0.5 * logvar
            )

            if self.settings.sample_latent:

                noise = torch.randn_like(mu)

                z_q = mu + std * noise

            else:

                z_q = mu

        else:

            z_q = mu

        return z_q



    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        z_e: torch.Tensor,
    ):


        single_sample = False


        if z_e.ndim == 1:

            z_e = z_e.unsqueeze(0)

            single_sample = True



        original_shape = z_e.shape


        #
        # Assignment
        #

        indices = self.assign(
            z_e
        )


        #
        # Lookup
        #

        z_q = self.quantize(
            indices
        )


        #
        # Losses
        #

        if self.settings.use_ema:

            #
            # No gradient codebook loss
            #

            vq_loss = torch.zeros_like(
                z_e.mean()
            )

            variance_loss = torch.zeros_like(
                torch.mean(
                    torch.exp(
                        self.embedding_logvar.weight
                    )
                )
            )


        else:

            vq_loss = F.mse_loss(

                z_q,

                z_e.detach(),

            )

            variance_loss = torch.mean(
                torch.exp(
                    self.embedding_logvar.weight
                )
            )

        commitment_loss = F.mse_loss(

            z_q.detach(),

            z_e,

        )

        
        #
        # EMA update
        #

        if (

            self.settings.use_ema

            and

            self.training

        ):

            self.ema_update(

                z_e,

                indices,

            )



        #
        # Straight-through estimator
        #

        z_q = (

            z_e

            +

            (z_q - z_e).detach()

        )



        entropy, rate, perplexity, utilization = (

            self._statistics_from_indices(

                indices

            )

        )



        z_q = z_q.reshape(
            original_shape
        )


        indices = indices.reshape(

            original_shape[:-1]

        )



        if single_sample:

            z_q = z_q.squeeze(0)

            indices = indices.squeeze(0)



        return VectorQuantizerOutput(

            z_q=z_q,

            indices=indices,

            vq_loss=vq_loss,

            variance_loss=variance_loss,

            commitment_loss=commitment_loss,

            batch_entropy=entropy,

            batch_rate=rate,

            batch_perplexity=perplexity,

            batch_utilization=utilization,

        )



    # ========================================================
    # EMA Update
    # ========================================================

    @torch.no_grad()
    def ema_update(
        self,
        z_e: torch.Tensor,
        indices: torch.Tensor,
    ):
        """
        Update codebook mean and variance using EMA.

        For every codeword k we maintain exponential moving
        estimates of

            E[z | k]

        and

            E[z^2 | k].

        The variance is then obtained from

            Var[z | k] = E[z^2 | k] - E[z | k]^2.
        """

        counts, sums, squared_sums = self.ema_statistics(
            z_e,
            indices,
        )

        decay = self.settings.ema_decay

        #
        # EMA cluster counts
        #

        self.ema_cluster_size.mul_(decay).add_(
            counts,
            alpha=1.0 - decay,
        )

        #
        # EMA first moment
        #

        self.ema_weight.mul_(decay).add_(
            sums,
            alpha=1.0 - decay,
        )

        #
        # EMA second moment
        #

        self.ema_squared_weight.mul_(decay).add_(
            squared_sums,
            alpha=1.0 - decay,
        )

        #
        # Avoid division by zero
        #

        n = self.ema_cluster_size + 1e-5

        #
        # Mean
        #

        mean = (
            self.ema_weight
            /
            n.unsqueeze(1)
        )

        #
        # Second moment
        #

        second_moment = (
            self.ema_squared_weight
            /
            n.unsqueeze(1)
        )

        #
        # Variance
        #
        # Var[X] = E[X²] - E[X]²
        #

        variance = (
            second_moment
            -
            mean.pow(2)
        )

        #
        # Numerical protection
        #

        variance = torch.clamp(
            variance,
            min=1e-6,
        )

        #
        # Update statistical codebook
        #

        self.embedding_mean.weight.copy_(
            mean
        )

        self.embedding_logvar.weight.copy_(
            torch.log(variance)
        )


    # ========================================================
    # EMA statistics
    # ========================================================

    @torch.no_grad()
    def ema_statistics(
        self,
        z_e: torch.Tensor,
        indices: torch.Tensor,
    ):
        """
        Compute minibatch sufficient statistics for EMA.

        Returns

            counts
            sums
            squared_sums

        for every codeword.
        """

        flat_z = z_e.reshape(
            -1,
            self.settings.embedding_dim,
        )

        flat_indices = indices.reshape(-1)

        #
        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------
        #

        counts = torch.zeros(
            self.settings.num_embeddings,
            device=z_e.device,
            dtype=z_e.dtype,
        )

        counts.scatter_add_(
            0,
            flat_indices,
            torch.ones_like(
                flat_indices,
                dtype=z_e.dtype,
            ),
        )

        #
        # ----------------------------------------------------
        # First moment
        # ----------------------------------------------------
        #

        sums = torch.zeros(
            self.settings.num_embeddings,
            self.settings.embedding_dim,
            device=z_e.device,
            dtype=z_e.dtype,
        )

        sums.index_add_(
            0,
            flat_indices,
            flat_z,
        )

        #
        # ----------------------------------------------------
        # Second moment
        # ----------------------------------------------------
        #

        squared_sums = torch.zeros(
            self.settings.num_embeddings,
            self.settings.embedding_dim,
            device=z_e.device,
            dtype=z_e.dtype,
        )

        squared_sums.index_add_(
            0,
            flat_indices,
            flat_z.pow(2),
        )

        return (
            counts,
            sums,
            squared_sums,
        )


    # ========================================================
    # Statistics
    # ========================================================

    def _statistics_from_indices(
        self,
        indices,
    ):

        counts = torch.bincount(

            indices.flatten(),

            minlength=self.settings.num_embeddings,

        ).float()



        probs = counts / counts.sum()


        mask = probs > 0



        entropy = -torch.sum(

            probs[mask]

            *

            torch.log2(
                probs[mask]
            )

        )


        rate = entropy


        perplexity = torch.pow(

            2.0,

            entropy,

        )


        utilization = (

            counts > 0

        ).float().mean()



        return (

            entropy,

            rate,

            perplexity,

            utilization,

        )



    # ========================================================
    # Dataset statistics
    # ========================================================

    @torch.no_grad()
    def compute_statistics(
        self,
        indices,
    ):

        entropy, rate, perplexity, utilization = (
            self._statistics_from_indices(indices)
        )


        counts = torch.bincount(

            indices.flatten(),

            minlength=self.settings.num_embeddings,

        )


        return {

            "entropy":
                entropy.item(),

            "rate":
                rate.item(),

            "perplexity":
                perplexity.item(),

            "utilization":
                utilization.item(),

            "active_codewords":
                int((counts > 0).sum()),

            "dead_codewords":
                int((counts == 0).sum()),

            "histogram":
                counts.cpu(),

        }



    # ========================================================
    # Initialization
    # ========================================================

    @torch.no_grad()
    def initialize_from_samples(
        self,
        samples: torch.Tensor,
    ):
        """
        Initialize codebook using K-Means.

        For EMA-VQ, also initialize the EMA statistics
        so that the first update does not destroy
        the K-Means initialization.

        Parameters
        ----------
        samples:
            Encoder outputs.

            Shape:
                (N, embedding_dim)

        """

        from sklearn.cluster import KMeans


        #
        # Move to CPU for sklearn
        #

        samples_np = (
            samples
            .detach()
            .cpu()
            .numpy()
        )


        #
        # K-Means
        #

        kmeans = KMeans(

            n_clusters=self.settings.num_embeddings,

            n_init="auto",

        )


        kmeans.fit(
            samples_np
        )


        #
        # Centroids
        #

        centroids = torch.tensor(

            kmeans.cluster_centers_,

            dtype=self.embedding_mean.weight.dtype,

            device=self.embedding_mean.weight.device,

        )


        #
        # Initialize embedding
        #

        self.embedding_mean.weight.copy_(
            centroids
        )


        #
        # Initialize EMA buffers
        #

        if self.settings.use_ema:


            labels = torch.tensor(

                kmeans.labels_,

                dtype=torch.long,

                device=self.embedding_mean.weight.device,

            )


            counts = torch.bincount(

                labels,

                minlength=self.settings.num_embeddings,

            ).float()


            #
            # Avoid zero divisions
            #

            counts = torch.clamp(

                counts,

                min=1.0,

            )


            #
            # EMA cluster size
            #

            self.ema_cluster_size.copy_(
                counts
            )


            #
            # EMA accumulated weights
            #
            # m_j = N_j * e_j
            #

            self.ema_weight.copy_(

                counts.unsqueeze(1)

                *

                centroids

            )