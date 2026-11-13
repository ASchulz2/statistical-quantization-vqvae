from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VQStatisticsSettings:
    """
    Statistics for analysing vector-quantizer codebooks.
    """

    eps: float = 1e-12

    dtype: np.dtype = np.float32  # type: ignore


class VQStatistics:

    def __init__(
        self,
        settings: VQStatisticsSettings,
    ):
        self.settings = settings

    # ----------------------------------------------------------
    # Histogram
    # ----------------------------------------------------------

    def histogram(
        self,
        indices: np.ndarray,
        n_codewords: int,
    ) -> np.ndarray:

        return np.bincount(
            indices,
            minlength=n_codewords,
        )

    # ----------------------------------------------------------
    # Assignment probabilities
    # ----------------------------------------------------------

    def probabilities(
        self,
        indices: np.ndarray,
        n_codewords: int,
    ) -> np.ndarray:

        hist = self.histogram(
            indices,
            n_codewords,
        ).astype(np.float64)

        return hist / np.sum(hist)

    # ----------------------------------------------------------
    # Entropy
    # ----------------------------------------------------------

    def entropy(
        self,
        indices: np.ndarray,
        n_codewords: int,
    ) -> float:

        p = self.probabilities(
            indices,
            n_codewords,
        )

        p = p[p > 0]

        return float(
            -np.sum(
                p * np.log2(p)
            )
        )

    # ----------------------------------------------------------
    # Perplexity
    # ----------------------------------------------------------

    def perplexity(
        self,
        indices: np.ndarray,
        n_codewords: int,
    ) -> float:

        return float(
            2.0 ** self.entropy(
                indices,
                n_codewords,
            )
        )

    # ----------------------------------------------------------
    # Utilization
    # ----------------------------------------------------------

    def utilization(
        self,
        indices: np.ndarray,
        n_codewords: int,
    ) -> float:

        return float(
            np.count_nonzero(
                self.histogram(
                    indices,
                    n_codewords,
                )
            )
            /
            n_codewords
        )

    # ----------------------------------------------------------
    # Dead codewords
    # ----------------------------------------------------------

    def dead_codewords(
        self,
        indices: np.ndarray,
        n_codewords: int,
    ) -> np.ndarray:

        hist = self.histogram(
            indices,
            n_codewords,
        )

        return np.where(hist == 0)[0]

    # ----------------------------------------------------------
    # Mean occupancy
    # ----------------------------------------------------------

    def mean_occupancy(
        self,
        indices: np.ndarray,
        n_codewords: int,
    ) -> float:

        hist = self.histogram(
            indices,
            n_codewords,
        )

        active = hist[hist > 0]

        if len(active) == 0:
            return 0.0

        return float(
            np.mean(active)
        )

    # ----------------------------------------------------------
    # Occupancy variance
    # ----------------------------------------------------------

    def occupancy_variance(
        self,
        indices: np.ndarray,
        n_codewords: int,
    ) -> float:

        hist = self.histogram(
            indices,
            n_codewords,
        )

        return float(
            np.var(hist)
        )

    # ----------------------------------------------------------
    # Complete analysis
    # ----------------------------------------------------------

    def analyze(
        self,
        indices: np.ndarray,
        n_codewords: int,
    ) -> dict:

        dead = self.dead_codewords(
            indices,
            n_codewords,
        )

        return {

            "entropy":
                self.entropy(
                    indices,
                    n_codewords,
                ),

            "perplexity":
                self.perplexity(
                    indices,
                    n_codewords,
                ),

            "utilization":
                self.utilization(
                    indices,
                    n_codewords,
                ),

            "mean_occupancy":
                self.mean_occupancy(
                    indices,
                    n_codewords,
                ),

            "occupancy_variance":
                self.occupancy_variance(
                    indices,
                    n_codewords,
                ),

            "dead_codewords":
                len(dead),

            "dead_indices":
                dead,
        }