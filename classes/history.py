from __future__ import annotations

from dataclasses import dataclass, field

from typing import Dict, List, Any


# ============================================================
# Settings
# ============================================================

@dataclass
class HistorySettings:

    verbose: bool = False



# ============================================================
# History
# ============================================================

class History:

    """
    Generic metric recorder.

    Example:

        history.update(
            {
                "loss": 0.23,
                "accuracy": 0.91,
            }
        )


        history.metrics["loss"]

        -> [0.23, ...]

    """

    def __init__(
        self,
        settings: HistorySettings | None = None,
    ):

        if settings is None:

            settings = HistorySettings()


        self.settings = settings


        self.metrics: Dict[
            str,
            List[Any]
        ] = {}



    # --------------------------------------------------------
    # Update
    # --------------------------------------------------------

    def update(
        self,
        values: Dict[str, Any],
    ):

        """
        Append one training step/epoch.
        """

        for key, value in values.items():

            if key not in self.metrics:

                self.metrics[key] = []


            #
            # Convert tensors/numpy scalars
            #

            if hasattr(
                value,
                "item"
            ):

                value = value.item()


            self.metrics[key].append(
                value
            )


        if self.settings.verbose:

            print(values)



    # --------------------------------------------------------
    # Access
    # --------------------------------------------------------

    def get(
        self,
        name: str,
    ) -> List[Any]:
        return self.metrics[name]


    def keys(self):

        return self.metrics.keys()



    def latest(
        self,
    ) -> Dict[str, Any]:

        return {

            key: values[-1]

            for key, values

            in self.metrics.items()

            if len(values) > 0

        }



    # --------------------------------------------------------
    # Utilities
    # --------------------------------------------------------

    def reset(
        self,
    ):

        self.metrics.clear()



    def as_dict(
        self,
    ):

        return self.metrics



    def __len__(
        self,
    ):

        if len(self.metrics) == 0:

            return 0


        first = next(
            iter(self.metrics.values())
        )


        return len(first)



    def __repr__(
        self,
    ):

        return (

            f"History("
            f"epochs={len(self)}, "
            f"metrics={list(self.metrics.keys())}"
            f")"

        )