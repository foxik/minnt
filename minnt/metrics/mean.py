# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch

from ..metric import Metric


class Mean(Metric):
    """A class tracking the (optionally weighted) mean of given values."""

    def __init__(self) -> None:
        """Create the Mean metric object."""
        super().__init__()
        self.register_buffer("_total", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        self.register_buffer("_count", torch.tensor(0.0, dtype=torch.float32), persistent=False)

    def update(
        self, y: torch.Tensor, y_true: torch.Tensor | None = None, sample_weights: torch.Tensor | None = None,
    ) -> None:
        """Update the accumulated mean by introducing new values.

        Optional sample weight might be provided; if not, all values are weighted with 1.

        Parameters:
          y: The values to average.
          y_true: This parameter is present for [minnt.Metric][] compatibility, but must be `None` for this metric.
          sample_weights: Optional sample weights. Their shape must be broadcastable to a prefix of the shape of `y`.
        """
        assert y_true is None, "The y_true parameter must be None for the Mean metric."

        if sample_weights is not None:
            while sample_weights.dim() < y.dim():
                sample_weights = sample_weights.unsqueeze(dim=-1)
            if sample_weights.shape != y.shape:
                sample_weights = sample_weights.expand(y.shape)

        self._total.add_(torch.sum(y * sample_weights) if sample_weights is not None else torch.sum(y))
        self._count.add_(torch.sum(sample_weights) if sample_weights is not None else y.numel())

    def compute(self) -> torch.Tensor:
        return self._total / self._count

    def reset(self):
        self._total.zero_()
        self._count.zero_()
