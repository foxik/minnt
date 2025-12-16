# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch

from .mean import Mean
from .. import losses


class CategoricalCrossEntropy(Mean):
    """Categorical cross-entropy metric implementation."""

    def __init__(
        self,
        dim: int = 1,
        *,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        probs: bool = False,
    ) -> None:
        """Create the CategoricalCrossEntropy metric object.

        Parameters:
          dim: If the input has 2 or more dimensions, this value specifies the dimension along which
            the classes are defined. The default is the same behavior as [torch.nn.CrossEntropyLoss][].
          ignore_index: An optional target class value that is ignored during loss computation (equivalent
            to zeroing out sample weights for the corresponding samples). Only applicable for sparse targets;
            when dense targets are used, the default of -100 cannot be overwritten and this parameter is
            ignored. This is the same behavior as [torch.nn.CrossEntropyLoss][].
          label_smoothing: A float in [0.0, 1.0] specifying the label smoothing factor.
            If greater than 0.0, the used ground-truth targets are computed as a mixture
            of the original targets and uniform distribution with weight `1 - label_smoothing`.
          probs: If False, the predictions are assumed to be logits; if `True`, the
            predictions are assumed to be probabilities. Note that gold targets are
            always expected to be probabilities.
        """
        super().__init__()
        self._cce_loss = losses.CategoricalCrossEntropy(
            dim=dim, ignore_index=ignore_index, label_smoothing=label_smoothing, probs=probs, reduction="none",
        )
        self._dim = dim
        self._ignore_index = ignore_index

    def update(
        self, y: torch.Tensor, y_true: torch.Tensor | None = None, sample_weights: torch.Tensor | None = None,
    ) -> None:
        """Update the accumulated categorical cross-entropy by introducing new values.

        Optional sample weight might be provided; if not, all values are weighted with 1.

        Parameters:
          y: The predicted outputs, either logits or probabilities (depending on the `probs` parameter).
            If they have 2 or more dimensions, the class dimension is specified by the `dim` parameter.
          y_true: The ground-truth targets in two possible formats:

            - The gold targets might be "sparse" class indices. In this case, their shape has to be
              exactly the same as `y` with the class dimension removed.
            - The gold targets might be full "dense" probability distributions. In this case, their
              shape has to be exactly the same as `y`.
          sample_weights: Optional sample weights. If provided, their shape must be broadcastable
            to a prefix of a shape of `y` with the class dimension removed, and the loss for each sample
            is weighted accordingly.
        """
        y_shape, y_true_shape = y.shape, y_true.shape

        dense = len(y_true_shape) == len(y_shape)
        if not dense:
            # For sparse targets, we need to mask out the ignored classes manually.
            ignore_index_weights = (y_true != self._ignore_index).to(dtype=torch.float32)
            if sample_weights is None:
                sample_weights = ignore_index_weights
            else:
                dim = self._dim if len(y_shape) > 1 else 0
                y_wo_class_dim_shape = y_shape[:dim] + y_shape[dim + 1:]

                while sample_weights.dim() < len(y_wo_class_dim_shape):
                    sample_weights = sample_weights.unsqueeze(dim=-1)
                if sample_weights.shape != y_wo_class_dim_shape:
                    sample_weights = sample_weights.expand(y_wo_class_dim_shape)

                sample_weights = sample_weights.to(dtype=torch.float32) * ignore_index_weights

        super().update(self._cce_loss(y, y_true), sample_weights=sample_weights)

    def compute(self) -> torch.Tensor:
        return self._total / self._count

    def reset(self):
        self._total.zero_()
        self._count.zero_()
