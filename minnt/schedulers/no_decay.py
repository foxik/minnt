# This file is part of Minnt <http://github.com/ufal/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch

from .generic_decay import GenericDecay


class NoDecay(GenericDecay):
    """A non-decaying learning rate scheduler with optional linear warmup.

    This scheduler is a convenience wrapper around [minnt.schedulers.GenericDecay][]
    with the `decay` parameter set to `"none"` and `total_steps` being optional.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int | None = None,
        *,
        warmup: int | float = 0,
        last_epoch: int = -1,
    ) -> None:
        """Creates a new NoDecay scheduler instance.

        Parameters:
          optimizer: The optimizer for which to schedule the learning rate.
          total_steps: An optional number of training steps. Useful only when `warmup` is
            specified as a fraction.
          warmup: Specifies the warmup phase. If a number smaller than 1 is given,
            it is treated as a fraction of `total_steps`; otherwise, it is treated as
            an absolute number of steps. Default is 0 (no warmup).
          last_epoch: The index of the last epoch when resuming training. Default is -1.
        """
        if total_steps is None:
            if 0 < warmup < 1:
                raise ValueError("If total_steps is None, warmup must be zero or a absolute number of steps.")
            total_steps = warmup

        super().__init__(
            optimizer,
            total_steps,
            decay="none",
            warmup=warmup,
            last_epoch=last_epoch,
            warn_about_exceeding_steps=False,
        )
