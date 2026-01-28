# This file is part of Minnt <http://github.com/ufal/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import math
from typing import Literal
import warnings

import torch


class GenericDecay(torch.optim.lr_scheduler.LambdaLR):
    """A generic cosine/linear decay learning rate scheduler with optional linear warmup.

    If specified, this scheduler first linearly increases the learning rate from 0 to the initial
    learning rate during an optional warmup phase. Then it decreases the learning rate according to a
    specified decay strategy (cosine/linear/none) to a final fraction of the initial
    learning rate defined by `final_decay` (default 0.0, i.e., decays to 0) over the remaining
    training steps.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        decay: Literal["cosine", "linear", "none"],
        *,
        final_decay: float = 0.0,
        warmup: int | float = 0,
        last_epoch: int = -1,
        warn_about_exceeding_steps: bool = True,
    ) -> None:
        r"""Creates a new GenericDecay scheduler instance.

        Parameters:
          optimizer: The optimizer for which to schedule the learning rate.
          total_steps: The total number of training steps in all epochs, including
            the optional warmup phase.
          decay: The decay strategy to use after the warmup phase, one of:

            - `"cosine"`: cosine decay, computed as

            $$\operatorname{decay\_factor}(t) = \mathit{final\_decay} + (1 - \mathit{final\_decay}) \cdot
              \bigg(\frac{1 + \cos(\pi \cdot t / \mathit{decay\_steps})}{2}\bigg);$$

            - `"linear"`: linear decay, computed as

            $$\operatorname{decay\_factor}(t) = \mathit{final\_decay} + (1 - \mathit{final\_decay}) \cdot
              \bigg(1 - \frac{t}{\mathit{decay\_steps}}\bigg);$$

            - `"none"`: no decay, i.e., keeping the initial learning rate.
          final_decay: The final learning rate as a fraction of the initial
            learning rate after decay. Default is 0.0 (decays to 0).
          warmup: Specifies the warmup phase. If a number smaller than 1 is given,
            it is treated as a fraction of `total_steps`; otherwise, it is treated as
            an absolute number of steps. Default is 0 (no warmup).
          last_epoch: The index of the last epoch when resuming training. Default is -1.
          warn_about_exceeding_steps: Whether to raise a [RuntimeWarning][] if the number of steps
            exceeds the `total_steps`.
        """
        assert decay in ("cosine", "linear", "none"), f"Unknown decay strategy: {decay}"
        self._decay = decay

        self._warmup_steps = int(warmup * total_steps if warmup < 1 else warmup)
        self._decay_steps = total_steps - self._warmup_steps
        assert self._warmup_steps <= total_steps, "Warmup steps must be at most the total steps"

        self._final_decay = final_decay
        self._warn_about_exceeding_steps = warn_about_exceeding_steps
        super().__init__(optimizer, self.compute_decay_factor, last_epoch)

    def compute_decay_factor(self, step: int) -> float:
        if step < self._warmup_steps:
            return step / self._warmup_steps

        if step > self._warmup_steps + self._decay_steps:
            if self._warn_about_exceeding_steps:
                warnings.warn(
                    f"Step {step} exceeds total steps ({self._warmup_steps + self._decay_steps}). "
                    "The final learning rate will be kept.", RuntimeWarning)
            step = self._warmup_steps + self._decay_steps

        if self._decay == "none" or self._decay_steps == 0:
            decay = 1.0
        elif self._decay == "cosine":
            decay = 0.5 * (1 + math.cos(math.pi * ((step - self._warmup_steps) / self._decay_steps)))
        elif self._decay == "linear":
            decay = 1.0 - (step - self._warmup_steps) / self._decay_steps

        if self._final_decay:
            decay = self._final_decay + (1 - self._final_decay) * decay

        return decay
