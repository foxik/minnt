# This file is part of Minnt <http://github.com/ufal/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch

from .generic_decay import GenericDecay


class CosineDecay(GenericDecay):
    """A cosine decay learning rate scheduler with optional linear warmup.

    This scheduler is a convenience wrapper around [minnt.schedulers.GenericDecay][]
    with the `decay` parameter set to `"cosine"`.
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        *,
        final_decay: float = 0.0,
        warmup: int | float = 0,
        last_epoch: int = -1,
        warn_about_exceeding_steps: bool = True,
    ) -> None:
        """Creates a new CosineDecay scheduler instance.

        Please refer to the documentation of [minnt.schedulers.GenericDecay][] for details.
        """
        super().__init__(
            optimizer,
            total_steps,
            decay="cosine",
            final_decay=final_decay,
            warmup=warmup,
            last_epoch=last_epoch,
            warn_about_exceeding_steps=warn_about_exceeding_steps,
        )
