# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Literal, Protocol

import torch

from .type_aliases import Logs


class Callback(Protocol):
    def __call__(self, module: "TrainableModule", epoch: int, logs: Logs) -> Literal["stop_training"] | None:
        """Represents a callback called after every training epoch.

        If the callback returns [TrainableModule.STOP_TRAINING][minnt.TrainableModule.STOP_TRAINING],
        the training stops.

        Parameters:
          module: the module being trained
          epoch: the current epoch number (one-based)
          logs: a dictionary of logs, newly computed metric or losses should be added here

        **Returns:**

          - [`module.STOP_TRAINING`][minnt.TrainableModule.STOP_TRAINING] to stop the training,
          - [`None`][None] to continue.
        """
        ...


class KeepPrevious:
    pass
keep_previous = KeepPrevious()  # noqa: E305


class TrainableModule(torch.nn.Module):
    STOP_TRAINING: Literal["stop_training"] = "stop_training"
    """A constant returned by callbacks to stop the training."""

    def __init__(self, module: torch.nn.Module | None = None):
        """Initialize the module, optionally with an existing PyTorch module.

        Parameters:
          module: An optional existing PyTorch module to wrap, e.g., a [torch.nn.Sequential][]
            or a pretrained Transformer. If given, the module still must be configured.
        """
        raise NotImplementedError()
