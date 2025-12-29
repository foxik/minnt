# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch


class KeepPrevious:
    pass
keep_previous = KeepPrevious()  # noqa: E305


class TrainableModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module | None = None):
        """Initialize the module, optionally with an existing PyTorch module.

        Parameters:
          module: An optional existing PyTorch module to wrap, e.g., a [torch.nn.Sequential][]
            or a pretrained Transformer. If given, the module still must be configured.
        """
        raise NotImplementedError()
