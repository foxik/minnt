# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""Types and type aliases used by Minnt."""
from typing import Any, TypeAlias

import numpy as np
import torch

AnyArray: TypeAlias = torch.Tensor | np.ndarray | list | tuple
"""A type alias for any array-like structure.

PyTorch tensors, NumPy arrays, lists, and tuples are supported.
"""

Tensor: TypeAlias = torch.Tensor | torch.nn.utils.rnn.PackedSequence
"""A type alias for a single tensor or a packed sequence of tensors."""

TensorOrTensors: TypeAlias = Tensor | tuple[Tensor, ...] | list[Tensor] | Any
"""A type alias for a single tensor/packed sequence of a sequence of them.

While a tensor or a tuple of them is the most common, any type is allowed
here to accomodate dictionaries or custom data structures.
"""
