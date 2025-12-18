# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import torch


def broadcast_to_prefix(tensor: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Broadcast the given tensor to the given shape from the left (prefix).

    The broadcasting is performed by unsqueezing the tensor's dimensions until
    it has the same number of dimensions as the given shape, and then expanding it.

    Parameters:
      tensor: The tensor to broadcast.
      shape: The shape to broadcast the tensor to (from the left).

    Returns:
      The broadcasted tensor.
    """
    while tensor.dim() < len(shape):
        tensor = tensor.unsqueeze(dim=-1)
    if tensor.shape != shape:
        tensor = tensor.expand(shape)
    return tensor
