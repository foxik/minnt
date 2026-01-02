# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import os

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


def maybe_remove_one_singleton_dimension(y: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Possibly remove one singleton dimension from the given tensor `y` to match the shape of `y_true`.

    If `y` has one more dimension than `y_true` and that extra dimension is a dimension of size 1,
    it will remove that dimension from `y`. Otherwise, `y` is returned unchanged.

    Parameters:
      y: The predicted outputs.
      y_true: The ground-truth targets.

    Returns:
      The tensor `y` with a surplus singleton dimension possibly removed.
    """
    y_shape, y_true_shape = y.shape, y_true.shape

    if len(y_shape) == len(y_true_shape) + 1:
        singleton_dim = 0
        while singleton_dim < len(y_true_shape) and y_shape[singleton_dim] == y_true_shape[singleton_dim]:
            singleton_dim += 1
        if y_shape[singleton_dim] == 1:
            y = y.squeeze(dim=singleton_dim)

    return y


def fill_and_standardize_path(path: str, **kwargs) -> str:
    """Fill placeholders in the path and standardize path separators.

    The template placeholders `{key}` in the path are replaced with the corresponding values
    from `kwargs` using `str.format`, and the both slashes and backslashes are replaced
    with the current OS path separator.

    Parameters:
      path: The path template with placeholders.
      **kwargs: The keyword arguments to fill the placeholders in the path.

    Returns:
      The standardized path with filled placeholders and OS-specific separators.
    """
    filled_path = path.format(**kwargs)
    standardized_path = filled_path.replace("\\", os.path.sep).replace("/", os.path.sep)
    return standardized_path
