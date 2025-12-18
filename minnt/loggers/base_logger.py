# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Any, Self

import torch

from ..logger import Logger
from ..type_aliases import AnyArray


class BaseLogger(Logger):
    """An abstract logger providing base functionality for other loggers."""

    def __del__(self) -> None:
        """Ensure proper resource cleanup on deletion."""
        self.close()

    def log_figure(self, label: str, figure: Any, epoch: int, tight_layout: bool = True, close: bool = True) -> Self:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg

        tight_layout and figure.tight_layout()
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        width, height = figure.canvas.get_width_height()
        image = torch.frombuffer(canvas.buffer_rgba(), dtype=torch.uint8).view(height, width, 4)
        close and plt.close(figure)

        return self.log_image(label, image, epoch)

    def preprocess_audio(self, audio: AnyArray) -> torch.Tensor:
        """Produce a CPU-based [torch.Tensor][] with `dtype=torch.int16` and shape `(L, {1/2})`."""
        audio = torch.as_tensor(audio, device="cpu")
        audio = audio * 32_767 if audio.dtype.is_floating_point else audio
        audio = audio.clamp(-32_768, 32_767).to(torch.int16)
        assert audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] in (1, 2)), \
            "Audio must have shape (L,) or (L, 1/2)"
        if audio.ndim == 1:
            audio = audio.unsqueeze(-1)
        return audio

    def preprocess_image(self, image: AnyArray) -> torch.Tensor:
        """Produce a CPU-based [torch.Tensor][] with `dtype=torch.uint8` and shape `(H, W, {1/3/4})`."""
        image = torch.as_tensor(image, device="cpu")
        image = (image * 255 if image.dtype.is_floating_point else image).clamp(0, 255).to(torch.uint8)
        assert image.ndim == 2 or (image.ndim == 3 and image.shape[2] in (1, 2, 3, 4)), \
            "Image must have shape (H, W) or (H, W, 1/2/3/4)"
        if image.ndim == 2:
            image = image.unsqueeze(-1)
        if image.shape[2] == 2:
            # Convert to RGBA
            image = torch.stack([image[:, :, 0]] * 3 + [image[:, :, 1]], dim=-1)
        return image
