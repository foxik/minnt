# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Any, Self

import torch

from .type_aliases import AnyArray, TensorOrTensors


class Logger:
    """An abstract logger interface for logging metrics and other information."""

    def log_audio(self, label: str, audio: AnyArray, sample_rate: int, epoch: int) -> Self:
        """Log the given audio with the given label at the given epoch.

        Parameters:
          label: The label of the logged audio.
          audio: The audio to log, represented as an array with any of the
            following shapes:

            - `(L,)` of `(L, 1)` for mono audio,
            - `(L, 2)` for stereo audio.

            If the sample values are floating-point numbers, they are expected
            to be in the `[-1, 1]` range; otherwise, they are assumed to be in the
            `[-32_768, 32_767]` range.
          sample_rate: The sample rate of the audio.
          epoch: The epoch number at which the audio is logged.
        """
        raise NotImplementedError()

    def log_config(self, config: dict[str, Any], epoch: int) -> Self:
        """Log the given configuration dictionary at the given epoch.

        Parameters:
          config: A JSON-serializable dictionary representing the configuration to log.
          epoch: The epoch number at which the configuration is logged.
        """
        raise NotImplementedError()

    def log_epoch(
        self, logs: dict[str, float], epoch: int, epochs: int | None = None, elapsed: float | None = None,
    ) -> Self:
        """Log metrics collected during a given epoch.

        Parameters:
          logs: A dictionary of logged metrics for the epoch.
          epoch: The epoch number at which the logs were collected.
          epochs: The total number of epochs, if known.
          elapsed: The time elapsed during the epoch, in seconds, if known.
        """
        raise NotImplementedError()

    def log_figure(self, label: str, figure: Any, epoch: int, tight_layout: bool = True, close: bool = True) -> Self:
        """Log the given matplotlib Figure with the given label at the given epoch.

        Parameters:
          label: The label of the logged image.
          figure: A matplotlib Figure.
          epoch: The epoch number at which the image is logged.
          tight_layout: Whether to apply tight layout to the figure before logging it.
          close: Whether to close the figure after logging it.
        """
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg

        tight_layout and figure.tight_layout()
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        width, height = figure.canvas.get_width_height()
        image = torch.frombuffer(canvas.buffer_rgba(), dtype=torch.uint8).view(height, width, 4)
        close and plt.close(figure)

        return self.log_image(label, image, epoch)

    def log_graph(self, graph: torch.nn.Module, data: TensorOrTensors, epoch: int) -> Self:
        """Log the given computation graph by tracing it with the given data.

        Alternatively, loggers may choose to log the graph using TorchScript or
        other mechanisms.

        Parameters:
          graph: The computation graph to log, represented as a PyTorch module.
          data: The input data to use for tracing the computation graph.
          epoch: The epoch number at which the computation graph is logged.
        """
        raise NotImplementedError()

    def log_image(self, label: str, image: AnyArray, epoch: int) -> Self:
        """Log the given image with the given label at the given epoch.

        Parameters:
          label: The label of the logged image.
          image: The image to log, represented as an array, which can have
            any of the following shapes:

            - `(H, W)` or `(H, W, 1)` for grayscale images,
            - `(H, W, 2)` for grayscale images with alpha channel,
            - `(H, W, 3)` for RGB images,
            - `(H, W, 4)` for RGBA images.

            If the pixel values are floating-point numbers, they are expected
            to be in the `[0, 1]` range; otherwise, they are assumed to be in the
            `[0, 255]` range.
          epoch: The epoch number at which the image is logged.
        """
        raise NotImplementedError()

    def log_text(self, label: str, text: str, epoch: int) -> Self:
        """Log the given text with the given label at the given epoch.

        Parameters:
          label: The label of the logged text.
          text: The text to log.
          epoch: The epoch number at which the text is logged.
        """
        raise NotImplementedError()

    def _process_audio(self, audio: AnyArray) -> torch.Tensor:
        """Produce a CPU-based torch.Tensor with dtype int16 and shape (L, 1/2)."""
        audio = torch.as_tensor(audio, device="cpu")
        audio = audio * 32_767 if audio.dtype.is_floating_point else audio
        audio = audio.clamp(-32_768, 32_767).to(torch.int16)
        assert audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] in (1, 2)), \
            "Audio must have shape (L,) or (L, 1/2)"
        if audio.ndim == 1:
            audio = audio.unsqueeze(-1)
        return audio

    def _process_image(self, image: AnyArray) -> torch.Tensor:
        """Produce a CPU-based torch.Tensor with dtype uint8 and shape (H, W, 1/3/4)."""
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
