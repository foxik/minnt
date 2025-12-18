# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import html
import json
from typing import Any, Self

import torch

from .base_logger import BaseLogger
from ..type_aliases import AnyArray, TensorOrTensors


class WandbLogger(BaseLogger):
    """A Wandb logger interface.

    The text values are by default also logged as HTML for better visualization.
    """
    def __init__(self, project: str, *, text_also_as_html: bool = True, **kwargs: dict[str, Any]) -> None:
        """Create the WandbLogger with the given project name.

        Additional keyword arguments are passed to `wandb.init()`.

        Parameters:
          project: The name of the Wandb project.
          text_also_as_html: Whether to log text messages also as HTML.
            That has the advantage of interactive visualization of the value
            at different epochs and preserving whitespace formatting.
          kwargs: Additional keyword arguments passed to `wandb.init()`.
        """
        import wandb
        self.wandb = wandb
        self.run = self.wandb.init(project=project, **kwargs)
        self._text_also_as_html = text_also_as_html

    def __del__(self) -> None:
        # Close the run.
        self.run.finish()

    def _maybe_as_html(self, label: str, text: str) -> dict[str, Any]:
        """Return a dict with the HTML version of the text if enabled.

        The text is converted to HTML-safe format and returned as a wandb.Html object.
        """
        if not self._text_also_as_html:
            return {}
        return {f"{label}_html": self.wandb.Html("<pre>" + html.escape(text) + "</pre>")}

    def log_audio(self, label: str, audio: AnyArray, sample_rate: int, epoch: int) -> Self:
        audio = self._process_audio(audio).numpy()
        self.run.log({label: self.wandb.Audio(audio, sample_rate=sample_rate)}, step=epoch)
        return self

    def log_config(self, config: dict[str, Any], epoch: int) -> Self:
        config = dict(sorted(config.items()))
        self.run.config.update(config)
        config = json.dumps(config, ensure_ascii=False, indent=2)
        self.run.log({"config": config} | self._maybe_as_html("config", config), step=epoch)
        return self

    def log_epoch(
        self, logs: dict[str, float], epoch: int, epochs: int | None = None, elapsed: float | None = None,
    ) -> Self:
        self.run.log(logs, step=epoch)
        return self

    def log_figure(self, label: str, figure: Any, epoch: int, tight_layout: bool = True, close: bool = True) -> Self:
        return super().log_figure(label, figure, epoch, tight_layout, close)

    def log_graph(self, graph: torch.nn.Module, data: TensorOrTensors, epoch: int) -> Self:
        self.run.watch(graph, log=None, log_graph=True)
        graph(data)  # Run the graph to log it.
        return self

    def log_image(self, label: str, image: AnyArray, epoch: int) -> Self:
        image = self._process_image(image).numpy()
        self.run.log({label: self.wandb.Image(image)}, step=epoch)
        return self

    def log_text(self, label: str, text: str, epoch: int) -> Self:
        self.run.log({label: text} | self._maybe_as_html(label, text), step=epoch)
        return self
