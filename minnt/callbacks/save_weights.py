# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import TYPE_CHECKING

from ..callback import Callback
from ..type_aliases import Logs
if TYPE_CHECKING:
    from ..trainable_module import TrainableModule


class SaveWeights(Callback):
    """A callback that saves model weights to a file after each epoch."""

    def __init__(
        self, path: str, optimizer_path: str | None = None, config_path: str | None = None, config: dict = {},
    ) -> None:
        """Create the SaveWeights callback.

        Parameters:
          path: A path where weights will be saved using the [minnt.TrainableModule.save_weights][]
            method after each epoch. Note that you can use templates like `{logdir}` and `{epoch[:formatting]}`.
          optimizer_path: An optional path passed to [minnt.TrainableModule.save_weights][] to
            save also the optimizer state.
          config_path: An optional path where to save the configuration. If given, the
            [minnt.TrainableModule.save_config][] method is used to save the configuration
            from the `config` parameter after each epoch.
          config: A directory with configuration parameters passed to the
            [minnt.TrainableModule.save_config][].
        """
        self._path = path
        self._optimizer_path = optimizer_path
        self._config_path = config_path
        self._config = config

    def __call__(self, module: "TrainableModule", epoch: int, logs: Logs) -> None:
        module.save_weights(self._path, optimizer_path=self._optimizer_path)
        if self._config_path is not None:
            module.save_config(self._config_path, self._config)
