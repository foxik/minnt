# This file is part of Minnt <http://github.com/ufal/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from typing import Literal, TYPE_CHECKING

from ..callback import Callback, StopTraining, STOP_TRAINING
from ..type_aliases import Logs
if TYPE_CHECKING:
    from ..trainable_module import TrainableModule


class EarlyStopping(Callback):
    """A callback that stops the training after a metric stops improving."""

    def __init__(
        self,
        metric: str,
        patience: int,
        mode: Literal["max", "min"] = "max",
    ) -> None:
        """Create the EarlyStopping callback.

        Parameters:
          metric: The metric name from `logs` dictionary to monitor.
          patience: The callback stops the training if the monitored metric does not improve for
            `patience` consecutive epochs.
          mode: One of `"max"` or `"min"`, indicating whether the monitored metric should be maximized
            or minimized.
        """
        assert mode in ("max", "min"), "mode must be one of 'max' or 'min'"

        self._metric = metric
        self._mode = mode
        self._patience = patience
        self._epochs_without_improvement = 0

        self.best_value = None

    best_value: float | None
    """The best metric value seen so far."""

    def __call__(self, module: "TrainableModule", epoch: int, logs: Logs) -> StopTraining | None:
        if (self.best_value is None
                or (self._mode == "max" and logs[self._metric] > self.best_value)
                or (self._mode == "min" and logs[self._metric] < self.best_value)):
            self.best_value = logs[self._metric]
            self._epochs_without_improvement = 0
        else:
            self._epochs_without_improvement += 1

        if self._epochs_without_improvement >= self._patience:
            return STOP_TRAINING
