# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
import json
import os
import re
import struct
from typing import Any, Self, TextIO
import wave
import zlib

import torch

from .base_logger import BaseLogger
from ..type_aliases import AnyArray, TensorOrTensors


class FileSystemLogger(BaseLogger):
    """A file system logger interface."""

    def __init__(self, logdir: str) -> None:
        """Initialize the file system logger.

        Parameters:
          logdir: The root directory where the log files will be stored.
        """
        self._logdir: str = logdir
        self._log_file: TextIO | None = None

    def close(self) -> None:
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def get_file(self) -> TextIO:
        """Possibly open and return log file object.

        Returns:
          file: The opened log file.
        """
        if not self._log_file:
            os.makedirs(self._logdir, exist_ok=True)
            self._log_file = open(os.path.join(self._logdir, "logs.txt"), "a", encoding="utf-8")

        return self._log_file

    def _sanitize_path(self, input_string: str) -> str:
        """Sanitize the given string to be safe for file paths.

        Parameters:
          input_string: The input string to sanitize.
        Returns:
          The sanitized string.
        """
        return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", input_string)

    def _split_label(self, label: str) -> tuple[str, str]:
        """Split the given label into directory and base label, and sanitize them.

        Returns:
          directory: The directory part of the label.
          label: The base label.
        """
        directory, label = label.split(":", maxsplit=1) if ":" in label else ("train", label)
        directory, label = map(self._sanitize_path, (directory, label))
        return os.path.join(self._logdir, directory), label

    def _maybe_epoch(self, epoch: int) -> str:
        """Return epoch suffix if epoch is non-zero.

        Returns:
          suffix: The epoch suffix.
        """
        return f".{epoch}" if epoch else ""

    def log_audio(self, label: str, audio: AnyArray, sample_rate: int, epoch: int) -> Self:
        audio = self.preprocess_audio(audio)

        directory, label = self._split_label(label)
        os.makedirs(directory, exist_ok=True)
        with wave.open(os.path.join(directory, f"{label}{self._maybe_epoch(epoch)}.wav"), "wb") as wav_file:
            wav_file.setsampwidth(2)  # 16 bits
            wav_file.setnchannels(audio.shape[-1])
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.numpy().tobytes())

        return self

    def log_config(self, config: dict[str, Any], epoch: int) -> Self:
        config = dict(sorted(config.items()))

        print("Config", f"epoch={epoch}", *[f"{k}={v}" for k, v in config.items()],
              file=self.get_file(), flush=True)

        os.makedirs(self._logdir, exist_ok=True)
        with open(os.path.join(self._logdir, f"config{self._maybe_epoch(epoch)}.json"), "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)

        return self

    def log_epoch(
        self, logs: dict[str, float], epoch: int, epochs: int | None = None, elapsed: float | None = None,
    ) -> Self:
        print(f"Epoch {epoch}" + (f"/{epochs}" if epochs is not None else ""),
              *[f"{elapsed:.1f}s"] if elapsed is not None else [],
              *[f"{k}={v:#.{0 < abs(v) < 2e-4 and '2e' or '4f'}}" for k, v in logs.items()],
              file=self.get_file(), flush=True)
        return self

    def log_figure(self, label: str, figure: Any, epoch: int, tight_layout: bool = True, close: bool = True) -> Self:
        return super().log_figure(label, figure, epoch, tight_layout, close)

    def log_graph(self, graph: torch.nn.Module, data: TensorOrTensors, epoch: int) -> Self:
        os.makedirs(self._logdir, exist_ok=True)
        with open(os.path.join(self._logdir, f"graph{self._maybe_epoch(epoch)}.txt"), "w", encoding="utf-8") as file:
            if isinstance(graph, torch.nn.Sequential):
                print("# Sequential Module", graph, file=file, sep="\n", end="\n\n")
            traced = torch.jit.trace(graph, data, strict=False)
            print("# Traced Code", traced.code, file=file, sep="\n", end="\n\n")
            print("# Traced Graph", traced.graph, file=file, sep="\n")
            print("# Traced Inlined Graph", traced.inlined_graph, file=file, sep="\n")
        return self

    def log_image(self, label: str, image: AnyArray, epoch: int) -> Self:
        image = self.preprocess_image(image).numpy()

        directory, label = self._split_label(label)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{label}{self._maybe_epoch(epoch)}.png"), "wb") as file:
            def add_block(chunk_type: bytes, data: bytes) -> None:
                file.write(struct.pack("!I", len(data)))
                data = chunk_type + data
                file.write(data)
                file.write(struct.pack("!I", zlib.crc32(data)))

            file.write(b"\x89PNG\r\n\x1a\n")
            add_block(b"IHDR", struct.pack(
                "!2I5B", image.shape[1], image.shape[0], 8, [0, 4, 2, 6][image.shape[2] - 1], 0, 0, 0))
            pixels = [b"\2" + (image[y] - (image[y - 1] if y else 0)).tobytes() for y in range(len(image))]
            add_block(b"IDAT", zlib.compress(b"".join(pixels), level=9))
            add_block(b"IEND", b"")

        return self

    def log_text(self, label: str, text: str, epoch: int) -> Self:
        directory, label = self._split_label(label)
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, f"{label}{self._maybe_epoch(epoch)}.txt"), "w", encoding="utf-8") as file:
            file.write(text)
        return self
