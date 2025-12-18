# This file is part of Minnt <http://github.com/foxik/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
from collections.abc import Iterable


class Vocabulary:
    """ A class for managing mapping between strings and indices.

    The vocabulary is initialized with a list of strings.

    It always contains a special padding token [Vocabulary.PAD][minnt.Vocabulary.PAD]
    at index 0, and optionally an unknown token [Vocabulary.UNK][minnt.Vocabulary.UNK]
    at index 1 (when [Vocabulary.has_unk][minnt.Vocabulary.has_unk]).
    """
    PAD: int = 0
    """The index of the padding token."""
    UNK: int = 1
    """The index of the unknown token, which might not be present."""

    def __init__(self, strings: Iterable[str], add_unk: bool = False) -> None:
        """Initialize the vocabulary with the given list of strings.

        The [Vocabulary.PAD][minnt.Vocabulary.PAD] is always the first token in the vocabulary;
        [Vocabulary.UNK][minnt.Vocabulary.UNK] is the second token but only when `add_unk=True`.
        """
        self._strings = ["[PAD]"] + (["[UNK]"] if add_unk else [])
        self._strings.extend(strings)
        self._string_map = {string: index for index, string in enumerate(self._strings)}
        self._has_unk = add_unk

    @property
    def has_unk(self) -> bool:
        """A boolean property indicating whether the vocabulary was constructed with an UNK token."""
        return self._has_unk

    def __len__(self) -> int:
        """The number of strings in the vocabulary.

        Returns:
          The size of the vocabulary.
        """
        return len(self._strings)

    def __iter__(self) -> Iterable[str]:
        """Return an iterator over strings in the vocabulary.

        Returns:
          An iterator over strings in the vocabulary.
        """
        return iter(self._strings)

    def string(self, index: int) -> str:
        """Convert vocabulary index to string.

        Parameters:
          index: The vocabulary index.

        Returns:
          The string corresponding to the given index.
        """
        return self._strings[index]

    def strings(self, indices: Iterable[int]) -> list[str]:
        """Convert a sequence of vocabulary indices to strings.

        Parameters:
          indices: An iterable of vocabulary indices.

        Returns:
          A list of strings corresponding to the given indices.
        """
        return [self._strings[index] for index in indices]

    def index(self, string: str) -> int | None:
        """Convert string to vocabulary index.

        Parameters:
          string: The string to convert.

        Returns:
          The index corresponding to the given string. If the string is not found in the vocabulary, then

            - if the vocabulary was constructed with an UNK token, it returns [Vocabulary.UNK][minnt.Vocabulary.UNK];
            - otherwise, it returns `None`.
        """
        return self._string_map.get(string, self.UNK if self._has_unk else None)

    def indices(self, strings: Iterable[str]) -> list[int | None]:
        """Convert a sequence of strings to vocabulary indices.

        Parameters:
          strings: An iterable of strings to convert.

        Returns:
          A list of indices corresponding to the given strings. For each string not found in the vocabulary, it returns

            - [Vocabulary.UNK][minnt.Vocabulary.UNK] if the vocabulary was constructed with an UNK token;
            - otherwise, it returns `None`.
        """
        return [self._string_map.get(string, self.UNK if self._has_unk else None) for string in strings]
