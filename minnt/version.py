# This file is part of Minnt <http://github.com/ufal/minnt/>.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

__version__ = "1.0.1"
"""The current version of the Minnt package, formatted according to [Semantic Versioning](https://semver.org/).

The version string is in the format _major.minor.patch[-prerelease]_, where the _prerelease_ part is
optional and is empty for stable releases.
"""


def require_version(required_version: str) -> None:
    """Verify the installed version is at least `required_version`, and set API compatibility for that version.

    This method has two purposes: to ensure that the installed version of Minnt meets the minimum
    required version, and to set the **API compatibility level** for the Minnt package.

    The goal of the API compatibility is to ensure that the API in newer versions of Minnt has the same
    **intended behavior** as in the `required_version`.

    Example:
      If a package required Minnt version `1.3` and version `1.4` introduced for example a new default
      override in [minnt.global_keras_initializers][], with `minnt.require_version("1.3")` the package
      would still get the old behavior from version `1.3`, even when running with Minnt `1.4` or newer.

    Warning:
      The API compatibility does not guarantee completely identical behavior between versions, for example
      bugs may be fixed in newer versions changing the original behavior. That is why we talk about
      **intended behavior**.

    Parameters:
      required_version: The minimum required version, in the format _major.minor.patch_,
        and the required API compatibility.
    """
    required = required_version.split(".")
    assert len(required) <= 3, "Expected at most 3 version components"
    assert all(part.isdecimal() for part in required), "Expected only numeric version components"

    required = list(map(int, required))
    current = list(map(int, __version__.split("-", maxsplit=1)[0].split(".")))

    assert current[:len(required)] >= required, \
        f"The minnt>={required_version} is required, but found only {__version__}."
