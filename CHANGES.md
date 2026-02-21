Version 1.0.4-dev
-----------------


Version 1.0.3 [21 Feb 2026]
---------------------------
- Require `setuptools < 80.9` as a temporal workaround for removed
  `pkg_resources` in `setuptools == 82.0`.
- Improve typing (`Metric` is now just a protocol, together with `Loss` they are
  more generic, `fit` and `evaluate` return explicitly `dict[str, float]`).


Version 1.0.2 [02 Feb 2026]
---------------------------
- Support lightweight profiling mode with reduced overhead collecting only basic
  information.
- In PyTorch 2.10, allocator settings are set via a new generic interface, not via
  the original CUDA-specific interface (which now generates a deprecation warning).
  Therefore, start using the generic allocator settings interface when available.
- In PyTorch 2.10, avoid false-positive warning about `acc_events` when profiling
  starts.


Version 1.0.1 [30 Jan 2026]
---------------------------
- Initial stable release.
