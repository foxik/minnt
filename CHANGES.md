Version 1.0.2-dev
-----------------
- In PyTorch 2.10, allocator settings are set via a new generic interface, not via
  the original CUDA-specific interface (which now generates a deprecation warning).
  Therefore, start using the generic allocator settings interface when available.
- In PyTorch 2.10, avoid false-positive warning about `acc_events` when profiling
  starts.


Version 1.0.1 [30 Jan 2026]
---------------------------
- Initial stable release.
