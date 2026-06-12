Version 1.0.6 [12 Jun 2026]
---------------------------
- In `TrainableModule`, move tensors with `non_blocking=True` in all code paths.
- Explicitly pass `weights_only=True` to `torch.load` for consistency and safety
  in older PyTorch installations.


Version 1.0.5 [05 Jun 2026]
---------------------------
- If `predict_step` is a generator function, it is assumed to generate
  individual predicted items. This is a convenient alternative to generating
  full batches in `predict_step` and overriding `unpack_batch`.
- Add `sort_keys` option to `Logger.log_config` method.
- Add `EarlyStopping` callback that stops training if a monitored metric
  fails to improve for a given number of epochs.
- Add `patience` option to `KeepBestWeights` and `SaveBestWeights` callbacks.
- Add `baseline` option to the `SaveBestWeights` callback.
- Dynamically resize progress bar on window resize.
- Export `StopTraining` type to replace the invalid `Literal[STOP_TRAINING]`.
- Fix a crash in `TrainableModule.get_tb_writer` by using a correct method name.
- Fix producing logs more frequently than `MINNT_REPORT_EACH` when more then
  10 seconds have elapsed between updates.
- Fix incorrect recursive calls in `unpack_batch`.
- When overriding dataset limit, do not create a `Subset` dataset.
- When wrapping a module by passing it to `TrainableModule` constructor,
  correctly pass all arguments (both positional and keyword) in `forward`.
- Do not consider norm layer subclasses in `initializers_override`.
- Override also `InstanceNorm` epsilon in `global_keras_initializers`.
- The `WandBLogger.log_graph` method no longer tries to log the graph to WandB.


Version 1.0.4 [24 Feb 2026]
---------------------------
- Improve `format_logdir` to correctly handle environments without `__file__`
  and calls with no kwargs.


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
