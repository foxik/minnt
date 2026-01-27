# Examples

## Using `TrainableModule`

We start with an example that illustrates just the [minnt.TrainableModule][]
class; the data preparation, data loaders, and losses are pure PyTorch.
Note that we do use [minnt.metrics.CategoricalAccuracy][] as a metric; but
```python
    metrics={"accuracy": torchmetrics.Accuracy("multiclass", num_classes=10)},
```
could also have been used.

=== "0_trainable_module.py"
    ```python linenums="1"
    --8<-- "examples/0_trainable_module.py"
    ```

When executed, the script trains a convolutional network on MNIST, using
a GPU or other accelerator if available or CPU otherwise, evaluating
loss and accuracy on the development set every epoch, and finally evaluating
loss and accuracy on the test data. The logs are shown both on a console, as
illustrated here:

```
Config epoch=0 batch_size=50 cnn_dim=16 dropout=0.2 epochs=10 hidden_layer_size=256 learning_rate=0.001 seed=42 threads=1
Epoch 1/10 7.7s loss=0.2236 accuracy=0.9304 dev:loss=0.0495 dev:accuracy=0.9840
Epoch 2/10 7.1s loss=0.0564 accuracy=0.9828 dev:loss=0.0412 dev:accuracy=0.9880
Epoch 3/10 loss=0.0440 accuracy=0.9850:  42%|███████▎         | 462/1100 [00:02<00:03, 168.96batch/s]
```

and furthermore (because `logdir` option of [configure][minnt.TrainableModule.configure] is specified),
a directory `logs/0_trainable_module-YYYYMMDD_HHMMSS-...` is created, containing the training and evaluation logs
both as plain text files and as TensorBoard logs (which can be browsed at `http://localhost:6006` after running
the `tensorboard --logdir logs` command).


## Full Minnt Example

We now present a full Minnt example, which extends the above script by using [minnt.TransformedDataset][]
to process training and evaluation data, using a configurable [learning rate decay][minnt.schedulers.GenericDecay]
(cosine by default with linear or none override), and using the [minnt.losses.CategoricalCrossEntropy][] loss.
Below you can view the whole script or just the diff to the previous example.

=== "Diff vs 0_trainable_module.py"
    ```sh exec="on" result="diff"
    scripts/example_diff 1_minnt.py 0_trainable_module.py
    ```
===+ "1_minnt.py"
    ```python linenums="1"
    --8<-- "examples/1_minnt.py"
    ```

Note that the data loaders are now created using [minnt.TransformedDataset.dataloader][],
and that the data processing now happens in [minnt.TransformedDataset.transform][].

We now illustrate several Minnt features and extensions by modifying
this example.

## Data Augmentation

To implement data augmentation using PyTorch transforms, we can construct two
augmentation pipelines for the training data: one applied on individual images
in [minnt.TransformedDataset.transform][], and another applied on batches
in [minnt.TransformedDataset.transform_batch][]. Note that for the latter to
work, the corresponding data loader must be created using the
[minnt.TransformedDataset.dataloader][] method.

Furthermore, the data augmentation can happen in subprocesses when
`--dataloader_workers` is set to a value greater than 0. By default,
Minnt avoids the `fork` start method on Unix-like systems (as is the
default in Python 3.14) and the [minnt.TransformedDataset.dataloader][]
changes the default of `persistent_workers` argument of
[torch.utils.data.DataLoader][] to `True` for better performance.

=== "Diff vs 1_minnt.py"
    ```sh exec="on" result="diff"
    scripts/example_diff 1b_data_augmentation.py 1_minnt.py
    ```
=== "1b_data_augmentation.py"
    ```python linenums="1"
    --8<-- "examples/1b_data_augmentation.py"
    ```

## Using Callbacks

[Callbacks][minnt.Callback] provide a way to perform per-epoch hooks like
customized evaluation. In the example below, we avoid passing `dev=dev` argument to
[minnt.TrainableModule.fit][], and instead pass two example callbacks:

- `evaluate_dev`, which manually logs a custom metric `dev:quality` and also
  performs [evaluation][minnt.TrainableModule.evaluate] on `dev` and adds
  the resulting dev metrics to the training logs;
- `dev_misclassifications`, which runs [prediction][minnt.TrainableModule.predict]
  on `dev` after each epoch, storing first misclassified image for every class
  to the logs (as an [image][minnt.Logger.log_image] and a [text][minnt.Logger.log_text]
  describing the predicted label). The prediction is stopped immediately after all first
  misclassifications have been found.

These callbacks demonstrate that custom metrics can be added by extending the
`logs` argument of a callback, and that the [minnt.TrainableModule.logger][] can
be used to log multimedia data like images, figures, audio, and text.

=== "Diff vs 1_minnt.py"
    ```sh exec="on" result="diff"
    scripts/example_diff 1c_callbacks.py 1_minnt.py
    ```
=== "1c_callbacks.py"
    ```python linenums="1"
    --8<-- "examples/1c_callbacks.py"
    ```

## Saving and Loading

To save and load a trained model, Minnt offers:

- [Saving][minnt.TrainableModule.save_weights] and [loading][minnt.TrainableModule.load_weights]
  of model weights. By default, only the model parameters are saved; the optimizer and scheduler
  states can be saved to a second file by passing an additional argument `optimizer_path`.

- [Saving][minnt.TrainableModule.save_options] and [loading][minnt.TrainableModule.load_options]
  of model options, which might be needed to reconstruct the model architecture
  before loading the weights. The options are saved as a JSON file and in
  addition to JSON types they also support [argparse.Namespace][] objects.
  Note that saving the options might not be needed if the model architecture is
  fixed and known in advance.

=== "Diff vs 1_minnt.py"
    ```sh exec="on" result="diff"
    scripts/example_diff 1d_saving_loading.py 1_minnt.py
    ```
=== "1d_saving_loading.py"
    ```python linenums="1"
    --8<-- "examples/1d_saving_loading.py"
    ```

### Saving via Callback

The model weights can be also saved via a [minnt.callbacks.SaveWeights][] callback after every epoch.
The path where to save the weights can include `{logdir}` and `{epoch}` placeholders,
which allows both saving the weights to a fixed path inside the log directory or
saving the weights to separate files for every epoch.

The callback can be also passed the `optimizer_path` argument to save the optimizer state;
but the options, if needed, must be saved separately using [minnt.TrainableModule.save_options][].

=== "Diff vs 1_minnt.py"
    ```sh exec="on" result="diff"
    scripts/example_diff 1e_saving_via_callback.py 1_minnt.py
    ```
=== "1e_saving_via_callback.py"
    ```python linenums="1"
    --8<-- "examples/1e_saving_via_callback.py"
    ```

### Saving Best Weights

If you want to save the weights of a model that performed best on a development set,
you can use the [minnt.callbacks.SaveBestWeights][] callback. It works similarly
to the previous callback, but only saves the weights when a specified metric
improves. Apart from specifying a metric, you might also specify whether the
metric should be maximixed (the default; `mode="max"`) or minimized (`mode="min"`).
After training, the best value of the monitored metric is available as
[minnt.callbacks.SaveBestWeights.best_value][].

=== "Diff vs 1_minnt.py"
    ```sh exec="on" result="diff"
    scripts/example_diff 1f_saving_best_weights.py 1_minnt.py
    ```
=== "1f_saving_best_weights.py"
    ```python linenums="1"
    --8<-- "examples/1f_saving_best_weights.py"
    ```

### Keeping Best Weights

In some circumstances, you might want to keep the best weights in memory instead of saving
them to disk. This can be done using the [minnt.callbacks.KeepBestWeights][]
callback, which keeps the best weights on a specified device (the model device by default).
At the end of training, the best weights stored in [minnt.callbacks.KeepBestWeights.best_state_dict][]
can be restored using the standard [torch.nn.Module.load_state_dict][] method.

Note that contrary to [minnt.callbacks.SaveBestWeights][], this callback does not allow saving
also the optimizer state.

=== "Diff vs 1_minnt.py"
    ```sh exec="on" result="diff"
    scripts/example_diff 1g_keeping_best_weights.py 1_minnt.py
    ```
=== "1g_keeping_best_weights.py"
    ```python linenums="1"
    --8<-- "examples/1g_keeping_best_weights.py"
    ```

## Using W&B Logger

Other loggers can be used instead of the default TensorBoard logger by specifying the `loggers`
argument of [minnt.TrainableModule.configure][]. In the following example, we illustrate
using the W&B logger [minnt.loggers.WandBLogger][] saving the logs to `logs/wandb` directory.

- When `loggers` are specified, the default TensorBoard logger is not used; if
  you want both TensorBoard and W&B logging, you need to explicitly specify the
  [minnt.loggers.TensorBoardLogger][] in addition to the W&B logger.

- If the `logdir` argument of [configure][minnt.TrainableModule.configure] is specified,
  the plain text logs are still saved to the log directory.

=== "Diff vs 1_minnt.py"
    ```sh exec="on" result="diff"
    scripts/example_diff 1h_wandb.py 1_minnt.py
    ```
=== "1h_wandb.py"
    ```python linenums="1"
    --8<-- "examples/1h_wandb.py"
    ```

## Profiling CPU & GPU

A [minnt.TrainableModule][] can be profiled by using the [minnt.TrainableModule.profile][] method.
The profiler tracks CPU usage, accelerator usage (if available), and memory usage, and the resulting
trace file can be inspected in TensorBoard using the `torch-tb-profiler` plugin (which can be installed
using `pip install torch-tb-profiler`). Given number of steps (forward calls) are profiled (either
during training or evaluation), after optional number of warmup steps.

The example below profiles 2 steps after a warmup of 3 steps. Note that we disable graph logging to avoid
profiling the corresponding graph tracing; we could have also used larger warmup (e.g., `warmup=5`)
to achieve a similar effect.

A memory timeline is not generated in the example because it requires the Matplotlib package;
you can try enabling it if you have it installed.

=== "Diff vs 1_minnt.py"
    ```sh exec="on" result="diff"
    scripts/example_diff 1i_profiling.py 1_minnt.py
    ```
=== "1i_profiling.py"
    ```python linenums="1"
    --8<-- "examples/1i_profiling.py"
    ```
