<div style="float: right; width: 40%; text-align: center"><img src="images/leaf-green.svg" style="height: 11em"></div>

# Minnt

**Minnt** is <s style="color: #777">**Mi**lan's</s> **Mi**nimalistic **N**eural **N**etwork
**T**rainer for PyTorch inspired by Keras.

<hr style="clear: both">

The central class in Minnt is the [TrainableModule](trainable_module.md) providing:

- high-level training, evaluation, and prediction API, including automatic device management;
- serialization and deserialization of weights (optionally including the optimizer) and configuration;
- automatic logging via various loggers;
- easy to use multi-GPU single-node training.

The [TransformedDataset](transformed_dataset.md) class allows applying both
per-example and per-batch transformation functions on a given dataset, and
simplifies the creation of a corresponding dataloader (in a multi-GPU setting if
required).

Furthermore, the package contains a collection of losses and metrics; however,
losses from PyTorch and metrics from [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)
can also be used directly.

Finally, several [utilities](utilities.md) are provided.

::: minnt.__version__
    options:
      heading: "The Minnt version"
      heading_level: 3
      toc_label: "Version"
      show_labels: false
      show_symbol_type_toc: false
      show_symbol_type_heading: false
