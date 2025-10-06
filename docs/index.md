---
title: Minnt
---

<div style="float: right; width: 40%; text-align: center">
  <svg viewBox="2 3 20 19" height="11em"><path d="M17,8C8,10 5.9,16.17 3.82,21.34L5.71,22L6.66,19.7C7.14,19.87 7.64,20 8,20C19,20 22,3 22,3C21,5 14,5.25 9,6.25C4,7.25 2,11.5 2,13.5C2,15.5 3.75,17.25 3.75,17.25C7,8 17,8 17,8Z" style="fill:#66c05b" /></svg>
</div>

# Minnt

**Minnt** is <s style="color: #777">**Mi**lan's</s> **Mi**nimalistic **N**eural **N**etwork
**T**rainer for PyTorch inspired by Keras.

<hr style="clear: both">

```sh
pip install minnt --extra-index-url=https://download.pytorch.org/whl/cu128 torch torchvision
```

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
      heading_level: 2
      toc_label: "Version"
      show_labels: false
      show_symbol_type_toc: false
      show_symbol_type_heading: false
