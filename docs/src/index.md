---
title: Minnt
---

<div style="float: right; width: 40%; text-align: center"><img src="images/leaf-green-noborder.svg" style="height: 11em"></div>

# Minnt 1.0.5-dev

**Minnt** is <s style="color: #777">**Mi**lan's</s> **Mi**nimalistic **N**eural **N**etwork
**T**rainer for PyTorch inspired by Keras.

See the [Examples](examples.md). Install using:

<pre style="clear: both"><code>pip install minnt --extra-index-url=https://download.pytorch.org/whl/cu128 torch torchvision</code></pre>

---

The central class in Minnt is the [TrainableModule](trainable_module.md) providing:

- high-level [training][minnt.TrainableModule.fit], [evaluation][minnt.TrainableModule.evaluate],
  and [prediction][minnt.TrainableModule.predict] API, including automatic device management;
- [serialization][minnt.TrainableModule.save_weights] and [deserialization][minnt.TrainableModule.load_weights]
  of weights (optionally including the optimizer) and module [options][minnt.TrainableModule.save_options].
- easy to use multi-GPU single-node training (planned, not yet implemented),
- automatic logging via various [loggers](logger.md),
- easy [profiling support][minnt.TrainableModule.profile].

The [TransformedDataset](transformed_dataset.md) class allows applying both
per-example and per-batch transformation functions on a given dataset, and
simplifies the creation of a corresponding dataloader (in a multi-GPU setting if
required).

Furthermore, the package contains a collection of [losses](loss.md) and [metrics](metric.md); however,
losses from PyTorch and metrics from [torchmetrics](https://lightning.ai/docs/torchmetrics/stable/)
can also be used directly.

Finally, several [utilities](utilities.md) are provided, including
a [Vocabulary](vocabulary.md) class for converting between strings and their
indices.
