#!/usr/bin/env python3
import argparse

import torch
import torchvision

import minnt

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--cnn_dim", default=16, type=int, help="Number of CNN filters.")
parser.add_argument("--dropout", default=0.2, type=float, help="Dropout rate.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--hidden_layer_size", default=256, type=int, help="Size of the hidden layer.")
parser.add_argument("--label_smoothing", default=0.1, type=float, help="Label smoothing factor.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_decay", default="cosine", choices=["cosine", "linear", "none"], help="LR decay.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")


class Model(minnt.TrainableModule):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.LazyConv2d(1 * args.cnn_dim, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(3, 2),
            torch.nn.LazyConv2d(2 * args.cnn_dim, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(3, 2),
            torch.nn.LazyConv2d(4 * args.cnn_dim, 3, padding=1), torch.nn.ReLU(), torch.nn.MaxPool2d(3, 2),
            torch.nn.Flatten(),
            torch.nn.LazyLinear(args.hidden_layer_size), torch.nn.ReLU(), torch.nn.Dropout(args.dropout),
            torch.nn.LazyLinear(10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Dataset(minnt.TransformedDataset):
    def transform(self, image, label):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, label


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    minnt.startup(args.seed, args.threads)
    minnt.global_keras_initializers()

    # Load the data using torchvision.
    mnist_train_dev = torchvision.datasets.MNIST("mnist", train=True, download=True)
    mnist_train = torch.utils.data.Subset(mnist_train_dev, list(range(len(mnist_train_dev)))[:-5000])
    mnist_dev = torch.utils.data.Subset(mnist_train_dev, list(range(len(mnist_train_dev)))[-5000:])
    mnist_test = torchvision.datasets.MNIST("mnist", train=False, download=True)

    # Create data loaders from the datasets.
    train = Dataset(mnist_train).dataloader(args.batch_size, shuffle=True)
    dev = Dataset(mnist_dev).dataloader(args.batch_size)
    test = Dataset(mnist_test).dataloader(args.batch_size)

    # Create a model according to the given arguments.
    model = Model(args)
    print("The following model has been created:", model)

    # Configure the model for training.
    model.configure(
        optimizer=(optimizer := torch.optim.Adam(model.parameters(), args.learning_rate)),
        scheduler=minnt.schedulers.GenericDecay(optimizer, args.epochs * len(train), args.learning_rate_decay),
        loss=minnt.losses.CategoricalCrossEntropy(label_smoothing=args.label_smoothing),
        metrics={"accuracy": minnt.metrics.CategoricalAccuracy()},
        logdir=minnt.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args)),
    )

    # Train the model.
    model.fit(train, dev=dev, epochs=args.epochs, log_config=vars(args), log_graph=True)

    # Evaluate the model on the test data.
    model.evaluate(test)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
