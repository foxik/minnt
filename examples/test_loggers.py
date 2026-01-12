#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

import minnt

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--fs_logger", default=False, action="store_true", help="Use FileSystemLogger.")
parser.add_argument("--hidden_layer_size", default=100, type=int, help="Size of the hidden layer.")
parser.add_argument("--logger", default="FileSystemLogger", help="Logger to use.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--tb_logger", default=False, action="store_true", help="Use TensorBoardLogger.")
parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
parser.add_argument("--wandb_logger", default=None, type=str, help="Use WandbLogger with given project name.")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    minnt.startup(args.seed, args.threads)

    # Create logdir name.
    logdir = minnt.format_logdir("logs/{file-}{timestamp}{-config}", **vars(args))

    # Create a simple model for MNIST.
    model = minnt.TrainableModule(torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.LazyLinear(args.hidden_layer_size),
        torch.nn.ReLU(),
        torch.nn.LazyLinear(10),
    ))
    example_input = torch.zeros(1, 28, 28)
    model(example_input)  # initialize lazy layers

    # Create testing audio, deliberately in PyTorch and with different rates.
    audio_mono = torch.sin(torch.linspace(0, 2 * 440 * 2 * torch.pi, 2 * 16_000))
    audio_stereo = torch.stack([
        torch.sin(torch.linspace(0, 2 * 440 * 2 * torch.pi, 2 * 44_100)),
        (torch.linspace(0, 440 * 2, 2 * 44_100) % 1) * 2 - 1,
    ], dim=1)

    # Create testing images.
    checker = torch.kron(torch.eye(2), torch.ones(8, 8)).repeat(4, 4)
    g = torch.linspace(0, 1., 256).expand(256, -1)
    gradient_ga = torch.stack((g, torch.rot90(g, 1)), dim=-1)
    gradient_rgb = torch.stack((g, torch.rot90(g, 1), torch.rot90(g, 2)), dim=-1)
    gradient_rgba = torch.stack((g, torch.rot90(g, 1), torch.rot90(g, 2), torch.rot90(g, 3)), dim=-1)

    # Method for creating testing figures.
    def create_figures():
        fig_2d = plt.figure()
        xs = np.linspace(0, 4 * np.pi, 100)
        plt.plot(xs, np.sin(xs))
        plt.title("Sine Wave")

        fig_3d = plt.figure()
        ax = fig_3d.add_subplot(111, projection="3d")
        x, y = np.linspace(-5, 5, 100), np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y, indexing="xy")
        Z = np.sin(np.sqrt(X**2 + Y**2))
        ax.plot_surface(X, Y, Z, cmap="viridis")
        plt.title("3D Surface Plot of sqrt(x^2 + y^2)")

        return fig_2d, fig_3d

    for logger_name in ["fs", "tb", "wandb"]:
        if not getattr(args, f"{logger_name}_logger"):
            continue

        if logger_name == "fs":
            logger = minnt.loggers.FileSystemLogger(logdir)
        elif logger_name == "tb":
            logger = minnt.loggers.TensorBoardLogger(logdir)
        elif logger_name == "wandb":
            logger = minnt.loggers.WandbLogger(project=args.wandb_logger, dir=logdir)

        # Configuration
        logger.log_config(vars(args), 0)

        # Graph
        logger.log_graph(model, example_input, 0)

        # Audio
        logger.log_audio("sine 440Hz", audio_mono, 16_000, 1)
        logger.log_audio("sine 440Hz + triangle", audio_stereo, 44_100, 1)

        # Images
        logger.log_image("checkerboard/G-float", checker, 1)
        logger.log_image("checkerboard/G-uint8-low", (checker * 128).to(torch.uint8), 1)
        logger.log_image("checkerboard/G-uint8-high", (checker * 127 + 128).to(torch.uint8), 1)
        logger.log_image("checkerboard/G-channel", torch.stack([checker], dim=-1), 1)
        logger.log_image("checkerboard/GA", torch.stack([0 * checker, checker], dim=-1), 1)
        logger.log_image("checkerboard/Rgb", torch.stack([checker, 0 * checker, 0 * checker], dim=-1), 1)
        logger.log_image("checkerboard/rGb", torch.stack([0 * checker, checker, 0 * checker], dim=-1), 1)
        logger.log_image("checkerboard/rgB", torch.stack([0 * checker, 0 * checker, checker], dim=-1), 1)
        logger.log_image("checkerboard/RgbA", torch.stack([checker, 0 * checker, 0 * checker, checker], dim=-1), 1)

        logger.log_image("gradient/G", gradient_ga.mean(dim=-1), 1)
        logger.log_image("gradient/GA", gradient_ga, 1)
        logger.log_image("gradient/RGB", gradient_rgb, 1)
        logger.log_image("gradient/RGB-CHW", gradient_rgb.movedim(-1, 0), 1, data_format="CHW")
        logger.log_image("gradient/RGBA", gradient_rgba, 1)
        logger.log_image("gradient/RGBA-uint8", (gradient_rgba * 255).to(torch.uint8), 1)

        # Scalars and text
        for epoch in range(1, 10 + 1):
            for dataset in ["train", "dev"] + (["test"] if epoch == 10 else []):
                offset = 0.05 if dataset == "dev" else 0.1 if dataset == "test" else 0.0
                logger.log_epoch({
                    f"{dataset}:accuracy": 0.45 + min(epoch, 4 + epoch % 2) / 10 - offset,
                    f"{dataset}:loss": 2 / epoch + offset,
                }, epoch, epochs=10, elapsed=4.2)

            logger.log_text("response", f"A sample log message in epoch {epoch}.", epoch)

        # Figures
        fig_2d, fig_3d = create_figures()

        logger.log_figure("2d figure", fig_2d, 10, tight_layout=False, close=False)
        logger.log_figure("3d figure", fig_3d, 10, tight_layout=False, close=False)

        logger.log_figure("2d figure", fig_2d, 11)
        logger.log_figure("3d figure", fig_3d, 11)


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)
