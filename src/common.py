"""Helpers shared by all experiment scripts: paths, seeding and plotting."""

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to disk, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"  # pretrained checkpoints used in the report
RUNS_DIR = ROOT / "runs"  # everything produced by running the scripts


def set_seed(seed):
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def plot_rewards(rewards, window, title, path):
    """Plot the mean reward of consecutive blocks of ``window`` episodes."""
    rewards = np.asarray(rewards, dtype=np.float64)
    n_blocks = len(rewards) // window
    if n_blocks == 0:  # fewer episodes than one window: plot them raw
        window, n_blocks = 1, len(rewards)
    block_means = rewards[: n_blocks * window].reshape(n_blocks, window).mean(axis=1)
    block_centers = np.arange(n_blocks) * window + window / 2

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(block_centers, block_means)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Mean reward (per {window} episodes)")
    ax.set_title(title)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {path}")


def summarize(rewards):
    rewards = np.asarray(rewards)
    print(f"Episodes: {len(rewards)}  mean reward: {rewards.mean():.2f}  std: {rewards.std():.2f}")
