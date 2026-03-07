"""Plotting helpers for stage-3 optimization diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_3d_compare(gtM, predM, title, out_png: Path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    gtM = np.asarray(gtM)
    predM = np.asarray(predM)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(gtM[:, 0], gtM[:, 1], gtM[:, 2], s=30, marker="o", label="GT refined")
    ax.scatter(predM[:, 0], predM[:, 1], predM[:, 2], s=30, marker="^", label="Pred aligned")

    for i in range(gtM.shape[0]):
        ax.plot(
            [gtM[i, 0], predM[i, 0]],
            [gtM[i, 1], predM[i, 1]],
            [gtM[i, 2], predM[i, 2]],
            linewidth=0.5,
        )

    ax.set_title(title)
    ax.legend()
    set_axes_equal(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_loss_curve(loss_hist, out_png: Path):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7, 4))
    plt.plot(loss_hist)
    plt.title("Optimization loss")
    plt.xlabel("iter")
    plt.ylabel("loss")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
