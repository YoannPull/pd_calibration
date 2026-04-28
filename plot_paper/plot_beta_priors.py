import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import beta as beta_dist
from matplotlib.ticker import ScalarFormatter


def plot_beta_priors_grid(outpath: Path, eps: float = 1e-3):
    """
    Original grid of Beta priors (same spirit as your initial script).
    """
    pairs = [
        (0.5, 0.5),
        (1.0, 1.0),
        (5.0, 5.0),
        (2.0, 5.0),
        (2.5, 2.0),
        (eps, eps),  # approx Beta(0,0)
    ]

    p = np.linspace(1e-6, 1 - 1e-6, 4000)

    fig, axes = plt.subplots(3, 2, figsize=(8, 9), sharex=True)
    axes = axes.ravel()

    for ax, (a, b) in zip(axes, pairs):
        y = beta_dist.pdf(p, a, b)
        linewidth = 3 if np.isclose(a, b) else 2
        ax.plot(p, y, linewidth=linewidth)

        if a == eps and b == eps:
            ax.set_title(rf"Haldane approx. $\alpha=\beta=\varepsilon={eps}$")
        else:
            ax.set_title(rf"$\alpha={a},\beta={b}$")

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 6)  # keep your original choice
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("p")
        ax.set_ylabel("Density")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_noninformative_beta_priors_grid(outpath: Path, eps: float = 1e-3):
    """
    Grid plot (one panel per prior) for classic "non-informative/objective" Beta priors:
    - Uniform (Laplace): Beta(1,1)
    - Jeffreys: Beta(1/2, 1/2)
    - Haldane (improper): Beta(0,0) approximated by Beta(eps, eps)

    Uses log y-scale to make boundary behavior visible and readable.
    """
    priors = [
        ("Uniform (Laplace)", 1.0, 1.0),
        ("Jeffreys (reference)", 0.5, 0.5),
        (rf"Haldane (improper, approx. $\varepsilon$={eps:g})", eps, eps),
    ]

    p = np.linspace(1e-6, 1 - 1e-6, 4000)

    # 3 priors -> 2x2 grid (one empty panel)
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    axes = axes.ravel()

    for ax, (name, a, b) in zip(axes, priors):
        y = beta_dist.pdf(p, a, b)
        y = np.maximum(y, 1e-300)  # safety for log scale
        linewidth = 3 if np.isclose(a, b) else 2

        ax.semilogy(p, y, linewidth=linewidth)
        ax.set_title(rf"{name} — $\alpha={a:g},\ \beta={b:g}$")

        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("p")
        ax.set_ylabel("Density (log)")

        # Disable weird scientific offset text on y-axis
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    # Hide unused axis (4th panel)
    for ax in axes[len(priors):]:
        ax.axis("off")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    eps = 2e-3

    out_grid = Path("plot_paper/reports/figures/beta_priors_grid.png")
    plot_beta_priors_grid(outpath=out_grid, eps=eps)
    print(f"Saved: {out_grid}")

    out_noninf = Path("plot_paper/reports/figures/beta_priors_noninformative_grid.png")
    plot_noninformative_beta_priors_grid(outpath=out_noninf, eps=eps)
    print(f"Saved: {out_noninf}")
