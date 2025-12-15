# experiments/plots/style.py
from pathlib import Path
import matplotlib.pyplot as plt

# Style matplotlib global (proche de ton script "article")
plt.style.use("seaborn-v0_8-white")

# Taille par défaut des figures (alignée sur l'article)
DEFAULT_FIGSIZE = (10, 6)

# Couleurs alignées sur ton script "article"
METHOD_STYLES = {
    "jeffreys": {
        "label": "Jeffreys equal-tailed",
        "color": "#d62728",  # red
    },
    "cp": {
        "label": "Exact Clopper–Pearson",
        "color": "#ff7f0e",  # orange
    },
    "normal": {
        "label": "Normal approximation",
        "color": "#2ca02c",  # green
    },
    "jeffreys_ecb": {
        "label": "Jeffreys ECB (upper)",
        "color": "#1f77b4",  # blue
    },
    "cp_upper": {
        "label": "Exact CP upper",
        "color": "#8c564b",  # brown
    },
    "normal_upper": {
        "label": "Normal upper",
        "color": "#9467bd",  # purple
    },
}


def new_figure(figsize=DEFAULT_FIGSIZE):
    """
    Crée une figure + axes avec un style homogène.
    Proche du style "article": seaborn white, grid léger, spines top/right off.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(alpha=0.3)

    # Spines comme dans ton script "article"
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return fig, ax


def finalize_ax(
    ax,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    add_legend: bool = True,
    nominal_level: float | None = None,
    nominal_label: str = "Nominal level",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """
    Applique le style commun : labels, titre, ligne de niveau nominal, légende,
    et éventuellement limites d'axes.
    """
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    if title is not None:
        ax.set_title(title)

    if nominal_level is not None:
        ax.axhline(
            y=nominal_level,
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=nominal_label,
        )

    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)

    if add_legend:
        ax.legend()

    return ax


def save_figure(fig, out_path: Path):
    """
    Sauvegarde au format PNG haute résolution (dpi=300) avec bbox propre.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)
