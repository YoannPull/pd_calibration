# experiments/temporal_drift/plot_temporal_drift.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_reject_rate_vs_time(df, save_dir=None, prefix="temporal_drift"):
    """
    Taux de rejet de H0 : p = p_hat en fonction du temps, une courbe par méthode.
    On sépare visuellement avant / après le début du drift.
    """
    T0 = df["T0"].iloc[0]
    T = df["T"].iloc[0]

    plt.figure(figsize=(18, 9))

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("t")
        plt.plot(
            sub["t"],
            sub["reject_rate"],
            marker="o",
            label=method,
        )

    # Ligne verticale : début du drift
    plt.axvline(x=T0 + 0.5, color="red", linestyle="--", label="Drift onset")

    # (optionnel) bande verticale pour la phase post-drift
    plt.axvspan(T0 + 0.5, T + 0.5, color="red", alpha=0.05)

    alpha_nominal = df["alpha_nominal"].iloc[0]
    plt.axhline(
        y=alpha_nominal,
        color="black",
        linestyle=":",
        label=f"Nominal level (α = {alpha_nominal:.2f})",
    )

    plt.xlabel("Time period t")
    plt.ylabel(r"Rejection rate of $H_0: p = \hat p$")
    plt.title("Temporal evolution of rejection rates under drift")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_dir is not None:
        out_path = save_dir / f"{prefix}_reject_rate_vs_time.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()

    plt.close()


def plot_coverage_vs_time(df, save_dir=None, prefix="temporal_drift"):
    """
    Couverture empirique de p_true(t) en fonction du temps, une courbe par méthode.
    """
    T0 = df["T0"].iloc[0]
    T = df["T"].iloc[0]
    conf = df["conf_level"].iloc[0]

    plt.figure(figsize=(18, 9))

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("t")
        plt.plot(
            sub["t"],
            sub["coverage"],
            marker="o",
            label=method,
        )

    plt.axvline(x=T0 + 0.5, color="red", linestyle="--", label="Drift onset")
    plt.axvspan(T0 + 0.5, T + 0.5, color="red", alpha=0.05)

    plt.axhline(
        y=conf,
        color="black",
        linestyle=":",
        label=f"Nominal coverage ({conf:.2f})",
    )

    plt.xlabel("Time period t")
    plt.ylabel(r"Coverage of $p(t)$")
    plt.title("Temporal evolution of coverage under drift")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_dir is not None:
        out_path = save_dir / f"{prefix}_coverage_vs_time.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()

    plt.close()


def plot_length_vs_time(df, save_dir=None, prefix="temporal_drift"):
    """
    Longueur moyenne des intervalles en fonction du temps, une courbe par méthode.
    """
    T0 = df["T0"].iloc[0]
    T = df["T"].iloc[0]

    plt.figure(figsize=(18, 9))

    for method in sorted(df["method"].unique()):
        sub = df[df["method"] == method].sort_values("t")
        plt.plot(
            sub["t"],
            sub["avg_length"],
            marker="o",
            label=method,
        )

    plt.axvline(x=T0 + 0.5, color="red", linestyle="--", label="Drift onset")
    plt.axvspan(T0 + 0.5, T + 0.5, color="red", alpha=0.05)

    plt.xlabel("Time period t")
    plt.ylabel("Average interval length")
    plt.title("Temporal evolution of interval lengths under drift")
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_dir is not None:
        out_path = save_dir / f"{prefix}_length_vs_time.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df_path = data_dir / "temporal_drift_results.csv"
    df = pd.read_csv(df_path)

    prefix = "temporal_drift"

    plot_reject_rate_vs_time(df, save_dir=figs_dir, prefix=prefix)
    plot_coverage_vs_time(df, save_dir=figs_dir, prefix=prefix)
    plot_length_vs_time(df, save_dir=figs_dir, prefix=prefix)
