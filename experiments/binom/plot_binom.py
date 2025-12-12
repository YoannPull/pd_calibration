# experiments/binom_coverage/plot_binom_coverage.py

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_method(
    p_values,
    coverage,
    confidence_level,
    n: int,
    method_label: str,
    filename: Path,
    color: str,
):
    """
    Trace coverage vs p pour une méthode donnée, avec ligne horizontale au niveau nominal.
    """
    plt.figure(figsize=(24, 12))
    plt.plot(p_values, coverage, label=method_label, color=color)
    plt.axhline(
        y=confidence_level,
        color="black",
        linestyle="--",
        label="Nominal level",
    )
    plt.xlabel("p")
    plt.ylabel("Coverage probability")
    plt.title(f"Coverage probability vs p (n={n}) - {method_label}")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close()


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # Doit correspondre au n utilisé dans la simu
    n = 100
    csv_path = data_dir / f"binom_coverage_n{n}.csv"

    df = pd.read_csv(csv_path)

    p_values = df["p"].values
    confidence_level = df["conf_level"].iloc[0]

    coverage_jeff = df["coverage_jeff"].values
    coverage_exact = df["coverage_exact"].values
    coverage_approx = df["coverage_approx"].values

    # Graphique Jeffreys (bilatéral)
    plot_method(
        p_values=p_values,
        coverage=coverage_jeff,
        confidence_level=confidence_level,
        n=n,
        method_label="Jeffreys equal-tailed interval",
        filename=figs_dir / f"binom_coverage_jeffreys_n{n}.png",
        color="green",
    )

    # Graphique Clopper–Pearson exact (bilatéral)
    plot_method(
        p_values=p_values,
        coverage=coverage_exact,
        confidence_level=confidence_level,
        n=n,
        method_label="Exact Clopper–Pearson",
        filename=figs_dir / f"binom_coverage_exact_cp_n{n}.png",
        color="red",
    )

    # Graphique approximation normale (bilatéral)
    plot_method(
        p_values=p_values,
        coverage=coverage_approx,
        confidence_level=confidence_level,
        n=n,
        method_label="Normal approximation",
        filename=figs_dir / f"binom_coverage_normal_n{n}.png",
        color="purple",
    )

    # (optionnel) si tu veux plus tard ajouter des graphes pour les unilatérales :
    # coverage_ecb    = df["coverage_ecb"].values
    # coverage_exact_unil  = df["coverage_exact_unil"].values
    # coverage_approx_unil = df["coverage_approx_unil"].values
