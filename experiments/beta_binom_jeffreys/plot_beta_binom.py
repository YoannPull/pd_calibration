# experiments/beta_binom_jeffreys/plot_beta_binom.py
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_coverage_vs_rho(df, n, p_true, save_dir=None, prefix="fig_beta_binom"):
    sub = df[(df["n"] == n) & (df["p_true"] == p_true)]
    plt.figure()
    for method in sub["method"].unique():
        tmp = sub[sub["method"] == method].sort_values("rho")
        plt.plot(tmp["rho"], tmp["coverage"], marker="o", label=method)
    conf = sub["conf_level"].iloc[0]
    plt.axhline(y=conf, linestyle="--", label="nominal level")
    plt.xlabel(r"Intra-class correlation $\rho$")
    plt.ylabel("Empirical coverage of $p$")
    plt.title(f"Coverage vs $\\rho$ (n={n}, p_true={p_true})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_dir is not None:
        fname = f"{prefix}_coverage_n{n}_p{p_true}.png"
        out_path = save_dir / fname
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close()


def plot_length_vs_rho(df, n, p_true, save_dir=None, prefix="fig_beta_binom"):
    sub = df[(df["n"] == n) & (df["p_true"] == p_true)]
    plt.figure()
    for method in sub["method"].unique():
        tmp = sub[sub["method"] == method].sort_values("rho")
        plt.plot(tmp["rho"], tmp["avg_length"], marker="o", label=method)
    plt.xlabel(r"Intra-class correlation $\rho$")
    plt.ylabel("Average interval length")
    plt.title(f"Interval length vs $\\rho$ (n={n}, p_true={p_true})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_dir is not None:
        fname = f"{prefix}_length_n{n}_p{p_true}.png"
        out_path = save_dir / fname
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved {out_path}")
    else:
        plt.show()
    plt.close()


def plot_reject_rate_vs_rho(df, n, p_true, save_dir=None, prefix="fig_beta_binom"):
    sub = df[(df["n"] == n) & (df["p_true"] == p_true)]
    plt.figure()
    for method in sub["method"].unique():
        tmp = sub[sub["method"] == method].sort_values("rho")
        plt.plot(tmp["rho"], tmp["reject_star_rate"], marker="o", label=method)
    conf = sub["conf_level"].iloc[0]
    alpha = 1 - conf
    plt.axhline(y=alpha, linestyle="--", label="nominal $\~\\alpha$ (Binomial)")
    plt.xlabel(r"Intra-class correlation $\rho$")
    plt.ylabel(r"Rejection rate of $H_0:p=p^\star$ (calibrated case)")
    plt.title(f"Rejection vs $\\rho$ (n={n}, p_true={p_true})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_dir is not None:
        fname = f"{prefix}_reject_n{n}_p{p_true}.png"
        out_path = save_dir / fname
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

    df_path = data_dir / "beta_binom_results.csv"
    df = pd.read_csv(df_path)

    # Exemple : tracer pour n=100, p_true=0.01
    n = 100
    p_true = 0.01
    prefix = "beta_binom"

    plot_coverage_vs_rho(df, n=n, p_true=p_true, save_dir=figs_dir, prefix=prefix)
    plot_length_vs_rho(df, n=n, p_true=p_true, save_dir=figs_dir, prefix=prefix)
    plot_reject_rate_vs_rho(df, n=n, p_true=p_true, save_dir=figs_dir, prefix=prefix)
