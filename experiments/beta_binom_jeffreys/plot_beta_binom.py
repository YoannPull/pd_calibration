# experiments/beta_binom_jeffreys/plot_beta_binom.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

from experiments.plots.style import (
    new_figure,
    finalize_ax,
    save_figure,
    METHOD_STYLES,
)


def plot_coverage_vs_p_all_methods_all_combinations(
    df: pd.DataFrame,
    save_dir: Path | None = None,
    prefix: str = "beta_binom",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = (0.80, 1.02),
):
    conf_levels = df["conf_level"].unique()
    if len(conf_levels) != 1:
        print("Warning: multiple conf_levels in df; using the first one.")
    conf = float(conf_levels[0])

    ns = sorted(df["n"].unique())
    rhos = sorted(df["rho"].unique())

    for n in ns:
        for rho in rhos:
            sub = df[(df["n"] == n) & (df["rho"] == rho)].copy()
            if sub.empty:
                continue

            sub = sub.sort_values("p_true")

            fig, ax = new_figure()

            for method in sorted(sub["method"].unique()):
                tmp = sub[sub["method"] == method]
                x_p = tmp["p_true"].values
                y_cov = tmp["coverage"].values

                style = METHOD_STYLES.get(method, {})
                label = style.get("label", method)
                color = style.get("color", None)

                ax.plot(
                    x_p,
                    y_cov,
                    label=label,
                    color=color,
                    linewidth=2.0,
                    solid_capstyle="round",
                    zorder=3,
                )

                # debug: min coverage
                min_cov = float(y_cov.min())
                idx_min = int(y_cov.argmin())
                p_at_min = float(x_p[idx_min])
                np_at_min = n * p_at_min
                print(
                    f"[n={n}, rho={rho}, method={method}] "
                    f"min coverage = {min_cov:.4f} / p = {p_at_min:.6f} / n*p = {np_at_min:.4f}"
                )

            rho_str = str(rho).replace(".", "_")

            finalize_ax(
                ax,
                xlabel=r"$p$",
                ylabel="Coverage probability",
                title=f"Coverage vs $p$ (n={n}, rho={rho})",
                nominal_level=conf,
                nominal_label=f"Nominal level ({conf:.0%})",
                xlim=xlim,
                ylim=ylim,
                add_legend=True,
                legend_loc="lower right",
            )

            if save_dir is not None:
                fname = f"{prefix}_coverage_all_methods_n{n}_rho{rho_str}.png"
                out_path = save_dir / fname
                save_figure(fig, out_path, also_pdf=True)
            else:
                import matplotlib.pyplot as plt
                plt.show()
                plt.close(fig)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df_path = data_dir / "beta_binom_results.csv"
    df = pd.read_csv(df_path)

    plot_coverage_vs_p_all_methods_all_combinations(
        df,
        save_dir=figs_dir,
        prefix="b_b",
        xlim=None,
        ylim=(0.80, 1.02),
    )
