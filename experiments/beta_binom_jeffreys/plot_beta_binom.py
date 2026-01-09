# experiments/beta_binom_jeffreys/plot_beta_binom.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.plots.style import (
    new_figure,
    finalize_ax,
    save_figure,
    annotate_min,
    annotate_min_below_ylim_at_crossing,
    METHOD_STYLES,
)

# ---------------------------------------------------------------------
# What rhos to keep for a "robustness" section (adds value without flooding)
# - 0.00 : i.i.d. Binomial benchmark (sanity check / link to previous section)
# - 0.01 : mild over-dispersion (realistic heterogeneity)
# - 0.05 : moderate clustering (portfolio concentration / latent segments)
# - 0.10 : severe stress (optional, but informative as an upper-bound stress)
#
# If your CSV does not contain some of them, the code will just keep those available.
# ---------------------------------------------------------------------
RHO_TARGETS: list[float] = [0.00, 0.01, 0.05, 0.10]


def _select_rhos(available: Iterable[float], targets: list[float]) -> list[float]:
    """Keep only the 'targets' that exist in the data (within isclose)."""
    avail = sorted({float(x) for x in available})
    keep: list[float] = []
    for t in targets:
        if any(np.isclose(t, a) for a in avail):
            # pick the matching value from avail to keep exact printed value
            a_match = min(avail, key=lambda a: abs(a - t))
            if not any(np.isclose(a_match, k) for k in keep):
                keep.append(a_match)
    # If nothing matched, don't filter (better than silently dropping everything)
    return sorted(keep) if keep else avail


def _rho_to_str(rho: float) -> str:
    return f"{rho:.3f}".rstrip("0").rstrip(".").replace(".", "_")


def plot_coverage_vs_p_all_methods_by_n_and_rho(
    df: pd.DataFrame,
    save_dir: Path,
    prefix: str = "beta_binom",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] = (0.80, 1.02),
    rho_targets: list[float] | None = None,
    annotate_global_min: bool = True,
):
    """
    For each (n, rho), plot coverage vs p_true for all methods on the same axes.
    Adds a 'smart' min annotation if the minimum coverage is below ylim (arrow to crossing),
    similar to experiments/binom_coverage/plot_binom.py.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    conf_levels = df["conf_level"].unique()
    if len(conf_levels) != 1:
        print("[WARN] Multiple conf_level values found; using the first one.")
    conf = float(conf_levels[0])

    ns = sorted(df["n"].unique())
    rhos_avail = sorted(df["rho"].unique())
    rhos = _select_rhos(rhos_avail, rho_targets or RHO_TARGETS)

    ylim_low = float(ylim[0])

    for n in ns:
        for rho in rhos:
            sub = df[(df["n"] == n) & (np.isclose(df["rho"], rho))].copy()
            if sub.empty:
                continue

            sub = sub.sort_values("p_true")

            fig, ax = new_figure()

            # Track global minimum (across methods) to avoid cluttering with many annotations
            global_min_y = np.inf
            global_min_p = np.nan
            global_min_series_x = None
            global_min_series_y = None
            global_min_idx = None

            for method in sorted(sub["method"].unique()):
                tmp = sub[sub["method"] == method].copy()
                if tmp.empty:
                    continue

                x_p = tmp["p_true"].to_numpy(dtype=float)
                y_cov = tmp["coverage"].to_numpy(dtype=float)

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

                # compute method min inside xlim if provided (for debugging / global min)
                if xlim is not None:
                    mask = (x_p >= xlim[0]) & (x_p <= xlim[1])
                    if not np.any(mask):
                        continue
                    x_m = x_p[mask]
                    y_m = y_cov[mask]
                else:
                    x_m, y_m = x_p, y_cov

                idx_min = int(np.argmin(y_m))
                min_y = float(y_m[idx_min])
                min_p = float(x_m[idx_min])

                # Update global min across methods (the one we will annotate)
                if min_y < global_min_y:
                    global_min_y = min_y
                    global_min_p = min_p
                    global_min_series_x = x_m
                    global_min_series_y = y_m
                    global_min_idx = idx_min

                # Optional debug print
                np_at_min = float(n) * min_p
                print(
                    f"[n={int(n)}, rho={float(rho):.3f}, method={method}] "
                    f"min coverage = {min_y:.4f} / p = {min_p:.6f} / n*p = {np_at_min:.4f}"
                )

            finalize_ax(
                ax,
                xlabel=r"True default probability ($p$)",
                ylabel="Coverage probability",
                title=f"Coverage vs $p$ (n={int(n)}, $\\rho$={float(rho):.3f})",
                nominal_level=conf,
                nominal_label=f"Nominal level ({conf:.0%})",
                xlim=xlim,
                ylim=ylim,
                add_legend=True,
                legend_loc="lower right",
            )

            # --- Annotate global minimum (with smart arrow if clipped) ---
            if annotate_global_min and global_min_series_x is not None:
                if global_min_y < ylim_low:
                    annotate_min_below_ylim_at_crossing(
                        ax,
                        x=global_min_series_x,
                        y=global_min_series_y,
                        idx_min=int(global_min_idx),
                        ylim_low=ylim_low,
                        text_color="red",
                    )
                else:
                    annotate_min(
                        ax,
                        float(global_min_p),
                        float(global_min_y),
                        f"Minimum coverage: {global_min_y:.3f} (p={global_min_p:.4f})",
                    )

            rho_str = _rho_to_str(float(rho))
            out_path = save_dir / f"{prefix}_coverage_all_methods_n{int(n)}_rho{rho_str}.png"
            save_figure(fig, out_path, also_pdf=True)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df_path = data_dir / "beta_binom_results.csv"
    df = pd.read_csv(df_path)

    # Reasonable defaults:
    # - keep xlim=None to use full simulated p_true range (often [1e-4, 5e-3] in LDP runs)
    # - set ylim to highlight deviations; if you work on very low p, 0.80 is OK.
    plot_coverage_vs_p_all_methods_by_n_and_rho(
        df,
        save_dir=figs_dir,
        prefix="beta_binom",
        xlim=None,              # e.g., (0.0, 0.01) if you want consistent axes
        ylim=(0.80, 1.02),
        rho_targets=RHO_TARGETS,
        annotate_global_min=True,
    )
