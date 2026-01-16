from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from experiments.plots.style import (
    new_figure,
    finalize_ax,
    save_figure,
    annotate_min,
    annotate_min_below_ylim_at_crossing,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _safe_slug(s: str) -> str:
    s = s.lower()
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def _focus_color(prior_name: str) -> str | None:
    name = prior_name.lower()
    if "jeffreys" in name:
        return "tab:blue"
    if "laplace" in name:
        return "tab:orange"
    if "haldane" in name:
        return "tab:green"
    return None


def _rmse_to_nominal(y: np.ndarray, nominal: float, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        y = y[mask]
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - nominal) ** 2)))


def _add_rmse_box_big(ax, text: str) -> None:
    ax.text(
        0.985,
        0.02,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        fontweight="semibold",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.35", linewidth=1.2),
        zorder=10,
    )


# -----------------------------------------------------------------------------
# Two-sided coverage plots (per focus prior) with proper min annotation
# -----------------------------------------------------------------------------
def plot_two_sided_per_prior(
    df: pd.DataFrame,
    figs_dir: Path,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    prefix: str,
    gray_alpha: float = 0.22,
    gray_color: str = "0.55",
    min_annot_threshold: float = 0.80,
):
    figs_dir.mkdir(parents=True, exist_ok=True)

    conf = float(df["conf_level"].iloc[0])
    ns = sorted(df["n"].unique())
    priors = sorted(df["prior"].unique())

    for n in ns:
        sub_n = df[(df["n"] == n) & (df["tail"] == "two_sided")].copy()
        if sub_n.empty:
            continue
        sub_n = sub_n.sort_values("p")

        for prior_focus in priors:
            fig, ax = new_figure()

            # --- all priors in gray ---
            for prior_name in priors:
                s = sub_n[sub_n["prior"] == prior_name].sort_values("p")
                p_all = s["p"].to_numpy()
                y_all = s["coverage"].to_numpy()
                ax.plot(
                    p_all,
                    y_all,
                    linewidth=1.6,
                    alpha=gray_alpha,
                    color=gray_color,
                    zorder=2,
                    solid_capstyle="round",
                )

            # --- focus prior in color ---
            s_f = sub_n[sub_n["prior"] == prior_focus].sort_values("p")
            p = s_f["p"].to_numpy()
            y = s_f["coverage"].to_numpy()
            color_focus = _focus_color(prior_focus)

            ax.plot(
                p,
                y,
                linewidth=2.4,
                color=color_focus,
                zorder=5,
                solid_capstyle="round",
                label=prior_focus,
            )

            # finalize axis (adds nominal line + styling)
            finalize_ax(
                ax,
                xlabel="True default probability (p)",
                ylabel="Coverage probability",
                title=f"Prior sensitivity — two-sided coverage — n={int(n)}",
                nominal_level=conf,
                nominal_label=f"Nominal level ({conf:.0%})",
                xlim=xlim,
                ylim=ylim,
                add_legend=True,
            )

            # RMSE (big box) on focus curve over x-window
            mask = (p >= xlim[0]) & (p <= xlim[1])
            rmse = _rmse_to_nominal(y, nominal=conf, mask=mask)
            if np.isfinite(rmse):
                _add_rmse_box_big(ax, f"RMSE to nominal: {rmse:.3f}")

            # --- Min annotation (EXACTLY like plot_binom.py) ---
            if np.any(mask):
                p_m = p[mask]
                y_m = y[mask]
                idx_min = int(np.argmin(y_m))
                min_p = float(p_m[idx_min])
                min_y = float(y_m[idx_min])
                ylim_low = float(ylim[0])

                if np.isfinite(min_y) and (min_y < float(min_annot_threshold)):
                    if min_y < ylim_low:
                        annotate_min_below_ylim_at_crossing(
                            ax,
                            x=p_m,
                            y=y_m,
                            idx_min=idx_min,
                            ylim_low=ylim_low,
                            text_color="red",
                        )
                    else:
                        annotate_min(
                            ax,
                            min_p,
                            min_y,
                            f"Minimum coverage: {min_y:.3f} (p={min_p:.3f})",
                        )

            out = figs_dir / f"{prefix}_two_sided_n{int(n)}__focus_{_safe_slug(prior_focus)}.png"
            save_figure(fig, out, also_pdf=False)


# -----------------------------------------------------------------------------
# One-sided conservatism plots (per focus prior)
# -----------------------------------------------------------------------------
def plot_upper_conservatism_per_prior(
    df: pd.DataFrame,
    figs_dir: Path,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float] | None,
    prefix: str,
    gray_alpha: float = 0.22,
    gray_color: str = "0.55",
):
    figs_dir.mkdir(parents=True, exist_ok=True)

    ns = sorted(df["n"].unique())
    priors = sorted(df["prior"].unique())

    for n in ns:
        sub_n = df[(df["n"] == n) & (df["tail"] == "upper")].copy()
        if sub_n.empty:
            continue
        sub_n = sub_n.sort_values("p")

        for prior_focus in priors:
            fig, ax = new_figure()

            # all priors in gray
            for prior_name in priors:
                s = sub_n[sub_n["prior"] == prior_name].sort_values("p")
                p_all = s["p"].to_numpy()
                y_all = s["conservatism_mean"].to_numpy()
                ax.plot(
                    p_all,
                    y_all,
                    linewidth=1.6,
                    alpha=gray_alpha,
                    color=gray_color,
                    zorder=2,
                    solid_capstyle="round",
                )

            # focus prior in color
            s_f = sub_n[sub_n["prior"] == prior_focus].sort_values("p")
            p = s_f["p"].to_numpy()
            y = s_f["conservatism_mean"].to_numpy()
            color_focus = _focus_color(prior_focus)

            ax.plot(
                p,
                y,
                linewidth=2.4,
                color=color_focus,
                zorder=5,
                solid_capstyle="round",
                label=prior_focus,
            )

            # finalize axis (here: reference line at 0.5 as nominal)
            ref = 0.5
            finalize_ax(
                ax,
                xlabel="True default probability (p)",
                ylabel="Conservatism: E[ P(P ≤ p_true | data) ]",
                title=f"Prior sensitivity — one-sided conservatism — n={int(n)}",
                nominal_level=ref,
                nominal_label="Reference (0.5)",
                xlim=xlim,
                ylim=ylim if ylim is not None else (0.0, 1.0),
                add_legend=True,
            )

            # RMSE to 0.5 (big box)
            mask = (p >= xlim[0]) & (p <= xlim[1])
            rmse = _rmse_to_nominal(y, nominal=ref, mask=mask)
            if np.isfinite(rmse):
                _add_rmse_box_big(ax, f"RMSE to 0.5: {rmse:.3f}")

            out = figs_dir / f"{prefix}_upper_conservatism_n{int(n)}__focus_{_safe_slug(prior_focus)}.png"
            save_figure(fig, out, also_pdf=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # MAIN
    df = pd.read_csv(data_dir / "prior_sensibility_all_n.csv")

    gray_alpha=0.40

    plot_two_sided_per_prior(
        df,
        figs_dir=figs_dir,
        xlim=(0.0, 0.10),
        ylim=(0.80, 1.02),
        prefix="prior_sens",
        gray_alpha=gray_alpha,
        min_annot_threshold=0.80,
    )

    plot_upper_conservatism_per_prior(
        df,
        figs_dir=figs_dir,
        xlim=(0.0, 0.10),
        ylim=(0.0, 1.0),
        prefix="prior_sens",
        gray_alpha=gray_alpha,
    )

    # LDP / rare-event
    ldp_path = data_dir / "prior_sensibility_ldp_n1000.csv"
    if ldp_path.exists():
        df_ldp = pd.read_csv(ldp_path)

        plot_two_sided_per_prior(
            df_ldp,
            figs_dir=figs_dir,
            xlim=(0.0001, 0.005),
            ylim=(0.50, 1.02),
            prefix="prior_sens_ldp",
            gray_alpha=gray_alpha,
            min_annot_threshold=0.80,
        )

        plot_upper_conservatism_per_prior(
            df_ldp,
            figs_dir=figs_dir,
            xlim=(0.0001, 0.005),
            ylim=(0.0, 1.0),
            prefix="prior_sens_ldp",
            gray_alpha=gray_alpha,
        )
    else:
        print(f"[WARN] Missing: {ldp_path}")
