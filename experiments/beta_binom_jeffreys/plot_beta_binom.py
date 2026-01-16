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

# Default (legacy) rho targets for "curves" design selection
RHO_TARGETS: list[float] = [0.00, 0.01, 0.05, 0.10]


def _select_rhos(available: Iterable[float], targets: list[float]) -> list[float]:
    avail = sorted({float(x) for x in available})
    keep: list[float] = []
    for t in targets:
        if any(np.isclose(t, a) for a in avail):
            a_match = min(avail, key=lambda a: abs(a - t))
            if not any(np.isclose(a_match, k) for k in keep):
                keep.append(a_match)
    return sorted(keep) if keep else avail


def _rho_to_str(rho: float) -> str:
    return f"{rho:.3f}".rstrip("0").rstrip(".").replace(".", "_")


def _load_df(data_dir: Path) -> pd.DataFrame:
    """
    Priority:
      1) beta_binom_results.csv (combined) if it exists
      2) else try individual files
    """
    combined = data_dir / "beta_binom_results.csv"
    if combined.exists():
        return pd.read_csv(combined)

    curves = data_dir / "beta_binom_results_curves.csv"
    scen = data_dir / "beta_binom_results_scenarios.csv"
    dfs = []
    if curves.exists():
        dfs.append(pd.read_csv(curves))
    if scen.exists():
        dfs.append(pd.read_csv(scen))
    if not dfs:
        raise FileNotFoundError(f"No beta-binomial results found in {data_dir}")
    return pd.concat(dfs, ignore_index=True)


def _split_designs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_curves, df_scenarios).
    If no 'design' column exists, treat all as curves (legacy).
    """
    if "design" not in df.columns:
        return df.copy(), pd.DataFrame(columns=df.columns)

    df_curves = df[df["design"] == "curves"].copy()
    df_scen = df[df["design"] == "scenarios"].copy()
    return df_curves, df_scen


def _sanity_report(df: pd.DataFrame) -> None:
    """Quick checks to distinguish plotting artifacts from real simulation bugs."""
    if df.empty:
        print("[INFO] Empty dataframe — nothing to plot.")
        return

    required = {"method", "coverage", "reject_star_rate"}
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"[WARN] Missing columns for sanity checks: {missing}")
        return

    err = (1.0 - df["coverage"].astype(float) - df["reject_star_rate"].astype(float)).abs()
    print(f"[CHECK] max |(1-coverage) - reject_star_rate| = {float(err.max()):.4e}")
    print(f"[CHECK] mean |(1-coverage) - reject_star_rate| = {float(err.mean()):.4e}")

    by_method = (
        df.groupby("method")[["coverage", "reject_star_rate"]]
        .agg(["min", "max"])
        .sort_index()
    )
    print("[INFO] coverage / reject ranges by method:")
    with pd.option_context("display.max_rows", 50, "display.width", 120):
        print(by_method)


# -----------------------------
# Identical-curve collapsing
# -----------------------------
def _style_label(method: str) -> str:
    style = METHOD_STYLES.get(method, {})
    return str(style.get("label", method))


def _group_identical_curves(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    atol: float = 1e-12,
) -> list[list[str]]:
    """
    Group methods whose curves are identical (same x and y allclose).
    curves: method -> (x, y)
    returns: list of groups, each group is a list of methods (first is representative).
    """
    methods = list(curves.keys())
    groups: list[list[str]] = []

    for m in methods:
        x, y = curves[m]
        placed = False
        for g in groups:
            m0 = g[0]
            x0, y0 = curves[m0]
            if x.shape == x0.shape and y.shape == y0.shape:
                if np.allclose(x, x0, atol=0.0, rtol=0.0) and np.allclose(y, y0, atol=atol, rtol=0.0):
                    g.append(m)
                    placed = True
                    break
        if not placed:
            groups.append([m])

    # stable ordering
    groups = [sorted(g) for g in groups]
    groups.sort(key=lambda g: g[0])
    return groups


def _collapsed_label(group: list[str]) -> str:
    rep = group[0]
    base = _style_label(rep)
    if len(group) == 1:
        return base
    others = [m for m in group[1:]]
    others_lbl = ", ".join(_style_label(m) for m in others)
    return f"{base} (≡ {others_lbl})"


def plot_curves_all_methods_by_n_and_rho(
    df: pd.DataFrame,
    save_dir: Path,
    prefix: str = "beta_binom",
    xlim: tuple[float, float] | None = None,
    ylim_cov: tuple[float, float] = (0.0, 1.02),
    ylim_rej: tuple[float, float] = (0.0, 1.0),
    rho_targets: list[float] | None = None,
    annotate_global_min: bool = True,
    collapse_identical: bool = True,
    collapse_atol: float = 1e-12,
):
    """
    CURVES design:
    For each (n, rho), plot vs p_true:
      - coverage (all methods)
      - reject_star_rate (all methods)

    If collapse_identical=True, identical curves are plotted once and the legend indicates equivalence.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    conf_levels = df["conf_level"].unique()
    conf = float(conf_levels[0]) if len(conf_levels) else 0.95
    alpha = 1.0 - conf

    ns = sorted(df["n"].unique())
    rhos = _select_rhos(sorted(df["rho"].unique()), rho_targets or RHO_TARGETS)

    for n in ns:
        for rho in rhos:
            sub = df[(df["n"] == n) & (np.isclose(df["rho"], rho))].copy()
            if sub.empty:
                continue
            sub = sub.sort_values("p_true")

            # -------------------
            # (1) Coverage figure
            # -------------------
            fig, ax = new_figure()

            # collect curves
            cov_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for method in sorted(sub["method"].unique()):
                tmp = sub[sub["method"] == method].copy()
                if tmp.empty:
                    continue
                x = tmp["p_true"].to_numpy(dtype=float)
                y = tmp["coverage"].to_numpy(dtype=float)
                cov_curves[method] = (x, y)

            groups = (
                _group_identical_curves(cov_curves, atol=collapse_atol)
                if (collapse_identical and cov_curves)
                else [[m] for m in sorted(cov_curves.keys())]
            )

            global_min_y = np.inf
            global_min_p = np.nan
            global_min_series_x = None
            global_min_series_y = None
            global_min_idx = None

            for g in groups:
                rep = g[0]
                x, y = cov_curves[rep]

                style = METHOD_STYLES.get(rep, {})
                label = _collapsed_label(g)
                color = style.get("color", None)

                ax.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    linewidth=2.0,
                    solid_capstyle="round",
                    zorder=3,
                )

                if annotate_global_min:
                    i_min = int(np.argmin(y))
                    if float(y[i_min]) < float(global_min_y):
                        global_min_y = float(y[i_min])
                        global_min_p = float(x[i_min])
                        global_min_series_x = x
                        global_min_series_y = y
                        global_min_idx = i_min

            finalize_ax(
                ax,
                xlabel=r"True PD ($p$)",
                ylabel="Coverage probability",
                title=f"n={int(n)}, $\\rho$={float(rho):.3f} — Coverage vs $p$",
                nominal_level=conf,
                nominal_label=f"Nominal level ({conf:.0%})",
                xlim=xlim,
                ylim=ylim_cov,
                add_legend=True,
                legend_loc="lower right",
            )

            if annotate_global_min and global_min_series_x is not None:
                ylim_low_cov = float(ylim_cov[0])
                if global_min_y < ylim_low_cov:
                    annotate_min_below_ylim_at_crossing(
                        ax,
                        x=global_min_series_x,
                        y=global_min_series_y,
                        idx_min=int(global_min_idx),
                        ylim_low=ylim_low_cov,
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
            out_path = save_dir / f"{prefix}_curves_coverage_n{int(n)}_rho{rho_str}.png"
            save_figure(fig, out_path, also_pdf=False)

            # -----------------------
            # (2) Reject-rate figure
            # -----------------------
            fig2, ax2 = new_figure()

            rej_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for method in sorted(sub["method"].unique()):
                tmp = sub[sub["method"] == method].copy()
                if tmp.empty:
                    continue
                x = tmp["p_true"].to_numpy(dtype=float)
                y = tmp["reject_star_rate"].to_numpy(dtype=float)
                rej_curves[method] = (x, y)

            groups_rej = (
                _group_identical_curves(rej_curves, atol=collapse_atol)
                if (collapse_identical and rej_curves)
                else [[m] for m in sorted(rej_curves.keys())]
            )

            for g in groups_rej:
                rep = g[0]
                x, y = rej_curves[rep]

                style = METHOD_STYLES.get(rep, {})
                label = _collapsed_label(g)
                color = style.get("color", None)

                ax2.plot(
                    x,
                    y,
                    label=label,
                    color=color,
                    linewidth=2.0,
                    solid_capstyle="round",
                    zorder=3,
                )

            finalize_ax(
                ax2,
                xlabel=r"True PD ($p$)",
                ylabel="Rejection rate",
                title=f"n={int(n)}, $\\rho$={float(rho):.3f} — Rejection rate vs $p$",
                nominal_level=alpha,
                nominal_label=rf"Nominal size ($\alpha$={alpha:.0%})",
                xlim=xlim,
                ylim=ylim_rej,
                add_legend=True,
                legend_loc="upper right",
            )

            out_path2 = save_dir / f"{prefix}_curves_reject_n{int(n)}_rho{rho_str}.png"
            save_figure(fig2, out_path2, also_pdf=False)


def plot_scenarios_vs_rho(
    df: pd.DataFrame,
    save_dir: Path,
    prefix: str = "beta_binom",
    ylim_cov: tuple[float, float] = (0.0, 1.02),
    ylim_rej: tuple[float, float] = (0.0, 1.0),
    rho_targets: list[float] | None = None,
    use_all_rhos_if_none: bool = True,
    collapse_identical: bool = True,
    collapse_atol: float = 1e-12,
):
    """
    SCENARIOS design:
    For each scenario (fixed n,p_true), plot vs rho:
      - coverage (all methods)
      - reject_star_rate (all methods)

    If rho_targets is None and use_all_rhos_if_none=True, we use ALL available rhos (recommended for granular grids).

    If collapse_identical=True, identical curves are plotted once and the legend indicates equivalence.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    conf_levels = df["conf_level"].unique()
    conf = float(conf_levels[0]) if len(conf_levels) else 0.95
    alpha = 1.0 - conf

    avail_rhos = sorted(df["rho"].unique())
    if rho_targets is None and use_all_rhos_if_none:
        rhos = [float(x) for x in avail_rhos]
    else:
        rhos = _select_rhos(avail_rhos, rho_targets or RHO_TARGETS)

    if "scenario" in df.columns:
        scen_keys = sorted(df["scenario"].unique())
        scen_iter = [(s, df[df["scenario"] == s].copy()) for s in scen_keys]
    else:
        scen_iter = []
        for (n, p_true), g in df.groupby(["n", "p_true"], sort=True):
            name = f"n{int(n)}_p{float(p_true):.4f}".replace(".", "_")
            scen_iter.append((name, g.copy()))

    for name, sub in scen_iter:
        if sub.empty:
            continue

        # filter to selected rhos (avoid plotting “gaps” if df has more)
        sub = sub[sub["rho"].isin(rhos)].copy()
        if sub.empty:
            continue

        # Coverage vs rho
        fig, ax = new_figure()

        cov_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for method in sorted(sub["method"].unique()):
            tmp = sub[sub["method"] == method].copy()
            if tmp.empty:
                continue
            tmp = tmp.sort_values("rho")
            x = tmp["rho"].to_numpy(dtype=float)
            y = tmp["coverage"].to_numpy(dtype=float)
            cov_curves[method] = (x, y)

        groups = (
            _group_identical_curves(cov_curves, atol=collapse_atol)
            if (collapse_identical and cov_curves)
            else [[m] for m in sorted(cov_curves.keys())]
        )

        for g in groups:
            rep = g[0]
            x, y = cov_curves[rep]
            style = METHOD_STYLES.get(rep, {})
            label = _collapsed_label(g)
            color = style.get("color", None)
            ax.plot(x, y, label=label, color=color, linewidth=2.0, solid_capstyle="round", zorder=3)

        finalize_ax(
            ax,
            xlabel=r"Clustering ($\rho$)",
            ylabel="Coverage probability",
            title=f"{name} — Coverage vs $\\rho$",
            nominal_level=conf,
            nominal_label=f"Nominal level ({conf:.0%})",
            xlim=(min(rhos), max(rhos)) if rhos else None,
            ylim=ylim_cov,
            add_legend=True,
            legend_loc="lower right",
        )

        out_path = save_dir / f"{prefix}_scen_coverage_{name}.png"
        save_figure(fig, out_path, also_pdf=False)

        # Reject vs rho
        fig2, ax2 = new_figure()

        rej_curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for method in sorted(sub["method"].unique()):
            tmp = sub[sub["method"] == method].copy()
            if tmp.empty:
                continue
            tmp = tmp.sort_values("rho")
            x = tmp["rho"].to_numpy(dtype=float)
            y = tmp["reject_star_rate"].to_numpy(dtype=float)
            rej_curves[method] = (x, y)

        groups_rej = (
            _group_identical_curves(rej_curves, atol=collapse_atol)
            if (collapse_identical and rej_curves)
            else [[m] for m in sorted(rej_curves.keys())]
        )

        for g in groups_rej:
            rep = g[0]
            x, y = rej_curves[rep]
            style = METHOD_STYLES.get(rep, {})
            label = _collapsed_label(g)
            color = style.get("color", None)
            ax2.plot(x, y, label=label, color=color, linewidth=2.0, solid_capstyle="round", zorder=3)

        finalize_ax(
            ax2,
            xlabel=r"Clustering ($\rho$)",
            ylabel="Rejection rate",
            title=f"{name} — Rejection rate vs $\\rho$",
            nominal_level=alpha,
            nominal_label=rf"Nominal size ($\alpha$={alpha:.0%})",
            xlim=(min(rhos), max(rhos)) if rhos else None,
            ylim=ylim_rej,
            add_legend=True,
            legend_loc="upper right",
        )

        out_path2 = save_dir / f"{prefix}_scen_reject_{name}.png"
        save_figure(fig2, out_path2, also_pdf=False)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df_all = _load_df(data_dir)
    _sanity_report(df_all)
    df_curves, df_scen = _split_designs(df_all)

    # 1) Curves (coverage/reject vs p_true)
    if not df_curves.empty:
        plot_curves_all_methods_by_n_and_rho(
            df_curves,
            save_dir=figs_dir,
            prefix="beta_binom",
            xlim=None,
            ylim_cov=(0.0, 1.02),
            ylim_rej=(0.0, 1.0),
            rho_targets=RHO_TARGETS,
            annotate_global_min=True,
            collapse_identical=True,
            collapse_atol=1e-12,
        )

        plot_curves_all_methods_by_n_and_rho(
            df_curves,
            save_dir=figs_dir,
            prefix="beta_binom_zoom",
            xlim=None,
            ylim_cov=(0.80, 1.02),
            ylim_rej=(0.00, 0.30),
            rho_targets=RHO_TARGETS,
            annotate_global_min=True,
            collapse_identical=True,
            collapse_atol=1e-12,
        )

    # 2) Scenarios (coverage/reject vs rho)
    if not df_scen.empty:
        # Here: rho_targets=None + use_all_rhos_if_none=True => use full granular rho grid
        plot_scenarios_vs_rho(
            df_scen,
            save_dir=figs_dir,
            prefix="beta_binom",
            ylim_cov=(0.0, 1.02),
            ylim_rej=(0.0, 1.0),
            rho_targets=None,
            use_all_rhos_if_none=True,
            collapse_identical=True,
            collapse_atol=1e-12,
        )

        plot_scenarios_vs_rho(
            df_scen,
            save_dir=figs_dir,
            prefix="beta_binom_zoom",
            ylim_cov=(0.80, 1.02),
            ylim_rej=(0.00, 0.30),
            rho_targets=None,
            use_all_rhos_if_none=True,
            collapse_identical=True,
            collapse_atol=1e-12,
        )
