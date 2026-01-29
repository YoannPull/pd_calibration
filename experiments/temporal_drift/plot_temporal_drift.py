# experiments/temporal_drift/plot_temporal_drift.py
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from experiments.plots.style import (
    finalize_ax,
    save_figure,
    METHOD_STYLES,
    add_drift_marker,
)


# -------------------------
# IO
# -------------------------
def _load_temporal_drift_df(data_dir: Path) -> pd.DataFrame:
    combined = data_dir / "temporal_drift_all_scenarios.csv"
    if combined.exists():
        return pd.read_csv(combined)

    files = sorted(data_dir.glob("temporal_drift_*.csv"))
    files = [p for p in files if p.name != "temporal_drift_all_scenarios.csv"]
    if not files:
        raise FileNotFoundError(
            f"No temporal drift CSV found in {data_dir}. "
            "Expected temporal_drift_all_scenarios.csv or temporal_drift_*.csv"
        )
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def _maybe_set_integer_xticks(ax, sub_df: pd.DataFrame):
    ts = sorted(sub_df["t"].unique())
    if len(ts) <= 30:
        ax.set_xticks(ts)


# -------------------------
# Style helper (match style.py look)
# -------------------------
def _apply_axes_style(ax):
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# -------------------------
# RMSE helpers + exports (same logic as other scripts)
# -------------------------
def _rmse_to_nominal(y: np.ndarray, nominal: float, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        y = y[mask]
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - nominal) ** 2)))


def _latex_macro_safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s))


def _macro_name(prefix: str, metric: str, context: str) -> str:
    # Example: \RMSEtemporaldraftcoveragescenmycase
    return "{RMSE" + _latex_macro_safe(prefix) + _latex_macro_safe(metric) + _latex_macro_safe(context) + "}"


def _write_rmse_exports(records: list[dict], figs_dir: Path, basename: str = "rmse_summary") -> None:
    """
    Writes:
      - rmse_summary.csv (one row per (scenario, metric, method group rep))
      - rmse_summary_macros.tex (one macro per (scenario, metric, rep_method) with numeric RMSE)
    """
    figs_dir.mkdir(parents=True, exist_ok=True)

    df_rmse = pd.DataFrame.from_records(records)
    df_rmse = df_rmse.sort_values(["prefix", "metric", "scenario_slug", "rep_method"], kind="mergesort")
    df_rmse.to_csv(figs_dir / f"{basename}.csv", index=False)

    tex_path = figs_dir / f"{basename}_macros.tex"
    lines: list[str] = [
        "% Auto-generated RMSE macros (do not edit by hand)",
        "% Usage: \\input{<path>/rmse_summary_macros.tex}",
        "% Macros are per scenario+metric+method rep: \\RMSE<prefix><metric><scenarioSlug><methodRep>",
        "",
    ]
    for r in records:
        context = f"{r['scenario_slug']}{r['rep_method']}"
        macro = _macro_name(r["prefix"], r["metric"], context)
        val = float(r["rmse"]) if np.isfinite(float(r["rmse"])) else np.nan
        if np.isfinite(val):
            lines.append(f"\\newcommand\\{macro}{{{val:.3f}}}")
        else:
            lines.append(f"\\newcommand\\{macro}{{nan}}")

    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -------------------------
# Y-lims helpers
# -------------------------
def _centered_ylim_around(
    target: float,
    series: pd.Series,
    min_halfspan: float,
    max_halfspan: float,
) -> tuple[float, float]:
    s = series.dropna().astype(float)
    if s.empty:
        return (max(0.0, target - min_halfspan), min(1.0, target + min_halfspan))

    max_dev = float((s - target).abs().quantile(0.98))
    half = max(min_halfspan, min(max_halfspan, max_dev + 0.01))
    return (max(0.0, target - half), min(1.0, target + half))


def _robust_ylim(series: pd.Series, pad: float, lo: float, hi: float) -> tuple[float, float]:
    s = series.dropna().astype(float)
    if s.empty:
        return (lo, hi)

    q1 = float(s.quantile(0.02))
    q2 = float(s.quantile(0.98))
    if q1 == q2:
        q1 = float(s.min())
        q2 = float(s.max())

    y0 = max(lo, q1 - pad)
    y1 = min(hi, q2 + pad)

    if y1 - y0 < 1e-6:
        y0 = max(lo, y0 - 0.01)
        y1 = min(hi, y1 + 0.01)

    return (y0, y1)


# -------------------------
# Overlap handling (curves superposed)
# -------------------------
def _method_label(method: str) -> str:
    st = METHOD_STYLES.get(method, {})
    return st.get("label", method)


def _short_label(label: str) -> str:
    out = label
    out = out.replace("Jeffreys equal-tailed", "Jeffreys")
    out = out.replace("Exact Clopper–Pearson", "CP").replace("Exact Clopper-Pearson", "CP")
    out = out.replace("Normal approximation", "Normal")
    return out


def _group_overlapping_methods(
    sub_df: pd.DataFrame,
    ycol: str,
    xcol: str = "t",
    atol: float = 1e-12,
) -> tuple[list[list[str]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Group methods whose (x,y) series are identical up to atol.
    Returns:
      - groups: list of lists of method names
      - series_by_method: dict method -> (x,y) arrays
    """
    methods = sorted(sub_df["method"].unique())
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for m in methods:
        s = sub_df[sub_df["method"] == m].sort_values(xcol)
        x = s[xcol].to_numpy(dtype=float)
        y = s[ycol].to_numpy(dtype=float)
        series[m] = (x, y)

    groups: list[list[str]] = []
    used: set[str] = set()

    for m in methods:
        if m in used:
            continue
        x0, y0 = series[m]
        grp = [m]
        used.add(m)

        for m2 in methods:
            if m2 in used:
                continue
            x2, y2 = series[m2]
            if (
                len(x2) == len(x0)
                and np.allclose(x2, x0, rtol=0.0, atol=0.0)
                and np.allclose(y2, y0, rtol=0.0, atol=atol)
            ):
                grp.append(m2)
                used.add(m2)

        groups.append(grp)

    return groups, series


def _format_label_with_overlaps(rep_method: str, group: list[str]) -> str:
    rep_label = _method_label(rep_method)
    others = [m for m in group if m != rep_method]
    if not others:
        return rep_label

    other_labels = [_short_label(_method_label(m)) for m in others]
    return f"{rep_label} (≡ {', '.join(other_labels)})"


def _rep_for_group(group: list[str]) -> str:
    return group[0]


def _scenario_slug(sub_df: pd.DataFrame, scenario: str) -> str:
    return (
        str(sub_df["scenario_slug"].iloc[0])
        if "scenario_slug" in sub_df.columns
        else scenario.lower().replace(" ", "_")
    )


# -------------------------
# Plots
# Titles removed + RMSE exported (no box on figure)
# -------------------------
def plot_reject_rate_vs_time_by_scenario(
    df: pd.DataFrame,
    save_dir: Path,
    prefix: str = "temporal_drift",
) -> list[dict]:
    save_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for scenario in sorted(df["scenario"].unique()):
        sub_df = df[df["scenario"] == scenario].copy()

        n_vals = sorted(sub_df["n"].unique())
        if len(n_vals) != 1:
            raise ValueError(f"Scenario '{scenario}' has multiple n values in data: {n_vals}")

        T0 = int(sub_df["T0"].iloc[0])
        T = int(sub_df["T"].iloc[0])
        alpha_nominal = float(sub_df["alpha_nominal"].iloc[0])
        p_hat = float(sub_df["p_hat"].iloc[0])
        n = int(sub_df["n"].iloc[0])

        fig, ax = plt.subplots(figsize=(10, 6))
        _apply_axes_style(ax)

        groups, series = _group_overlapping_methods(sub_df, ycol="reject_rate", xcol="t", atol=1e-12)
        for grp in groups:
            rep = _rep_for_group(grp)
            x, y = series[rep]

            st = METHOD_STYLES.get(rep, {})
            color = st.get("color", None)
            label = _format_label_with_overlaps(rep, grp)

            ax.plot(
                x,
                y,
                label=label,
                color=color,
                linewidth=1.8,
                solid_capstyle="round",
                zorder=3,
            )

            # RMSE export only (reject-rate to nominal alpha)
            rmse = _rmse_to_nominal(y, nominal=alpha_nominal, mask=None)

            records.append(
                {
                    "prefix": prefix,
                    "metric": "reject_rate",
                    "scenario": str(scenario),
                    "scenario_slug": _scenario_slug(sub_df, scenario),
                    "n": int(n),
                    "T0": int(T0),
                    "T": int(T),
                    "p_hat": float(p_hat),
                    "nominal": float(alpha_nominal),
                    "rep_method": str(rep),
                    "rep_label": _short_label(_method_label(rep)),
                    "collapsed_group": "|".join(grp),
                    "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
                    "figure": f"{prefix}_{_scenario_slug(sub_df, scenario)}_reject_rate_vs_time.png",
                }
            )

        add_drift_marker(ax, T0=T0, T=T)

        ylim_rej = _centered_ylim_around(alpha_nominal, sub_df["reject_rate"], min_halfspan=0.05, max_halfspan=0.25)

        finalize_ax(
            ax,
            xlabel="Time period t",
            ylabel=r"Rejection rate of $H_0: p = \hat p$",
            title=None,  # <-- remove titles
            nominal_level=alpha_nominal,
            nominal_label=f"Nominal level (α = {alpha_nominal:.2f})",
            add_legend=True,
            legend_loc="upper left",
            xlim=(1, T),
            ylim=ylim_rej,
        )

        _maybe_set_integer_xticks(ax, sub_df)

        slug = _scenario_slug(sub_df, scenario)
        save_figure(fig, save_dir / f"{prefix}_{slug}_reject_rate_vs_time.png", also_pdf=False)

    return records


def plot_length_vs_time_by_scenario(
    df: pd.DataFrame,
    save_dir: Path,
    prefix: str = "temporal_drift",
) -> list[dict]:
    save_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    length_col = "avg_length" if "avg_length" in df.columns else "len_mean"

    for scenario in sorted(df["scenario"].unique()):
        sub_df = df[df["scenario"] == scenario].copy()

        n_vals = sorted(sub_df["n"].unique())
        if len(n_vals) != 1:
            raise ValueError(f"Scenario '{scenario}' has multiple n values in data: {n_vals}")

        T0 = int(sub_df["T0"].iloc[0])
        T = int(sub_df["T"].iloc[0])
        p_hat = float(sub_df["p_hat"].iloc[0])
        n = int(sub_df["n"].iloc[0])

        fig, ax = plt.subplots(figsize=(10, 6))
        _apply_axes_style(ax)

        groups, series = _group_overlapping_methods(sub_df, ycol=length_col, xcol="t", atol=1e-12)
        for grp in groups:
            rep = _rep_for_group(grp)
            x, y = series[rep]

            st = METHOD_STYLES.get(rep, {})
            color = st.get("color", None)
            label = _format_label_with_overlaps(rep, grp)

            ax.plot(
                x,
                y,
                label=label,
                color=color,
                linewidth=1.8,
                solid_capstyle="round",
                zorder=3,
            )

            # RMSE export only (length to its own mean, since no nominal)
            # You can change nominal definition here if you prefer something else.
            nominal = float(np.nanmean(y)) if y.size else float("nan")
            rmse = _rmse_to_nominal(y, nominal=nominal, mask=None)

            records.append(
                {
                    "prefix": prefix,
                    "metric": "length",
                    "scenario": str(scenario),
                    "scenario_slug": _scenario_slug(sub_df, scenario),
                    "n": int(n),
                    "T0": int(T0),
                    "T": int(T),
                    "p_hat": float(p_hat),
                    "nominal": float(nominal) if np.isfinite(nominal) else np.nan,
                    "rep_method": str(rep),
                    "rep_label": _short_label(_method_label(rep)),
                    "collapsed_group": "|".join(grp),
                    "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
                    "figure": f"{prefix}_{_scenario_slug(sub_df, scenario)}_length_vs_time.png",
                }
            )

        add_drift_marker(ax, T0=T0, T=T)

        s_len = sub_df[length_col].dropna().astype(float)
        ylim_len = None
        if not s_len.empty:
            pad = 0.02 * float(s_len.quantile(0.98))
            hi = float(s_len.quantile(0.995)) * 1.10 if float(s_len.quantile(0.995)) > 0 else 1.0
            ylim_len = _robust_ylim(s_len, pad=pad, lo=0.0, hi=hi)

        finalize_ax(
            ax,
            xlabel="Time period t",
            ylabel="Average interval length",
            title=None,  # <-- remove titles
            nominal_level=None,
            add_legend=True,
            legend_loc="upper left",
            xlim=(1, T),
            ylim=ylim_len,
        )

        _maybe_set_integer_xticks(ax, sub_df)

        slug = _scenario_slug(sub_df, scenario)
        save_figure(fig, save_dir / f"{prefix}_{slug}_length_vs_time.png", also_pdf=False)

    return records


def plot_coverage_vs_time_by_scenario_with_min_scatter(
    df: pd.DataFrame,
    save_dir: Path,
    prefix: str = "temporal_drift",
) -> list[dict]:
    """
    Main panel: zoomed coverage vs time + drift marker (NO arrow annotations).
    Bottom panel: scatter of (t*, min coverage), colored by method.
    Overlap-aware: if two methods produce identical curves, we plot one and label equivalence.

    Titles removed + RMSE exported (coverage to nominal conf).
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for scenario in sorted(df["scenario"].unique()):
        sub_df = df[df["scenario"] == scenario].copy()

        n_vals = sorted(sub_df["n"].unique())
        if len(n_vals) != 1:
            raise ValueError(f"Scenario '{scenario}' has multiple n values in data: {n_vals}")

        T0 = int(sub_df["T0"].iloc[0])
        T = int(sub_df["T"].iloc[0])
        conf = float(sub_df["conf_level"].iloc[0])
        p_hat = float(sub_df["p_hat"].iloc[0])
        n = int(sub_df["n"].iloc[0])

        fig, (ax, ax_min) = plt.subplots(
            nrows=2,
            figsize=(10, 7.2),
            gridspec_kw={"height_ratios": [4.0, 1.4], "hspace": 0.06},
        )
        _apply_axes_style(ax)
        _apply_axes_style(ax_min)

        # Main axis zoom
        ylim_cov = _centered_ylim_around(
            target=conf,
            series=sub_df["coverage"],
            min_halfspan=0.06,
            max_halfspan=0.20,
        )

        groups, series = _group_overlapping_methods(sub_df, ycol="coverage", xcol="t", atol=1e-12)

        pts = []  # (min_t, min_y, color, short_label)
        for grp in groups:
            rep = _rep_for_group(grp)
            t, y = series[rep]

            st = METHOD_STYLES.get(rep, {})
            color = st.get("color", None)
            label = _format_label_with_overlaps(rep, grp)

            ax.plot(
                t,
                y,
                label=label,
                color=color,
                linewidth=1.8,
                solid_capstyle="round",
                zorder=3,
            )

            # RMSE export only
            rmse = _rmse_to_nominal(y, nominal=conf, mask=None)
            records.append(
                {
                    "prefix": prefix,
                    "metric": "coverage",
                    "scenario": str(scenario),
                    "scenario_slug": _scenario_slug(sub_df, scenario),
                    "n": int(n),
                    "T0": int(T0),
                    "T": int(T),
                    "p_hat": float(p_hat),
                    "nominal": float(conf),
                    "rep_method": str(rep),
                    "rep_label": _short_label(_method_label(rep)),
                    "collapsed_group": "|".join(grp),
                    "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
                    "figure": f"{prefix}_{_scenario_slug(sub_df, scenario)}_coverage_vs_time.png",
                }
            )

            idx_min = int(np.argmin(y))
            min_t = float(t[idx_min])
            min_y = float(y[idx_min])

            rep_short = _short_label(_method_label(rep))
            if len(grp) > 1:
                other_short = [_short_label(_method_label(m)) for m in grp[1:]]
                rep_short = f"{rep_short}≡{'+'.join(other_short)}"

            pts.append((min_t, min_y, color, rep_short))

        add_drift_marker(ax, T0=T0, T=T)

        finalize_ax(
            ax,
            xlabel="",  # x-label on bottom panel
            ylabel=r"Coverage of $p(t)$",
            title=None,  # <-- remove titles
            nominal_level=conf,
            nominal_label=f"Nominal coverage ({conf:.0%})",
            add_legend=True,
            legend_loc="lower left",
            xlim=(1, T),
            ylim=ylim_cov,
        )
        _maybe_set_integer_xticks(ax, sub_df)
        ax.tick_params(labelbottom=False)

        # ---- Bottom panel: (t*, min) colored by method ----
        for (tt, yy, cc, short) in pts:
            ax_min.scatter([tt], [yy], s=75, zorder=3, color=cc)
            ax_min.annotate(
                short,
                (tt, yy),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )

        ax_min.axhline(conf, color="black", linestyle="--", linewidth=1.2, alpha=0.8)
        onset = T0 + 0.5
        ax_min.axvline(x=onset, color="black", alpha=0.6, linestyle=(0, (3, 3)), linewidth=1.0)

        mins = np.array([p[1] for p in pts], dtype=float)
        y0 = max(0.0, float(mins.min()) - 0.04)
        y1 = min(1.0, float(max(conf, mins.max())) + 0.04)
        if y1 - y0 < 0.14:
            y0 = max(0.0, y0 - 0.06)
            y1 = min(1.0, y1 + 0.06)

        ax_min.set_xlim(1, T)
        ax_min.set_ylim(y0, y1)

        ax_min.set_xlabel("Time period t", fontsize=14)
        ax_min.set_ylabel("Min coverage", fontsize=12)
        ax_min.grid(alpha=0.18)

        _maybe_set_integer_xticks(ax_min, sub_df)

        slug = _scenario_slug(sub_df, scenario)
        save_figure(fig, save_dir / f"{prefix}_{slug}_coverage_vs_time.png", also_pdf=False)

    return records


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df = _load_temporal_drift_df(data_dir)

    prefix = "temporal_drift"

    all_records: list[dict] = []
    all_records += plot_reject_rate_vs_time_by_scenario(df, save_dir=figs_dir, prefix=prefix)
    all_records += plot_coverage_vs_time_by_scenario_with_min_scatter(df, save_dir=figs_dir, prefix=prefix)
    all_records += plot_length_vs_time_by_scenario(df, save_dir=figs_dir, prefix=prefix)

    _write_rmse_exports(all_records, figs_dir=figs_dir, basename="rmse_summary")
    print(
        f"[OK] RMSE exported to: {figs_dir / 'rmse_summary.csv'} and {figs_dir / 'rmse_summary_macros.tex'}"
    )
