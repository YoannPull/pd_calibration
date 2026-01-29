from __future__ import annotations

from pathlib import Path
import re
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


# -----------------------------------------------------------------------------
# RMSE export helpers (CSV + LaTeX macros)
# -----------------------------------------------------------------------------
def _latex_macro_safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s))


def _latex_escape(s: str) -> str:
    # minimal escaping so the macro content is safe in captions
    s = str(s)
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("$", r"\$")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("{", r"\{")
        .replace("}", r"\}")
        .replace("~", r"\textasciitilde{}")
        .replace("^", r"\textasciicircum{}")
    )


def _macro_name(prefix: str, metric: str, context: str) -> str:
    # Example: \RMSEpriorsenstwosidedn100focusjeffreys
    return "{\RMSE" + _latex_macro_safe(prefix) + _latex_macro_safe(metric) + _latex_macro_safe(context) + "}"


def _write_rmse_exports(records: list[dict], figs_dir: Path, basename: str = "rmse_summary") -> None:
    """
    Writes:
      - rmse_summary.csv (long format: one row per figure x prior_focus x metric)
      - rmse_summary_macros.tex (one macro per figure/metric; value = single RMSE number)
    """
    figs_dir.mkdir(parents=True, exist_ok=True)

    df_rmse = pd.DataFrame.from_records(records)
    df_rmse = df_rmse.sort_values(["prefix", "metric", "context", "prior_focus"], kind="mergesort")
    df_rmse.to_csv(figs_dir / f"{basename}.csv", index=False)

    tex_path = figs_dir / f"{basename}_macros.tex"
    lines: list[str] = [
        "% Auto-generated RMSE macros (do not edit by hand)",
        "% Usage: \\input{<path>/rmse_summary_macros.tex}",
        "",
    ]
    for r in records:
        macro = _macro_name(r["prefix"], r["metric"], r["context"])
        val = float(r["rmse"]) if np.isfinite(float(r["rmse"])) else np.nan
        if np.isfinite(val):
            lines.append(f"\\newcommand{macro}{{{val:.3f}}}")
        else:
            lines.append(f"\\newcommand{macro}{{nan}}")

    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Two-sided coverage plots (per focus prior) with proper min annotation
# Titles removed + RMSE exported (no box on figure)
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
) -> list[dict]:
    figs_dir.mkdir(parents=True, exist_ok=True)

    conf = float(df["conf_level"].iloc[0])
    ns = sorted(df["n"].unique())
    priors = sorted(df["prior"].unique())

    records: list[dict] = []

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

            # finalize axis (adds nominal line + styling) -- TITLE REMOVED
            finalize_ax(
                ax,
                xlabel="True default probability (p)",
                ylabel="Coverage probability",
                title=None,  # <-- remove titles
                nominal_level=conf,
                nominal_label=f"Nominal level ({conf:.0%})",
                xlim=xlim,
                ylim=ylim,
                add_legend=True,
            )

            # RMSE (export only) on focus curve over x-window
            mask = (p >= xlim[0]) & (p <= xlim[1])
            rmse = _rmse_to_nominal(y, nominal=conf, mask=mask)

            context = f"twoSided_n{int(n)}_focus{_safe_slug(prior_focus)}"
            records.append(
                {
                    "prefix": prefix,
                    "metric": "two_sided_coverage",
                    "context": context,
                    "n": int(n),
                    "prior_focus": str(prior_focus),
                    "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
                    "nominal": float(conf),
                    "xlim_low": float(xlim[0]),
                    "xlim_high": float(xlim[1]),
                    "figure": f"{prefix}_two_sided_n{int(n)}__focus_{_safe_slug(prior_focus)}.png",
                }
            )

            # --- Min annotation (same logic as plot_binom.py) ---
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

    return records


# -----------------------------------------------------------------------------
# One-sided conservatism plots (per focus prior)
# Titles removed + RMSE exported (no box on figure)
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
) -> list[dict]:
    figs_dir.mkdir(parents=True, exist_ok=True)

    ns = sorted(df["n"].unique())
    priors = sorted(df["prior"].unique())

    records: list[dict] = []

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

            # finalize axis (reference line at 0.5) -- TITLE REMOVED
            ref = 0.5
            finalize_ax(
                ax,
                xlabel="True default probability (p)",
                ylabel="Conservatism: E[ P(P ≤ p_true | data) ]",
                title=None,  # <-- remove titles
                nominal_level=ref,
                nominal_label="Reference (0.5)",
                xlim=xlim,
                ylim=ylim if ylim is not None else (0.0, 1.0),
                add_legend=True,
            )

            # RMSE (export only)
            mask = (p >= xlim[0]) & (p <= xlim[1])
            rmse = _rmse_to_nominal(y, nominal=ref, mask=mask)

            context = f"upperCons_n{int(n)}_focus{_safe_slug(prior_focus)}"
            records.append(
                {
                    "prefix": prefix,
                    "metric": "upper_conservatism",
                    "context": context,
                    "n": int(n),
                    "prior_focus": str(prior_focus),
                    "rmse": float(rmse) if np.isfinite(rmse) else np.nan,
                    "nominal": float(ref),
                    "xlim_low": float(xlim[0]),
                    "xlim_high": float(xlim[1]),
                    "figure": f"{prefix}_upper_conservatism_n{int(n)}__focus_{_safe_slug(prior_focus)}.png",
                }
            )

            out = figs_dir / f"{prefix}_upper_conservatism_n{int(n)}__focus_{_safe_slug(prior_focus)}.png"
            save_figure(fig, out, also_pdf=False)

    return records


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

    gray_alpha = 0.40

    all_records: list[dict] = []

    all_records += plot_two_sided_per_prior(
        df,
        figs_dir=figs_dir,
        xlim=(0.0, 0.10),
        ylim=(0.80, 1.02),
        prefix="prior_sens",
        gray_alpha=gray_alpha,
        min_annot_threshold=0.80,
    )

    all_records += plot_upper_conservatism_per_prior(
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

        all_records += plot_two_sided_per_prior(
            df_ldp,
            figs_dir=figs_dir,
            xlim=(0.0001, 0.005),
            ylim=(0.50, 1.02),
            prefix="prior_sens_ldp",
            gray_alpha=gray_alpha,
            min_annot_threshold=0.80,
        )

        all_records += plot_upper_conservatism_per_prior(
            df_ldp,
            figs_dir=figs_dir,
            xlim=(0.0001, 0.005),
            ylim=(0.0, 1.0),
            prefix="prior_sens_ldp",
            gray_alpha=gray_alpha,
        )
    else:
        print(f"[WARN] Missing: {ldp_path}")

    # Export RMSE (CSV + LaTeX macros)
    _write_rmse_exports(all_records, figs_dir=figs_dir, basename="rmse_summary")
    print(
        f"[OK] RMSE exported to: {figs_dir / 'rmse_summary.csv'} and {figs_dir / 'rmse_summary_macros.tex'}"
    )
