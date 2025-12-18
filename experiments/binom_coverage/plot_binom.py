# experiments/binom_coverage/plot_binom.py
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
    METHOD_STYLES,
)


def plot_coverage_vs_p_by_method_and_n(
    df: pd.DataFrame,
    figs_dir: Path,
    prefix: str = "binom",
    xlim: tuple[float, float] = (0.0, 0.10),
    ylim: tuple[float, float] = (0.80, 1.02),
):
    conf = float(df["conf_level"].iloc[0])
    ns = sorted(df["n"].unique())

    methods_cols = {
        "jeffreys": "coverage_jeff",
        "cp": "coverage_exact",
        "normal": "coverage_approx",
        # if present
        "jeffreys_ecb": "coverage_ecb",
        "cp_upper": "coverage_exact_unil",
        "normal_upper": "coverage_approx_unil",
    }

    ylim_low = float(ylim[0])
    figs_dir.mkdir(parents=True, exist_ok=True)

    for n in ns:
        sub = df[df["n"] == n].sort_values("p")
        p = sub["p"].to_numpy()

        for method_key, col in methods_cols.items():
            if col not in sub.columns:
                continue

            y = sub[col].to_numpy()

            style = METHOD_STYLES.get(method_key, {})
            label = style.get("label", method_key)
            color = style.get("color", None)

            fig, ax = new_figure()
            ax.plot(
                p,
                y,
                color=color,
                linewidth=1.8,             # <- un poil plus fin
                solid_capstyle="round",
                zorder=3,
            )

            finalize_ax(
                ax,
                xlabel="True default probability (p)",
                ylabel="Coverage probability",
                title=f"{label} — Coverage vs p (n={int(n)})",
                nominal_level=conf,
                nominal_label=f"Nominal level ({conf:.0%})",
                xlim=xlim,
                ylim=ylim,
                add_legend=False,
            )

            # --- Min annotation (with smart arrow if clipped) ---
            mask = (p >= xlim[0]) & (p <= xlim[1])
            if np.any(mask):
                p_m = p[mask]
                y_m = y[mask]
                idx_min = int(np.argmin(y_m))
                min_p = float(p_m[idx_min])
                min_y = float(y_m[idx_min])

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
                        f"Minimum coverage: {min_y:.3f} (p={min_p:.3f})")


            out_path = figs_dir / f"{prefix}_coverage_{method_key}_n{int(n)}.png"
            save_figure(fig, out_path, also_pdf=False)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "binom_coverage_all_n.csv"
    df = pd.read_csv(csv_path)

    plot_coverage_vs_p_by_method_and_n(
        df,
        figs_dir=figs_dir,
        prefix="binom",
        xlim=(0.0, 0.10),
        ylim=(0.80, 1.02),
    )
