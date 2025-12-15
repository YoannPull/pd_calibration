# experiments/binom_coverage/plot_binom.py

from pathlib import Path
import pandas as pd

from experiments.plots.style import (
    new_figure,
    finalize_ax,
    save_figure,
    METHOD_STYLES,
)


def plot_coverage_vs_p_by_method_and_n(df, figs_dir: Path, prefix: str = "binom"):
    """
    Un graphique par méthode et par n : coverage vs p
    (une seule courbe par figure).
    """
    conf = df["conf_level"].iloc[0]
    ns = sorted(df["n"].unique())

    methods_cols = {
        "jeffreys": "coverage_jeff",
        "cp": "coverage_exact",
        "normal": "coverage_approx",
    }

    for n in ns:
        sub_n = df[df["n"] == n].sort_values("p")

        for method_key, col in methods_cols.items():
            if col not in sub_n.columns:
                continue

            style = METHOD_STYLES.get(method_key, {})
            label = style.get("label", method_key)
            color = style.get("color", None)

            fig, ax = new_figure()
            ax.plot(
                sub_n["p"].values,
                sub_n[col].values,
                label=label,
                color=color,
            )

            finalize_ax(
                ax,
                xlabel="p",
                ylabel="Coverage probability",
                title=f"{label} — Coverage vs p (n={n})",
                nominal_level=conf,
            )

            out_path = figs_dir / f"{prefix}_coverage_{method_key}_n{n}.png"
            save_figure(fig, out_path)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "binom_coverage_all_n.csv"
    df = pd.read_csv(csv_path)

    plot_coverage_vs_p_by_method_and_n(df, figs_dir=figs_dir, prefix="binom")
