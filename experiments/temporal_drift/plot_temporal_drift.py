# experiments/temporal_drift/plot_temporal_drift.py

from pathlib import Path
import pandas as pd

from experiments.plots.style import (
    new_figure,
    finalize_ax,
    save_figure,
    METHOD_STYLES,
)


def _fmt_p_hat(p_hat: float) -> str:
    """
    Format court et safe pour les noms de fichiers.
    Ex: 0.01 -> "0p01"
    """
    s = f"{p_hat:.6g}"
    return s.replace(".", "p")


def plot_reject_rate_vs_time(df, save_dir=None, prefix: str = "temporal_drift"):
    """
    Taux de rejet de H0 : p = p_hat en fonction du temps.
    Si plusieurs p_hat sont présents, produit un plot par p_hat.
    """
    if "p_hat" in df.columns:
        p_hats = sorted(df["p_hat"].unique())
    else:
        p_hats = [None]

    for p_hat in p_hats:
        sub_df = df if p_hat is None else df[df["p_hat"] == p_hat]

        T0 = int(sub_df["T0"].iloc[0])
        T = int(sub_df["T"].iloc[0])
        alpha_nominal = float(sub_df["alpha_nominal"].iloc[0])

        fig, ax = new_figure()

        for method in sorted(sub_df["method"].unique()):
            sub = sub_df[sub_df["method"] == method].sort_values("t")
            style = METHOD_STYLES.get(method, {})
            label = style.get("label", method)
            color = style.get("color", None)

            ax.plot(
                sub["t"],
                sub["reject_rate"],
                label=label,
                color=color,
                linewidth=1.5,
            )

        ax.axvline(x=T0 + 0.5, color="red", linestyle="--", label="Drift onset")
        ax.axvspan(T0 + 0.5, T + 0.5, color="red", alpha=0.05)

        title = "Temporal evolution of rejection rates under drift"
        if p_hat is not None:
            title += f" (p_hat={p_hat:g})"

        finalize_ax(
            ax,
            xlabel="Time period t",
            ylabel=r"Rejection rate of $H_0: p = \hat p$",
            title=title,
            nominal_level=alpha_nominal,
            nominal_label=f"Nominal level (α = {alpha_nominal:.2f})",
        )

        if save_dir is not None:
            suffix = "" if p_hat is None else f"_phat{_fmt_p_hat(float(p_hat))}"
            out_path = save_dir / f"{prefix}_reject_rate_vs_time{suffix}.png"
            save_figure(fig, out_path)
        else:
            import matplotlib.pyplot as plt
            plt.show()
            plt.close(fig)


def plot_coverage_vs_time(df, save_dir=None, prefix: str = "temporal_drift"):
    """
    Couverture empirique de p_true(t) en fonction du temps.
    Si plusieurs p_hat sont présents, produit un plot par p_hat.
    """
    if "p_hat" in df.columns:
        p_hats = sorted(df["p_hat"].unique())
    else:
        p_hats = [None]

    for p_hat in p_hats:
        sub_df = df if p_hat is None else df[df["p_hat"] == p_hat]

        T0 = int(sub_df["T0"].iloc[0])
        T = int(sub_df["T"].iloc[0])
        conf = float(sub_df["conf_level"].iloc[0])

        fig, ax = new_figure()

        for method in sorted(sub_df["method"].unique()):
            sub = sub_df[sub_df["method"] == method].sort_values("t")
            style = METHOD_STYLES.get(method, {})
            label = style.get("label", method)
            color = style.get("color", None)

            ax.plot(
                sub["t"],
                sub["coverage"],
                label=label,
                color=color,
                linewidth=1.5,
            )

        ax.axvline(x=T0 + 0.5, color="red", linestyle="--", label="Drift onset")
        ax.axvspan(T0 + 0.5, T + 0.5, color="red", alpha=0.05)

        title = "Temporal evolution of coverage under drift"
        if p_hat is not None:
            title += f" (p_hat={p_hat:g})"

        finalize_ax(
            ax,
            xlabel="Time period t",
            ylabel=r"Coverage of $p(t)$",
            title=title,
            nominal_level=conf,
            nominal_label=f"Nominal coverage ({conf:.2f})",
        )

        if save_dir is not None:
            suffix = "" if p_hat is None else f"_phat{_fmt_p_hat(float(p_hat))}"
            out_path = save_dir / f"{prefix}_coverage_vs_time{suffix}.png"
            save_figure(fig, out_path)
        else:
            import matplotlib.pyplot as plt
            plt.show()
            plt.close(fig)


def plot_length_vs_time(df, save_dir=None, prefix: str = "temporal_drift"):
    """
    Longueur moyenne des intervalles en fonction du temps.
    Si plusieurs p_hat sont présents, produit un plot par p_hat.

    Remarque: on utilise 'avg_length' pour compat, mais si tu veux
    passer à 'len_mean', ça marche aussi.
    """
    if "p_hat" in df.columns:
        p_hats = sorted(df["p_hat"].unique())
    else:
        p_hats = [None]

    length_col = "avg_length" if "avg_length" in df.columns else "len_mean"

    for p_hat in p_hats:
        sub_df = df if p_hat is None else df[df["p_hat"] == p_hat]

        T0 = int(sub_df["T0"].iloc[0])
        T = int(sub_df["T"].iloc[0])

        fig, ax = new_figure()

        for method in sorted(sub_df["method"].unique()):
            sub = sub_df[sub_df["method"] == method].sort_values("t")
            style = METHOD_STYLES.get(method, {})
            label = style.get("label", method)
            color = style.get("color", None)

            ax.plot(
                sub["t"],
                sub[length_col],
                label=label,
                color=color,
                linewidth=1.5,
            )

        ax.axvline(x=T0 + 0.5, color="red", linestyle="--", label="Drift onset")
        ax.axvspan(T0 + 0.5, T + 0.5, color="red", alpha=0.05)

        title = "Temporal evolution of interval lengths under drift"
        if p_hat is not None:
            title += f" (p_hat={p_hat:g})"

        finalize_ax(
            ax,
            xlabel="Time period t",
            ylabel="Average interval length",
            title=title,
            nominal_level=None,
        )

        if save_dir is not None:
            suffix = "" if p_hat is None else f"_phat{_fmt_p_hat(float(p_hat))}"
            out_path = save_dir / f"{prefix}_length_vs_time{suffix}.png"
            save_figure(fig, out_path)
        else:
            import matplotlib.pyplot as plt
            plt.show()
            plt.close(fig)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    figs_dir = base_dir / "figs"
    figs_dir.mkdir(parents=True, exist_ok=True)

    df_path = data_dir / "temporal_drift_results.csv"
    df = pd.read_csv(df_path)

    prefix = "temporal_drift"

    plot_reject_rate_vs_time(df, save_dir=figs_dir, prefix=prefix)
    plot_coverage_vs_time(df, save_dir=figs_dir, prefix=prefix)
    plot_length_vs_time(df, save_dir=figs_dir, prefix=prefix)
