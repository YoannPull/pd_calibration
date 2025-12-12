# experiments/beta_binom_jeffreys/plot_beta_binom.py
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_coverage_vs_p_all_methods_all_combinations(
    df,
    save_dir=None,
    prefix="beta_binom",
):
    """
    Pour chaque combinaison (n, rho), trace la probabilité de couverture
    en fonction de p pour toutes les méthodes.

    - df       : DataFrame (beta_binom_results.csv)
    - save_dir : Path vers le dossier où sauver les PNG (ou None pour plt.show())
    - prefix   : préfixe des noms de fichiers
    """
    conf_levels = df["conf_level"].unique()
    if len(conf_levels) != 1:
        print("Attention: plusieurs niveaux de confiance différents dans df.")
    conf = conf_levels[0]

    ns = sorted(df["n"].unique())
    rhos = sorted(df["rho"].unique())

    for n in ns:
        for rho in rhos:
            sub = df[(df["n"] == n) & (df["rho"] == rho)].copy()
            if sub.empty:
                continue

            # Ordonner par p_true
            sub = sub.sort_values("p_true")

            plt.figure(figsize=(24, 12))

            for method in sorted(sub["method"].unique()):
                tmp = sub[sub["method"] == method]
                x_p = tmp["p_true"]
                y_cov = tmp["coverage"]

                plt.plot(
                    x_p,
                    y_cov,
                    marker="o",
                    label=method,
                )

                # Info de debug / résumé dans le terminal : min coverage
                min_cov = y_cov.min()
                idx_min = y_cov.idxmin()
                p_at_min = tmp.loc[idx_min, "p_true"]
                np_at_min = tmp.loc[idx_min, "n"] * p_at_min
                print(
                    f"[n={n}, rho={rho}, method={method}] "
                    f"min coverage = {min_cov:.4f} / p = {p_at_min:.6f} / n*p = {np_at_min:.4f}"
                )

            # Ligne horizontale : niveau nominal
            plt.axhline(y=conf, color="black", linestyle="--", label="Nominal level")

            plt.xlabel(r"$p$")
            plt.ylabel("Coverage probability")
            plt.title(
                f"Coverage vs $p$ (n={n}, rho={rho})"
            )
            plt.legend()
            plt.grid(True)

            if save_dir is not None:
                rho_str = str(rho).replace(".", "_")
                fname = f"{prefix}_coverage_all_methods_n{n}_rho{rho_str}.png"
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

    plot_coverage_vs_p_all_methods_all_combinations(
        df,
        save_dir=figs_dir,
        prefix="b_b",
    )
