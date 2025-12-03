#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_vintage_grade_report.py
================================

Rapport d'évolution des GRADES par VINTAGE (ou tout autre segment) à partir d'un fichier scored.

Convention de grade (alignée sur le pipeline de training) :
    - grade 1 = classe la moins risquée
    - grade N = classe la plus risquée
    - on s'attend donc à ce que les PD soient CROISSANTES avec le grade.

Entrée : un fichier parquet/csv contenant au minimum :
    - une colonne vintage (ou segment) (ex: 'vintage')
    - une colonne de grade (ex: 'grade')
Optionnel :
    - une colonne de PD modèle (ex: 'pd')
    - une cible binaire (ex: 'default_24m')

Référence TTC (master scale) :
    - issue du fichier bucket_stats.json produit au training
      (section "train", colonne "pd" par bucket/grade).

Sortie : un fichier HTML avec :
    - métriques globales de scoring (si PD + target dispo)
    - tableaux volumes / % par vintage & grade
    - stacked bar : distribution des grades par vintage
    - (optionnel) PD moyenne par vintage & grade
    - (optionnel) DR observée par vintage
    - tableaux de calibration par (vintage, grade) vs PD TTC master scale :
        PD_TTC_master(grade) = PD par grade de la master scale (bucket_stats.json)
    - commentaire sur la monotonie par grade (pd_ttc, pd_hat, pd_obs)
"""

import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # noqa: F401 (importé pour cohérence / éventuelles extensions)

from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


# --- Style dark, cohérent avec ton autre report ---
DARK_BG = "#1E1E1E"
DARK_PANEL = "#2A2A2A"
BLUE = "#4EA8FF"
YELLOW = "#FFDD57"
GREY = "#ABB2BF"

plt.rcParams.update({
    "axes.facecolor": DARK_PANEL,
    "figure.facecolor": DARK_BG,
    "savefig.facecolor": DARK_BG,
    "text.color": GREY,
    "axes.labelcolor": GREY,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "axes.edgecolor": GREY,
    "grid.color": "#555555",
})


# =============================================================================
# Helpers
# =============================================================================

def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{encoded}"


def _sort_grades_like_numbers(cols):
    """
    Trie les grades numériquement si possible (1, 2, 3, ..., 10),
    sinon lexicographiquement. Permet de garder la logique 1 = moins risqué,
    N = plus risqué, de gauche à droite / haut en bas.
    """
    try:
        return sorted(cols, key=lambda x: float(x))
    except Exception:
        return sorted(cols)


# =============================================================================
# Charger la PD TTC de master scale (bucket_stats.json)
# =============================================================================

def load_pd_ttc_from_master_scale(bucket_stats_path: Path):
    """
    Lit bucket_stats.json (généré au training) et retourne un dict:
        {grade/bucket -> PD_TTC}
    basé sur la section "train".
    """
    if bucket_stats_path is None:
        print("[WARN] Aucun chemin bucket_stats.json fourni, pd_ttc sera vide.",
              file=sys.stderr)
        return None

    if not bucket_stats_path.exists():
        print(f"[WARN] bucket_stats.json non trouvé à {bucket_stats_path}. "
              f"pd_ttc sera vide dans le rapport.", file=sys.stderr)
        return None

    stats = json.loads(bucket_stats_path.read_text())
    train_stats = stats.get("train", [])
    if not train_stats:
        print(f"[WARN] Section 'train' manquante ou vide dans bucket_stats.json.",
              file=sys.stderr)
        return None

    df_train = pd.DataFrame(train_stats)
    # On s'attend à avoir au moins les colonnes: 'bucket' et 'pd'
    if "bucket" not in df_train.columns or "pd" not in df_train.columns:
        print("[WARN] Colonnes 'bucket'/'pd' manquantes dans bucket_stats.json.",
              file=sys.stderr)
        return None

    pd_ttc_map = df_train.set_index("bucket")["pd"].to_dict()
    return pd_ttc_map


# =============================================================================
# Calibration Error (ECE)
# =============================================================================

def expected_calibration_error(y_true, y_prob, n_bins=10):
    df = pd.DataFrame({"y": y_true, "pred": y_prob})
    df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")

    grp = df.groupby("bin").agg(
        count=("y", "size"),
        obs=("y", "mean"),
        pred=("pred", "mean"),
    ).dropna()

    total = grp["count"].sum()
    ece = float(np.sum(np.abs(grp["obs"] - grp["pred"]) * (grp["count"] / total)))
    return ece


# =============================================================================
# 1. Tableau volumes & % par vintage / grade
# =============================================================================

def build_volume_tables(df, vintage_col, grade_col):
    tab_count = (
        df.groupby([vintage_col, grade_col])
          .size()
          .rename("count")
          .reset_index()
    )

    pivot_count = tab_count.pivot(index=vintage_col, columns=grade_col, values="count").fillna(0).astype(int)
    pivot_pct = pivot_count.div(pivot_count.sum(axis=1), axis=0) * 100

    # On impose un tri des grades cohérent avec la logique 1 (moins risqué) → N (plus risqué)
    try:
        sorted_cols = _sort_grades_like_numbers(pivot_count.columns)
        pivot_count = pivot_count.reindex(columns=sorted_cols)
        pivot_pct = pivot_pct.reindex(columns=sorted_cols)
    except Exception:
        pass

    return pivot_count, pivot_pct


# =============================================================================
# 2. Stacked bar : distribution des grades par vintage
# =============================================================================

def plot_grade_distribution(pivot_pct: pd.DataFrame, vintage_col: str, grade_name: str):
    fig, ax = plt.subplots(figsize=(10, 5))

    pivot_pct = pivot_pct.sort_index()
    bottoms = np.zeros(len(pivot_pct))
    x = np.arange(len(pivot_pct.index))

    # On s'assure que les colonnes (grades) sont ordonnées numériquement
    cols = _sort_grades_like_numbers(pivot_pct.columns)

    for g in cols:
        vals = pivot_pct[g].values
        ax.bar(x, vals, bottom=bottoms, label=f"{grade_name} {g}")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(pivot_pct.index, rotation=45, ha="right")
    ax.set_ylabel("Volume (%)")
    ax.set_xlabel(vintage_col)
    ax.set_title("Distribution des grades par vintage")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend(ncol=4, fontsize=8)

    return fig_to_base64(fig)


# =============================================================================
# 3. PD moyenne par vintage & grade (si PD dispo)
# =============================================================================

def plot_pd_by_vintage_and_grade(df, vintage_col, grade_col, pd_col):
    fig, ax = plt.subplots(figsize=(10, 5))

    grp = (
        df.groupby([vintage_col, grade_col])[pd_col]
          .mean()
          .reset_index()
    )

    # Boucle sur les grades dans l'ordre 1 → N
    for g in _sort_grades_like_numbers(grp[grade_col].unique()):
        sub = grp[grp[grade_col] == g].copy()
        sub = sub.sort_values(vintage_col)
        ax.plot(sub[vintage_col], sub[pd_col], marker="o", label=f"Grade {g}")

    ax.set_xlabel(vintage_col)
    ax.set_ylabel(f"PD moyenne ({pd_col})")
    ax.set_title("PD moyenne par vintage et par grade")
    ax.grid(True, alpha=0.2)
    ax.legend(ncol=3, fontsize=8)
    plt.xticks(rotation=45, ha="right")

    return fig_to_base64(fig)


# =============================================================================
# 4. DR observée par vintage (si target dispo)
# =============================================================================

def plot_dr_by_vintage(df, vintage_col, target_col):
    fig, ax = plt.subplots(figsize=(8, 4))

    grp = (
        df.groupby(vintage_col)[target_col]
          .mean()
          .reset_index()
          .sort_values(vintage_col)
    )

    ax.plot(grp[vintage_col], grp[target_col], marker="o", color=BLUE)
    ax.set_xlabel(vintage_col)
    ax.set_ylabel(f"Default rate ({target_col})")
    ax.set_title("Taux de défaut observé par vintage")
    ax.grid(True, alpha=0.2)
    plt.xticks(rotation=45, ha="right")

    return fig_to_base64(fig)


# =============================================================================
# 5. Metrics globales de scoring
# =============================================================================

def compute_global_metrics(df, target_col, pd_col):
    y_true = df[target_col].astype(int).values
    y_prob = df[pd_col].astype(float).values

    auc = roc_auc_score(y_true, y_prob)
    gini = 2 * auc - 1
    ll = log_loss(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)

    return {
        "AUC": auc,
        "Gini": gini,
        "LogLoss": ll,
        "Brier": brier,
        "ECE": ece,
        "N": len(y_true),
        "Default Rate": float(y_true.mean()),
    }


# =============================================================================
# 6. Calibration vs PD TTC master scale
# =============================================================================

def build_calibration_table(
    df,
    vintage_col,
    grade_col,
    pd_col=None,
    target_col=None,
    pd_ttc_map=None,
):
    """
    PD_TTC_master(grade) = PD par grade issue de la master scale
    (bucket_stats.json, section 'train', colonne 'pd').

    Compare, pour chaque (vintage, grade) :
      - n (volume)
      - n_defaults (nb de défauts, si cible dispo)
      - pd_hat (PD moyenne modèle)
      - pd_obs (DR observée, si cible dispo)
      - pd_ttc (référence master scale par grade)
    """

    # Base : n par (vintage, grade)
    calib = (
        df.groupby([vintage_col, grade_col])
          .size()
          .rename("n")
          .reset_index()
    )

    # PD hat (moyenne modèle)
    if pd_col is not None and pd_col in df.columns:
        pd_hat = (
            df.groupby([vintage_col, grade_col])[pd_col]
              .mean()
              .reset_index()
              .rename(columns={pd_col: "pd_hat"})
        )
        calib = calib.merge(pd_hat, on=[vintage_col, grade_col], how="left")

    # PD observée + n_defaults (si cible dispo)
    if target_col is not None and target_col in df.columns:
        agg = (
            df.groupby([vintage_col, grade_col])[target_col]
              .agg(pd_obs="mean", n_defaults="sum")
              .reset_index()
        )
        calib = calib.merge(agg, on=[vintage_col, grade_col], how="left")

    # PD TTC master par grade (issue de pd_ttc_map)
    if pd_ttc_map is not None:
        master = pd.DataFrame({
            grade_col: list(pd_ttc_map.keys()),
            "pd_ttc": list(pd_ttc_map.values()),
        })
        calib = calib.merge(master, on=grade_col, how="left")
    else:
        calib["pd_ttc"] = np.nan

    return calib


def build_calibration_tables_by_vintage_html(calib_df: pd.DataFrame, vintage_col: str, grade_col: str) -> str:
    """
    Construit un bloc HTML avec un tableau par vintage pour lisibilité.
    Met n_defaults juste à côté de n.
    Trie les grades pour respecter la logique 1 (moins risqué) → N (plus risqué).
    """
    if calib_df.empty:
        return "<p>Aucune information de calibration disponible.</p>"

    parts = []
    for v in sorted(calib_df[vintage_col].unique()):
        sub = calib_df[calib_df[vintage_col] == v].copy()

        # Tri des lignes par grade dans l'ordre 1 → N
        try:
            sub[grade_col] = sub[grade_col].astype(float)
            sub = sub.sort_values(by=grade_col)
        except Exception:
            sub = sub.sort_values(by=grade_col)

        # On évite de répéter la colonne vintage à l'intérieur du tableau
        if vintage_col in sub.columns:
            sub = sub.drop(columns=[vintage_col])

        # Réordonnage des colonnes pour avoir n et n_defaults côte à côte
        preferred_order = [grade_col, "n", "n_defaults", "pd_hat", "pd_obs", "pd_ttc"]
        cols = [c for c in preferred_order if c in sub.columns]
        others = [c for c in sub.columns if c not in cols]
        sub = sub[cols + others]

        parts.append(f"<h4>Vintage = {v}</h4>")
        parts.append(sub.round(6).to_html(index=False))

    return "\n".join(parts)


# =============================================================================
# 7. Monotonicité par grade
# =============================================================================

def build_monotonicity_comment(calib_df: pd.DataFrame, grade_col: str) -> str:
    """
    Vérifie la monotonie par grade (agrégé sur tous les vintages) pour :
      - pd_ttc
      - pd_hat
      - pd_obs

    Convention :
      - grade 1 = moins risqué
      - grade N = plus risqué
    On teste donc la monotonie CROISSANTE (grade ↑ => PD ↑).
    """
    if calib_df.empty or grade_col not in calib_df.columns:
        return "<p>Monotonicité : non évaluée (données insuffisantes).</p>"

    # Agrégation par grade
    agg = calib_df.groupby(grade_col)[["pd_ttc", "pd_hat", "pd_obs"]].mean()

    # Tri des grades (numérique si possible, sinon lexicographique)
    try:
        idx_sorted = sorted(agg.index, key=lambda x: float(x))
        agg = agg.loc[idx_sorted]
    except Exception:
        agg = agg.sort_index()

    msgs = []
    for col, label in [
        ("pd_ttc", "PD TTC (pd_ttc)"),
        ("pd_hat", "PD modèle (pd_hat)"),
        ("pd_obs", "PD observée (pd_obs)"),
    ]:
        if col not in agg.columns or agg[col].isna().all():
            continue
        vals = agg[col].values
        if len(vals) <= 1:
            continue
        diffs = np.diff(vals)
        # On s'attend à ce que la PD augmente avec le grade
        if np.all(diffs >= -1e-8):
            msgs.append(f"{label} : monotonie respectée.")
        else:
            msgs.append(f"{label} : <strong>monotonie cassée</strong>.")

    if not msgs:
        return "<p>Monotonicité : non évaluée (pas de colonnes PD exploitables).</p>"

    return "<p><strong>Commentaire monotonicité (par grade, agrégé sur tous les vintages) :</strong><br>" + "<br>".join(msgs) + "</p>"


# =============================================================================
# HTML
# =============================================================================

def build_html(
    out_path: Path,
    vintage_col: str,
    grade_col: str,
    pivot_count: pd.DataFrame,
    pivot_pct: pd.DataFrame,
    img_dist: str,
    img_pd_by_grade: str | None,
    img_dr: str | None,
    metrics: dict | None,
    calib_tables_html: str,
    mono_comment_html: str,
    sample_name: str,
    vintage_min: str,
    vintage_max: str,
):
    # Table de métriques si dispo
    if metrics is not None:
        metrics_rows = "\n".join(
            f"<tr><td>{k}</td><td>{(v if k=='N' else round(v, 6))}</td></tr>"
            for k, v in metrics.items()
        )
        metrics_html = f"""
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {metrics_rows}
        </table>
        """
    else:
        metrics_html = "<p>Aucune métrique disponible (PD ou target manquante).</p>"

    legend_html = """
    <p><strong>Légende (tables de calibration) :</strong></p>
    <ul>
        <li><code>grade</code> : classe de risque (master scale), avec 1 = moins risqué et N = plus risqué.</li>
        <li><code>n</code> : nombre d'expositions dans le couple (vintage, grade).</li>
        <li><code>n_defaults</code> : nombre de défauts observés dans ce couple (si la cible est disponible).</li>
        <li><code>pd_hat</code> : PD moyenne du modèle sur ce couple (moyenne de la colonne pd).</li>
        <li><code>pd_obs</code> : taux de défaut observé sur ce couple (si la cible est disponible).</li>
        <li><code>pd_ttc</code> : PD TTC master scale pour ce grade (issue de bucket_stats.json, section train).</li>
    </ul>
    """

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Vintage / Grade Report – {sample_name} ({vintage_min} → {vintage_max})</title>
<style>
    body {{
        background-color: #1E1E1E;
        font-family: Arial, sans-serif;
        color: #D0D0D0;
        margin: 0;
        padding: 0;
    }}
    .container {{
        max-width: 1200px;
        margin: auto;
        padding: 30px;
        background-color: #1E1E1E;
    }}
    h1, h2, h3, h4 {{
        color: #E8E8E8;
    }}
    .section {{
        margin-top: 30px;
        padding: 20px;
        background-color: #2A2A2A;
        border-radius: 8px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        color: #E8E8E8;
        font-size: 13px;
    }}
    th {{
        background-color: #333333;
        padding: 6px;
        text-align: center;
    }}
    td {{
        background-color: #2E2E2E;
        padding: 6px;
        text-align: right;
    }}
    img {{
        width: 100%;
        border-radius: 6px;
        margin-top: 15px;
    }}
</style>
</head>
<body>
<div class="container">

    <h1>Vintage / Grade Report – {sample_name}</h1>
    <p><em>Vintages couverts :</em> <code>{vintage_min} → {vintage_max}</code></p>

    <div class="section">
        <h2>0. Global Scoring Metrics</h2>
        {metrics_html}
    </div>

    <div class="section">
        <h2>1. Volumes par {vintage_col} et {grade_col}</h2>
        <h3>Tableau des volumes (n)</h3>
        {pivot_count.to_html(classes="", border=0)}

        <h3>Tableau des proportions (%)</h3>
        {pivot_pct.round(2).to_html(classes="", border=0)}
    </div>

    <div class="section">
        <h2>2. Distribution des grades par vintage</h2>
        <img src="{img_dist}">
    </div>
"""

    if img_pd_by_grade is not None:
        html += f"""
    <div class="section">
        <h2>3. PD moyenne par vintage et par grade</h2>
        <img src="{img_pd_by_grade}">
    </div>
"""

    if img_dr is not None:
        html += f"""
    <div class="section">
        <h2>4. Taux de défaut observé par vintage</h2>
        <img src="{img_dr}">
    </div>
"""

    html += f"""
    <div class="section">
        <h2>5. Calibration par vintage / grade vs PD TTC master scale</h2>
        <p>PD_TTC_master(grade) = PD par grade issue de la master scale (bucket_stats.json).</p>
        {legend_html}
        {mono_comment_html}
        {calib_tables_html}
    </div>

</div>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Vintage / Grade evolution + calibration report from a scored file."
    )
    p.add_argument("--data", required=True, help="Fichier scored (parquet/csv).")
    p.add_argument("--out", required=True, help="Fichier HTML de sortie.")
    p.add_argument("--vintage-col", default="vintage")
    p.add_argument("--grade-col", default="grade")
    p.add_argument("--pd-col", default="pd", help="Nom de la colonne PD modèle (optionnel).")
    p.add_argument("--target", default=None, help="Nom de la colonne cible (optionnel).")
    p.add_argument("--sample-name", default="OOS", help="Nom du sample (OOS, Train, Val, etc.).")
    p.add_argument(
        "--bucket-stats",
        default=None,
        help="Chemin vers bucket_stats.json (master scale TTC)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    df = load_any(args.data)

    if args.vintage_col not in df.columns:
        raise ValueError(f"Colonne vintage '{args.vintage_col}' absente du fichier.")
    if args.grade_col not in df.columns:
        raise ValueError(f"Colonne grade '{args.grade_col}' absente du fichier.")

    # On force un type string sur vintage pour l'affichage
    df[args.vintage_col] = df[args.vintage_col].astype(str)

    # Min / max du segment (vintage)
    vintages_sorted = sorted(df[args.vintage_col].unique())
    vintage_min = vintages_sorted[0]
    vintage_max = vintages_sorted[-1]

    # 1. Tables volumes / %
    pivot_count, pivot_pct = build_volume_tables(df, args.vintage_col, args.grade_col)

    # 2. Stacked bar distribution
    img_dist = plot_grade_distribution(pivot_pct, args.vintage_col, args.grade_col)

    # 3. PD moyenne par vintage & grade (si pd_col dispo)
    img_pd_by_grade = None
    if args.pd_col is not None and args.pd_col in df.columns:
        img_pd_by_grade = plot_pd_by_vintage_and_grade(df, args.vintage_col, args.grade_col, args.pd_col)

    # 4. DR observée par vintage (si target dispo)
    img_dr = None
    if args.target is not None and args.target in df.columns:
        img_dr = plot_dr_by_vintage(df, args.vintage_col, args.target)

    # 5. Metrics globales (si PD + target dispo)
    metrics = None
    if (args.target is not None and args.target in df.columns
            and args.pd_col is not None and args.pd_col in df.columns):
        metrics = compute_global_metrics(df, args.target, args.pd_col)

    # 6. Charger la master scale TTC
    pd_ttc_map = None
    if args.bucket_stats is not None:
        pd_ttc_map = load_pd_ttc_from_master_scale(Path(args.bucket_stats))
    else:
        print("[WARN] --bucket-stats non fourni, pd_ttc sera vide dans les tables.",
              file=sys.stderr)

    # 7. Table de calibration vs PD TTC master
    calib_df = build_calibration_table(
        df,
        vintage_col=args.vintage_col,
        grade_col=args.grade_col,
        pd_col=args.pd_col if args.pd_col in df.columns else None,
        target_col=args.target if (args.target and args.target in df.columns) else None,
        pd_ttc_map=pd_ttc_map,
    )
    calib_tables_html = build_calibration_tables_by_vintage_html(calib_df, args.vintage_col, args.grade_col)

    # Commentaire monotonicité (global par grade)
    mono_comment_html = build_monotonicity_comment(calib_df, args.grade_col)

    # 8. HTML final
    out_path = Path(args.out)
    build_html(
        out_path=out_path,
        vintage_col=args.vintage_col,
        grade_col=args.grade_col,
        pivot_count=pivot_count,
        pivot_pct=pivot_pct,
        img_dist=img_dist,
        img_pd_by_grade=img_pd_by_grade,
        img_dr=img_dr,
        metrics=metrics,
        calib_tables_html=calib_tables_html,
        mono_comment_html=mono_comment_html,
        sample_name=args.sample_name,
        vintage_min=vintage_min,
        vintage_max=vintage_max,
    )

    print(f"✔ Vintage / Grade + calibration report généré : {out_path}")


if __name__ == "__main__":
    main()
