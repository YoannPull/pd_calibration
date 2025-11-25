#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
src/generate_report.py
----------------------
Générateur de Rapport de Validation de Modèle (HTML).
Mise à jour : Inclut désormais l'affichage des Coefficients du modèle.
"""

import argparse
import base64
import io
import sys
import time
from pathlib import Path

import matplotlib
# Force le backend 'Agg' pour serveurs sans écran
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib  # Nécessaire pour charger le modèle

from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, roc_auc_score, log_loss, brier_score_loss

# Style des graphiques
plt.style.use('ggplot')
sns.set_palette("husl")

def parse_args():
    p = argparse.ArgumentParser(description="Generate Credit Risk Validation Report (HTML)")
    p.add_argument("--data", required=True, help="Path to scored data (parquet/csv)")
    p.add_argument("--out", required=True, help="Path to output HTML report")
    p.add_argument("--target", default="target", help="Column name for target")
    p.add_argument("--score", default="score", help="Column name for score points")
    p.add_argument("--pd", default="pd", help="Column name for probability of default")
    p.add_argument("--grade", default="grade", help="Column name for risk bucket/grade")
    # Nouvel argument pour le modèle
    p.add_argument("--model", default=None, help="Path to model_best.joblib (Optionnel, pour afficher les coefs)")
    return p.parse_args()

def fig_to_base64(fig):
    """Convertit une figure Matplotlib en string base64 pour l'HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    data = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{data}"

def plot_roc(y_true, y_prob):
    """Génère la courbe ROC."""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f'Model (AUC = {auc:.4f})', lw=2)
        ax.plot([0, 1], [0, 1], 'k--', lw=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc="lower right")
        return fig_to_base64(fig), auc
    except Exception as e:
        print(f"[WARN] Impossible de tracer la ROC: {e}")
        return "", 0.0

def plot_calibration(y_true, y_prob):
    """Génère la courbe de calibration."""
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='quantile')
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve')
        ax.legend()
        return fig_to_base64(fig)
    except Exception as e:
        print(f"[WARN] Impossible de tracer la Calibration: {e}")
        return ""

def plot_score_dist(df, score_col, target_col):
    """Distribution des scores par classe."""
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data=df, x=score_col, hue=target_col, kde=True, bins=30, ax=ax, common_norm=False, stat="density")
        ax.set_title('Score Distribution by Target')
        return fig_to_base64(fig)
    except Exception as e:
        print(f"[WARN] Impossible de tracer la Distribution des Scores: {e}")
        return ""

def plot_master_scale(df, grade_col, target_col, pd_col):
    """Analyse par Grade (Bucket)."""
    try:
        grp = df.groupby(grade_col).agg(
            count=(target_col, 'size'),
            bad=(target_col, 'sum'),
            mean_pred_pd=(pd_col, 'mean')
        ).reset_index()
        
        grp['obs_pd'] = grp['bad'] / grp['count']
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(grp[grade_col].astype(str), grp['count'], alpha=0.3, color='gray', label='Volume')
        ax1.set_ylabel('Volume (Count)')
        ax1.legend(loc='upper left')
        ax1.grid(False) 
        
        ax2 = ax1.twinx()
        ax2.plot(grp[grade_col].astype(str), grp['obs_pd'], 'r-o', lw=2, label='Observed PD')
        ax2.plot(grp[grade_col].astype(str), grp['mean_pred_pd'], 'b--x', lw=2, label='Predicted PD')
        ax2.set_ylabel('Probability of Default')
        ax2.legend(loc='upper right')
        
        ax1.set_title('Master Scale Analysis')
        return fig_to_base64(fig), grp
    except Exception as e:
        print(f"[WARN] Impossible de tracer la Master Scale: {e}")
        return "", pd.DataFrame()

def plot_coefficients(df_coef):
    """Trace l'importance des variables (Coefficients)."""
    try:
        # Tri par valeur absolue pour l'importance visuelle
        df_sorted = df_coef.reindex(df_coef['Coefficient'].abs().sort_values(ascending=False).index)
        
        fig, ax = plt.subplots(figsize=(8, len(df_coef) * 0.4 + 2))
        sns.barplot(data=df_sorted, y='Feature', x='Coefficient', ax=ax, palette="vlag")
        ax.set_title("Logistic Regression Coefficients (Log-Odds impact)")
        ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
        return fig_to_base64(fig)
    except Exception as e:
        print(f"[WARN] Impossible de tracer les coefficients: {e}")
        return ""

def main():
    args = parse_args()
    
    print(f"Loading data from {args.data}...")
    try:
        if str(args.data).endswith(".parquet"):
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data)
    except Exception as e:
        print(f"[ERR] Failed to load data: {e}")
        sys.exit(1)

    # 1. Metrics & Plots
    # ------------------
    df = df.dropna(subset=[args.target]).copy()
    y_true = df[args.target].astype(int)
    y_pd = df[args.pd]
    
    print("Generating performance plots...")
    auc_score = roc_auc_score(y_true, y_pd)
    gini_score = 2 * auc_score - 1
    brier = brier_score_loss(y_true, y_pd)
    ll = log_loss(y_true, y_pd)
    
    roc_img, _ = plot_roc(y_true, y_pd)
    cal_img = plot_calibration(y_true, y_pd)
    dist_img = plot_score_dist(df, args.score, args.target)
    ms_img, ms_df = plot_master_scale(df, args.grade, args.target, args.pd)

    # 2. Model Coefficients (Si demandé)
    # ----------------------------------
    coef_html = "<p>No model file provided.</p>"
    coef_img = ""
    intercept_val = 0.0
    
    if args.model:
        print(f"Loading model from {args.model}...")
        try:
            model_pkg = joblib.load(args.model)
            best_lr = model_pkg["best_lr"]
            kept_features = model_pkg["kept_features"]
            
            # Extraction
            coefs = best_lr.coef_[0]
            intercept_val = best_lr.intercept_[0]
            
            df_coef = pd.DataFrame({
                "Feature": kept_features,
                "Coefficient": coefs
            }).sort_values(by="Coefficient", ascending=False)
            
            # Affichage Terminal
            print("\n--- MODEL COEFFICIENTS ---")
            print(f"Intercept : {intercept_val:.4f}")
            print(df_coef.to_string(index=False))
            print("--------------------------\n")
            
            # Génération Plot & HTML Table
            coef_img = plot_coefficients(df_coef)
            coef_html = df_coef.to_html(classes='table', float_format=lambda x: "{:.4f}".format(x), index=False)
            
        except Exception as e:
            print(f"[WARN] Impossible de charger le modèle pour les coefs: {e}")
            coef_html = f"<p>Error loading model: {e}</p>"

    # 3. HTML Generation
    # ------------------
    # Correction lambda pour float_format
    ms_table_html = ms_df.to_html(classes='table', float_format=lambda x: "{:.4f}".format(x), index=False) if not ms_df.empty else "<p>No Grade Data</p>"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Risk Model Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }}
            .container {{ max-width: 1100px; margin: auto; background: white; padding: 30px; box-shadow: 0 0 15px rgba(0,0,0,0.1); border-radius: 8px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; border-left: 5px solid #3498db; padding-left: 10px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }}
            .metric-box {{ background: #ecf0f1; padding: 20px; text-align: center; border-radius: 8px; }}
            .metric-title {{ font-size: 0.9em; text-transform: uppercase; color: #7f8c8d; }}
            .metric-val {{ font-size: 1.8em; font-weight: bold; color: #2980b9; margin-top: 5px; }}
            .chart-row {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin-bottom: 30px; }}
            .chart-box {{ flex: 1; min-width: 450px; text-align: center; border: 1px solid #eee; padding: 10px; border-radius: 5px; }}
            img {{ max-width: 100%; height: auto; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.9em; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
            th {{ background-color: #2c3e50; color: white; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Validation Report: {Path(args.data).name}</h1>
            <p><strong>Generated on:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>1. Global Performance Metrics</h2>
            <div class="metrics-grid">
                <div class="metric-box"><div class="metric-title">AUC</div><div class="metric-val">{auc_score:.4f}</div></div>
                <div class="metric-box"><div class="metric-title">Gini</div><div class="metric-val">{gini_score:.4f}</div></div>
                <div class="metric-box"><div class="metric-title">LogLoss</div><div class="metric-val">{ll:.4f}</div></div>
                <div class="metric-box"><div class="metric-title">Brier Score</div><div class="metric-val">{brier:.4f}</div></div>
            </div>

            <h2>2. Discrimination & Calibration</h2>
            <div class="chart-row">
                <div class="chart-box">
                    <h3>ROC Curve</h3>
                    <img src="{roc_img}" />
                </div>
                <div class="chart-box">
                    <h3>Calibration Curve</h3>
                    <img src="{cal_img}" />
                </div>
            </div>

            <h2>3. Score Distribution</h2>
            <div class="chart-row">
                 <div class="chart-box" style="flex: 2;">
                    <img src="{dist_img}" />
                </div>
            </div>

            <h2>4. Master Scale Analysis (Grades)</h2>
            <div class="chart-row">
                 <div class="chart-box" style="flex: 2;">
                    <img src="{ms_img}" />
                </div>
            </div>
            
            <h3>Detailed Grade Statistics</h3>
            {ms_table_html}

            <h2>5. Model Specification</h2>
            <p><strong>Intercept:</strong> {intercept_val:.4f}</p>
            <div class="chart-row">
                 <div class="chart-box" style="flex: 2;">
                    <img src="{coef_img}" />
                </div>
            </div>
            <h3>Coefficients Table</h3>
            {coef_html}

        </div>
    </body>
    </html>
    """

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✔ Report generated: {out_path}")

if __name__ == "__main__":
    main()