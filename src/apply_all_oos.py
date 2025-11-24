#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

# --- helpers binning déjà dans ton repo ---
from features.binning import load_bins_json, transform_with_learned_bins

# ============== I/O =================
def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)

def save_any(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

# ============== housekeeping =================
def drop_missing_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les colonnes de type was_missing_* ou *_missing si demandé."""
    cols = df.columns
    mask = cols.str.startswith("was_missing_") | cols.str.endswith("_missing")
    todrop = cols[mask].tolist()
    if todrop:
        return df.drop(columns=todrop, errors="ignore")
    return df

# ============== WOE =================
def resolve_bin_col(df: pd.DataFrame, raw: str, tag: str) -> str | None:
    pref = f"{tag}{raw}"
    suff = f"{raw}{tag}"
    if pref in df.columns:
        return pref
    if suff in df.columns:
        return suff
    return None

def apply_woe_with_maps(
    df_any: pd.DataFrame,
    maps: dict[str, dict],
    kept_vars_raw: list[str],
    bin_tag: str
) -> pd.DataFrame:
    """Construit les colonnes *_WOE à partir des colonnes BIN et des woe_maps."""
    cols = []
    for raw in kept_vars_raw:
        bcol = resolve_bin_col(df_any, raw, bin_tag)
        if bcol is None or raw not in maps or bcol not in df_any.columns:
            continue
        ser = df_any[bcol].astype("Int64")
        wmap = maps[raw]["map"]
        wdef = float(maps[raw]["default"])
        x = ser.map(wmap).astype(float).fillna(wdef)
        cols.append((f"{raw}_WOE", x))
    if not cols:
        return pd.DataFrame(index=df_any.index)
    return pd.concat([s for _, s in cols], axis=1)

# ============== Bucketing =================
def assign_bucket(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    inner = edges[1:-1]
    return (np.digitize(np.asarray(scores, float), inner, right=False) + 1).astype(int)

# ============== Metrics (si cible dispo) =================
def ks_best_threshold(y, p):
    from sklearn.metrics import roc_curve
    y = pd.Series(y).astype(int).to_numpy()
    p = pd.Series(p).astype(float).to_numpy()
    if np.unique(y).size < 2:
        return np.nan, np.nan
    fpr, tpr, thr = roc_curve(y, p)
    ks_arr = tpr - fpr
    i = int(np.nanargmax(ks_arr))
    return float(ks_arr[i]), float(thr[i])

def compute_metrics_if_target(df_scored: pd.DataFrame, target: str, proba_col: str) -> dict:
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    if target not in df_scored.columns:
        return {}
    y = df_scored[target].astype(int).values
    p = df_scored[proba_col].astype(float).values
    out = {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p)),
    }
    ks, thr = ks_best_threshold(y, p)
    out.update({"ks": ks, "ks_threshold": thr})
    return out

# ============== CLI =================
def parse_args():
    p = argparse.ArgumentParser(
        "Apply imputer → binning → (WOE ou BIN) → model (LR brute de préférence) → risk buckets sur OOS."
    )
    p.add_argument("--data", required=True, help="Chemin du OOS brut (parquet/csv).")
    p.add_argument("--out", required=True, help="Fichier de sortie scoré (parquet/csv).")
    # artefacts
    p.add_argument("--imputer", default="artifacts/imputer/imputer.joblib",
                   help="Imputer sklearn (obligatoire si données brutes).")
    p.add_argument("--imputer-meta", default="artifacts/imputer/imputer_meta.json",
                   help="Meta imputer (optionnel, utilisé s’il existe, pour les noms de colonnes out).")
    p.add_argument("--bins", default="artifacts/binning_maxgini/bins.json",
                   help="bins.json appris.")
    p.add_argument("--model", default="artifacts/model_from_binned/model_best.joblib",
                   help="Modèle entraîné (joblib) avec clés model / best_lr / kept_woe / woe_maps / target.")
    p.add_argument("--buckets", default="artifacts/model_from_binned/risk_buckets.json",
                   help="JSON avec 'edges' pour les classes (1..K).")
    # colonnes & options
    p.add_argument("--bin-suffix", default="__BIN", help="Suffixe/prefixe des colonnes de binning.")
    p.add_argument("--target", default="default_24m", help="Nom de la cible si présente dans OOS.")
    p.add_argument("--id-cols", default="loan_sequence_number,vintage",
                   help="Colonnes à recopier dans la sortie (séparées par virgules).")
    p.add_argument("--drop-missing-flags", action="store_true",
                   help="Supprime *_missing / was_missing_* avant traitement.")
    return p.parse_args()

def main():
    args = parse_args()
    out_path = Path(args.out)

    # 1) OOS brut
    df_raw = load_any(args.data)
    if args.drop_missing_flags:
        df_raw = drop_missing_flag_columns(df_raw)

    # 2) Imputation --> mêmes colonnes que pendant le training
    if not args.imputer or not Path(args.imputer).exists():
        raise FileNotFoundError(f"Imputer introuvable: {args.imputer}")
    imputer = load(args.imputer)

    # Détermine les colonnes de sortie attendues
    cols_out = None
    meta_path = Path(args.imputer_meta)
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            cols_out = meta.get("columns_out") or meta.get("feature_names_out_")
        except Exception:
            cols_out = None
    if cols_out is None:
        cols_out = getattr(imputer, "feature_names_out_", None)

    X_imp = imputer.transform(df_raw)
    if cols_out is not None:
        df_imp = pd.DataFrame(X_imp, columns=cols_out, index=df_raw.index)
    else:
        # fallback: garder l’ordre des colonnes brutes
        df_imp = pd.DataFrame(X_imp, columns=df_raw.columns, index=df_raw.index)

    # 3) Binning appris
    learned = load_bins_json(args.bins)
    # NB: la fonction de ton repo n'attend pas de paramètre bin_col_suffix
    df_binned = transform_with_learned_bins(df_imp, learned)

    # 4) Modèle & features attendues
    art = load(args.model)
    model = art["model"]                 # CalibratedClassifierCV
    best_lr = art.get("best_lr", None)   # LogisticRegression brute (post-ablation)
    kept_woe = art["kept_woe"]
    woe_maps = art.get("woe_maps", None)
    target_col = art.get("target", args.target)
    bin_tag = args.bin_suffix

    feature_names = list(kept_woe)

    # Détection du type de features attendu par le modèle
    any_woe_like = any(c.endswith("_WOE") for c in feature_names)
    all_bin_like = (
        bin_tag
        and all(c.endswith(bin_tag) or c.startswith(bin_tag) for c in feature_names)
    )

    # 4.a Cas 1 : modèle entraîné sur WOE (feature_names en *_WOE)
    if any_woe_like:
        if woe_maps is None:
            raise ValueError(
                "Le modèle semble attendre des colonnes WOE (*_WOE) mais 'woe_maps' est absent de l'artefact."
            )
        kept_raw = [c.replace("_WOE", "") for c in feature_names]
        X_woe = apply_woe_with_maps(df_binned, woe_maps, kept_raw, bin_tag=bin_tag)
        # Reindex pour avoir exactement les colonnes dans le bon ordre
        X = X_woe.reindex(columns=feature_names).astype(float).fillna(0.0)

    # 4.b Cas 2 : modèle entraîné directement sur BIN (feature_names en __BIN)
    elif all_bin_like:
        missing = [c for c in feature_names if c not in df_binned.columns]
        if missing:
            raise ValueError(
                f"Colonnes BIN attendues par le modèle manquantes dans le dataset binned: {missing}"
            )
        X = df_binned[feature_names].astype(float).fillna(0.0)

    # 4.c Cas 3 : cas mixte/ambiguous → fallback direct (rare)
    else:
        # On tente un alignement direct sur df_binned
        X = df_binned.reindex(columns=feature_names).astype(float).fillna(0.0)

    # 5) Probas : on privilégie la LR brute pour le scoring OOS
    if best_lr is not None:
        proba = best_lr.predict_proba(X)[:, 1]
    else:
        print("[WARN] Pas de 'best_lr' dans l'artefact, utilisation du modèle calibré.")
        proba = model.predict_proba(X)[:, 1]

    df_scored = pd.DataFrame({"proba": proba}, index=df_binned.index)

    # 6) Buckets
    if args.buckets and Path(args.buckets).exists():
        edges = np.asarray(json.loads(Path(args.buckets).read_text())["edges"], dtype=float)
        df_scored["risk_bucket"] = assign_bucket(df_scored["proba"].values, edges)

    # 7) Id-cols + cible (si disponible dans df_raw)
    if args.id_cols:
        for c in [c.strip() for c in args.id_cols.split(",") if c.strip()]:
            if c in df_raw.columns and c not in df_scored.columns:
                df_scored[c] = df_raw[c].values
    if target_col and target_col in df_raw.columns:
        df_scored[target_col] = df_raw[target_col].astype(int).values
        metrics = compute_metrics_if_target(df_scored, target_col, "proba")
    else:
        metrics = {}

    # 8) Save
    save_any(df_scored.reset_index(drop=True), out_path)

    # 9) Log
    print(f"✔ OOS scored → {out_path}")
    keep = ["proba"]
    if "risk_bucket" in df_scored.columns:
        keep.append("risk_bucket")
    if target_col in df_scored.columns:
        keep.append(target_col)
    print("  Columns:", ", ".join(keep))
    if metrics:
        print("  Metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
