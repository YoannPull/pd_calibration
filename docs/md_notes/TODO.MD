





voici les codes dans src/ :


apply_all_oos.py  

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apply_all_oos.py

Imputer → binning → (BIN→WOE si woe_maps dispo) → modèle → PD + buckets sur SCORE sur OOS brut.
Logique alignée sur le notebook de diagnostic et la master scale définie sur le score LR.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from joblib import load

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


# ============== helpers BIN / WOE =====
def raw_name_from_bin(col: str, tag: str) -> str:
    """Inverse de l’ajout de suffixe/prefixe BIN."""
    if col.startswith(tag):
        return col[len(tag):]
    if col.endswith(tag):
        return col[:-len(tag)]
    return col


def resolve_bin_col(df: pd.DataFrame, raw: str, tag: str) -> Optional[str]:
    """Retrouve la colonne BIN correspondant à une variable 'raw'."""
    pref = f"{tag}{raw}"
    suff = f"{raw}{tag}"
    if pref in df.columns:
        return pref
    if suff in df.columns:
        return suff
    return None


def apply_woe_with_maps(
    df_any: pd.DataFrame,
    maps: Dict[str, Dict[str, Any]],
    kept_vars_raw: List[str],
    bin_tag: str,
) -> pd.DataFrame:
    """
    Construit les colonnes *_WOE à partir des colonnes BIN et des woe_maps.
    Même convention que dans le training.
    """
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
    """Buckets 1..K en fonction du SCORE (décision LR)."""
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
        "n": int(len(y)),
        "base_rate": float(y.mean()) if len(y) > 0 else np.nan,
        "proba_mean": float(p.mean()) if len(p) > 0 else np.nan,
        "proba_std": float(p.std()) if len(p) > 0 else np.nan,
    }

    if np.unique(y).size < 2:
        return out

    p_clip = np.clip(p, 1e-15, 1 - 1e-15)
    out.update(
        {
            "auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, p_clip)),
        }
    )
    ks, thr = ks_best_threshold(y, p)
    out.update({"ks": ks, "ks_threshold": thr})
    return out


# ============== CLI =================
def parse_args():
    p = argparse.ArgumentParser(
        "Apply imputer → binning → (BIN→WOE si woe_maps dispo) → modèle → PD + risk buckets (sur score) sur OOS."
    )
    p.add_argument("--data", required=True, help="Chemin du OOS brut (parquet/csv).")
    p.add_argument("--out", required=True, help="Fichier de sortie scoré (parquet/csv).")
    # artefacts
    p.add_argument(
        "--imputer",
        default="artifacts/imputer/imputer.joblib",
        help="Imputer sklearn (obligatoire si données brutes).",
    )
    p.add_argument(
        "--imputer-meta",
        default="artifacts/imputer/imputer_meta.json",
        help="Meta imputer (optionnel, utilisé s’il existe, pour les noms de colonnes out).",
    )
    p.add_argument("--bins", default="artifacts/binning_maxgini/bins.json", help="bins.json appris.")
    p.add_argument(
        "--model",
        default="artifacts/model_from_binned/model_best.joblib",
        help="Modèle entraîné (joblib) avec clés model / best_lr / kept_woe / woe_maps / target.",
    )
    p.add_argument(
        "--buckets",
        default="artifacts/model_from_binned/risk_buckets.json",
        help="JSON avec 'edges' (sur SCORE) pour les classes (1..K).",
    )
    # colonnes & options
    p.add_argument("--bin-suffix", default="__BIN", help="Suffixe/prefixe des colonnes de binning.")
    p.add_argument("--target", default="default_24m", help="Nom de la cible si présente dans OOS.")
    p.add_argument(
        "--id-cols",
        default="loan_sequence_number,vintage",
        help="Colonnes à recopier dans la sortie (séparées par virgules).",
    )
    p.add_argument(
        "--drop-missing-flags",
        action="store_true",
        help="Supprime *_missing / was_missing_* avant traitement.",
    )
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
        df_imp = pd.DataFrame(X_imp, columns=df_raw.columns, index=df_raw.index)

    # 3) Binning appris (mêmes bins que train/val)
    learned = load_bins_json(args.bins)
    df_binned = transform_with_learned_bins(df_imp, learned)

    # 4) Modèle & artefacts
    art = load(args.model)
    model = art["model"]          # CalibratedClassifierCV ou LR
    best_lr = art.get("best_lr")  # LR brute (pour le score/grille)
    kept_woe = art["kept_woe"]    # features utilisées à l'entraînement (ordre à respecter)
    woe_maps = art.get("woe_maps")
    target_col = art.get("target", args.target)
    bin_tag = args.bin_suffix

    # 5) Construction de la matrice X (logique notebook)
    if woe_maps is not None:
        # (a) récupérer toutes les colonnes __BIN dispo dans df_binned
        bin_cols = [
            c for c in df_binned.columns
            if c.endswith(bin_tag) or c.startswith(bin_tag)
        ]
        # (b) raw_name_from_bin pour toutes ces colonnes
        raw_to_bin = {raw_name_from_bin(c, bin_tag): c for c in bin_cols}
        keep_raw = sorted(raw_to_bin.keys())

        # (c) construire toutes les colonnes WOE possibles pour ces raw
        X_woe_full = apply_woe_with_maps(df_binned, woe_maps, keep_raw, bin_tag=bin_tag)

        if X_woe_full.empty:
            raise SystemExit(
                "Échec de la reconstruction des WOE sur l'OOS : X_woe_full est vide.\n"
                "Vérifier bins.json / woe_maps / colonnes BIN."
            )

        # (d) aligner exactement sur kept_woe (comme dans le notebook)
        X = X_woe_full.reindex(columns=kept_woe).astype(float).fillna(0.0)
    else:
        # Fallback : pas de WOE → on aligne directement sur les colonnes de df_binned
        print("[WARN] woe_maps absent de l'artefact, utilisation directe des colonnes binned.")
        X = df_binned.reindex(columns=kept_woe).astype(float).fillna(0.0)

    # 6) PD : utiliser le modèle calibré si dispo, sinon best_lr
    if model is not None and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif best_lr is not None and hasattr(best_lr, "predict_proba"):
        proba = best_lr.predict_proba(X)[:, 1]
    else:
        est = best_lr or model
        if est is None:
            raise SystemExit("Artefact modèle invalide : ni 'best_lr' ni 'model' n'est défini.")
        if hasattr(est, "decision_function"):
            z = est.decision_function(X)
        else:
            raise SystemExit("Impossible de calculer des probas: ni predict_proba ni decision_function disponibles.")
        proba = 1.0 / (1.0 + np.exp(-z))

    # 7) Score structurel (pour le bucketing) : best_lr.decision_function en priorité
    if best_lr is not None and hasattr(best_lr, "decision_function"):
        score = best_lr.decision_function(X)
    elif model is not None and hasattr(model, "decision_function"):
        score = model.decision_function(X)
    else:
        # fallback : logit(proba)
        score = np.log(
            np.clip(proba, 1e-15, 1 - 1e-15) / np.clip(1.0 - proba, 1e-15, 1.0)
        )

    df_scored = pd.DataFrame({"proba": proba}, index=df_binned.index)

    # 8) Buckets (sur SCORE)
    if args.buckets and Path(args.buckets).exists():
        edges = np.asarray(json.loads(Path(args.buckets).read_text())["edges"], dtype=float)
        df_scored["risk_bucket"] = assign_bucket(score, edges)

    # 9) Id-cols + cible (si disponible dans df_raw)
    if args.id_cols:
        for c in [c.strip() for c in args.id_cols.split(",") if c.strip()]:
            if c in df_raw.columns and c not in df_scored.columns:
                df_scored[c] = df_raw[c].values
    if target_col and target_col in df_raw.columns:
        df_scored[target_col] = df_raw[target_col].astype(int).values
        metrics = compute_metrics_if_target(df_scored, target_col, "proba")
    else:
        metrics = {}

    # 10) Save
    save_any(df_scored.reset_index(drop=True), out_path)

    # 11) Log
    print(f"✔ OOS scored → {out_path}")
    keep_cols = ["proba"]
    if "risk_bucket" in df_scored.columns:
        keep_cols.append("risk_bucket")
    if target_col in df_scored.columns:
        keep_cols.append(target_col)
    print("  Columns:", ", ".join(keep_cols))
    if metrics:
        print("  Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()


apply_imputer.py :

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply a saved DataImputer to one or many datasets.

Examples
--------
# 1) Fichier unique (in → out)
python src/apply_imputer.py \
  --imputer artifacts/imputer/imputer.joblib \
  --data data/processed/labels/window=24m/quarter=2023Q1/data.parquet \
  --out  data/processed/imputed/2023Q1.parquet

# 2) Batch via glob (répertoire out + suffix)
python src/apply_imputer.py \
  --imputer artifacts/imputer/imputer.joblib \
  --glob "data/processed/labels/window=24m/quarter=*/data.parquet" \
  --out-dir data/processed/imputed/quarters \
  --suffix "" \
  --meta artifacts/imputer/imputer_meta.json \
  --drop-extras

Notes
-----
- Optionnellement, on peut ré-attacher une cible présente via --target.
- --meta et --drop-extras permettent d’aligner strictement les colonnes sur
  celles vues après imputation du train (ordre + filtrage).
- --drop-missing-flags supprime les colonnes indicatrices de NA si tu n’en veux pas.
"""

import argparse
from pathlib import Path
from typing import List, Optional
import json
import re

import pandas as pd
from joblib import load


def load_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_any(df: pd.DataFrame, path: Path, force_format: Optional[str] = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = (force_format or path.suffix.lower().lstrip(".") or "").lower()
    if not fmt:
        fmt = "parquet"
        path = path.with_suffix(".parquet")
    if fmt in ("parquet", "pq"):
        df.to_parquet(path, index=False)
    elif fmt == "csv":
        df.to_csv(path, index=False)
    else:
        raise SystemExit(f"Unsupported save format: {fmt} (path={path})")


def derive_out_path(inp: Path, out_dir: Path, suffix: str, force_format: Optional[str]) -> Path:
    stem = inp.stem
    ext = ("." + force_format) if force_format else (inp.suffix or ".parquet")
    return out_dir / f"{stem}{suffix}{ext}"


def parse_args():
    p = argparse.ArgumentParser(description="Apply a saved DataImputer to dataset(s)")
    p.add_argument("--imputer", required=True, help="Path to imputer.joblib")
    # Inputs
    p.add_argument("--data", action="append", default=[], help="Input file (repeatable)")
    p.add_argument("--glob", default=None, help="Glob pattern for batch (e.g. 'dir/*.parquet')")
    # Outputs
    p.add_argument("--out", default=None, help="Single output path (only if single input)")
    p.add_argument("--out-dir", default=None, help="Output dir for batch mode")
    p.add_argument("--suffix", default="_imputed", help="Suffix added before extension in batch")
    p.add_argument("--format", choices=["parquet", "csv"], default=None, help="Force output format")
    # Options
    p.add_argument("--target", default=None, help="Optional target column to preserve/reattach")
    p.add_argument("--meta", default=None, help="Path to imputer_meta.json to align features")
    p.add_argument("--drop-extras", action="store_true", help="Drop columns not in meta features (if --meta)")
    p.add_argument("--drop-missing-flags", action="store_true",
                   help="Drop columns like 'was_missing_*' or '*_missing'")
    return p.parse_args()


def align_to_meta(df: pd.DataFrame, meta_path: Path, drop_extras: bool) -> pd.DataFrame:
    meta = json.loads(meta_path.read_text())
    feats = meta.get("features", None)
    if not feats:
        return df
    cols = list(df.columns)
    if drop_extras:
        cols = [c for c in cols if c in feats]
    remaining = [c for c in cols if c not in feats]
    ordered = [c for c in feats if c in cols] + remaining
    return df[ordered]


def drop_missing_flags(df: pd.DataFrame) -> pd.DataFrame:
    rx = re.compile(r"(?:^was_missing_|_missing$)", re.I)
    to_drop = [c for c in df.columns if rx.search(c)]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def apply_one(imputer, inp: Path, out: Path, fmt: Optional[str],
              target: Optional[str], meta_path: Optional[Path],
              drop_extras: bool, drop_flags: bool):
    df = load_any(inp)

    y = None
    if target and target in df.columns:
        y = df[target].copy()
        df = df.drop(columns=[target])

    df_imp = imputer.transform(df)

    if y is not None:
        df_imp[target] = y.values

    if meta_path and meta_path.exists():
        df_imp = align_to_meta(df_imp, meta_path, drop_extras=drop_extras)

    if drop_flags:
        df_imp = drop_missing_flags(df_imp)

    save_any(df_imp, out, fmt)
    print(f"✔ Imputation applied: {inp} → {out}  shape={df_imp.shape}")


def main():
    args = parse_args()
    imputer = load(args.imputer)

    # Collect inputs
    inputs: List[Path] = []
    for d in args.data:
        inputs.append(Path(d))
    if args.glob:
        for p in sorted(Path().glob(args.glob)):
            if p.is_file():
                inputs.append(p)
    inputs = sorted(set(inputs))
    if not inputs:
        raise SystemExit("No input. Use --data and/or --glob.")

    meta_path = Path(args.meta) if args.meta else None

    if len(inputs) == 1:
        inp = inputs[0]
        if args.out is None and args.out_dir is None:
            out = derive_out_path(inp, inp.parent, args.suffix, args.format)
        elif args.out is not None:
            out = Path(args.out)
        else:
            out = derive_out_path(inp, Path(args.out_dir), args.suffix, args.format)

        apply_one(imputer, inp, out, args.format, args.target, meta_path,
                  drop_extras=args.drop_extras, drop_flags=args.drop_missing_flags)
        return

    if args.out_dir is None:
        raise SystemExit("Multiple inputs detected: please provide --out-dir.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for inp in inputs:
        out = derive_out_path(inp, out_dir, args.suffix, args.format)
        apply_one(imputer, inp, out, args.format, args.target, meta_path,
                  drop_extras=args.drop_extras, drop_flags=args.drop_missing_flags)

    print(f"✔ Done. Wrote {len(inputs)} file(s) into {out_dir.resolve()}")


if __name__ == "__main__":
    main()



apply_binning.py :

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
apply_binning.py
Applique des bins appris (bins.json) à un nouveau dataset imputé.

- Supporte les variables catégorielles (mapping modalities -> bin_id)
  et numériques (pd.cut avec edges_for_cut).
- Suffixe de colonnes BIN paramétrable (par défaut "__BIN").
- Robustesse :
    * colonnes absentes -> warning et skip
    * modalités inconnues -> code -2
    * NaN -> code -1
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd


# ---------------- I/O helpers ----------------
def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_any(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


def load_bins_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------- Core apply ----------------
def to_float_series(s: pd.Series) -> pd.Series:
    # Convertit dates/periods en jours, sinon numérique (avec coerce)
    # Remplace l'appel déprécié à is_period_dtype par isinstance(dtype, pd.PeriodDtype)
    if isinstance(s.dtype, pd.PeriodDtype):
        ts = s.dt.to_timestamp(how="start")
        days = (ts.astype("int64") // 86_400_000_000_000)
        return days.astype("float64")
    if pd.api.types.is_datetime64_any_dtype(s):
        s_dt = s
        try:
            if getattr(s_dt.dt, "tz", None) is not None:
                s_dt = s_dt.dt.tz_convert(None)
        except Exception:
            pass
        s_dt = s_dt.astype("datetime64[ns]")
        days = (s_dt.astype("int64") // 86_400_000_000_000)
        return days.astype("float64")
    return pd.to_numeric(s, errors="coerce").astype("float64")


def apply_binnings_to_df(
    df: pd.DataFrame,
    bins_spec: Dict[str, Any],
    bin_col_suffix: str = "__BIN",
    include_missing: Optional[bool] = None,
    missing_label: Optional[str] = None,
) -> pd.DataFrame:
    """
    Ajoute des colonnes <var><bin_col_suffix> dans une copie de df.
    - Catégorielles: mapping modalities -> bin_id (inconnu -> -2, NaN -> -1 si include_missing)
    - Numériques: pd.cut(edges_for_cut) (NaN -> -1 si include_missing)
    """
    out = df.copy()

    # Récupère meta si existants dans le JSON
    inc_miss = include_missing
    miss_lab = missing_label
    if inc_miss is None:
        inc_miss = bool(bins_spec.get("include_missing", True))
    if miss_lab is None:
        miss_lab = bins_spec.get("missing_label", "__MISSING__")

    cat_res = bins_spec.get("cat_results", {}) or {}
    num_res = bins_spec.get("num_results", {}) or {}

    # Catégorielles
    for var, info in cat_res.items():
        if var not in out.columns:
            print(f"[WARN] categorical '{var}' not in data → skipped.")
            continue
        ser = out[var].astype("object")
        if inc_miss:
            ser = ser.where(ser.notna(), miss_lab)
        mapping = info.get("mapping", {}) or {}
        mapped = ser.map(mapping)
        mapped = mapped.astype("Int64").fillna(-2).astype("Int64")
        if inc_miss:
            is_nan = df[var].isna()
            mapped = mapped.where(~is_nan, -1).astype("Int64")
        out[var + bin_col_suffix] = mapped

    # Numériques
    for var, info in num_res.items():
        if var not in out.columns:
            print(f"[WARN] numeric '{var}' not in data → skipped.")
            continue
        s = to_float_series(out[var])
        e = np.asarray(info.get("edges_for_cut", info.get("edges", [])), dtype="float64")
        if e.size < 2:
            b = pd.Series(0, index=out.index, dtype="Int64")
        else:
            b = pd.cut(s, bins=e, include_lowest=True, duplicates="drop").cat.codes.astype("Int64")
        if inc_miss and s.isna().any():
            b = b.where(~s.isna(), -1).astype("Int64")
        out[var + bin_col_suffix] = b

    return out


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Apply learned bins (bins.json) to an imputed dataset.")
    p.add_argument("--data", required=True, help="CSV/Parquet imputé")
    p.add_argument("--bins", required=True, help="bins.json produit par fit_binning.py")
    p.add_argument("--out", required=True, help="sortie CSV/Parquet avec colonnes __BIN ajoutées")
    # Nom d’argument attendu par le Makefile :
    p.add_argument("--bin-col-suffix", default="__BIN", dest="bin_col_suffix",
                   help="Suffixe des colonnes BIN (par défaut: __BIN)")
    # Alias pour compatibilité éventuelle :
    p.add_argument("--bin-suffix", dest="bin_col_suffix_alias", default=None,
                   help="Alias de --bin-col-suffix")
    return p.parse_args()


def main():
    args = parse_args()

    # Résout l'éventuel alias
    bin_suffix = args.bin_col_suffix_alias if args.bin_col_suffix_alias is not None else args.bin_col_suffix
    if not isinstance(bin_suffix, str) or not bin_suffix:
        raise SystemExit("Invalid bin suffix (empty).")

    df = load_any(args.data)
    bins_spec = load_bins_json(args.bins)

    df_out = apply_binnings_to_df(
        df,
        bins_spec,
        bin_col_suffix=bin_suffix,
        include_missing=bins_spec.get("include_missing", True),
        missing_label=bins_spec.get("missing_label", "__MISSING__"),
    )

    save_any(df_out, args.out)
    print(f"✔ Binning applied: {args.data} → {args.out}  shape={df_out.shape}")
    added = [c for c in df_out.columns if c.endswith(bin_suffix)]
    print(f"  Added BIN columns: {len(added)} (suffix='{bin_suffix}')")


if __name__ == "__main__":
    main()


apply_model :

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Apply imputer → binning → (WOE ou BIN selon training) → modèle → PD + risk buckets sur OOS.

Hypothèses sur l'artefact model_best.joblib (dict):
  - 'model'      : CalibratedClassifierCV (LR calibrée) ou LogisticRegression
  - 'best_lr'    : LogisticRegression brute (post-ablation), entraînée sur les features finales
  - 'kept_woe'   : liste des features retenues côté training
       * soit des noms WOE explicites:  <raw>_WOE
       * soit des noms de colonnes BIN: <raw>__BIN (mais contenant des WOE au moment du fit)
  - 'computed_woe' : bool, True si le training a utilisé des WOE internes à partir des BIN
  - 'woe_maps'  : dict raw_name -> {'map': {bin_id: woe}, 'default': woe_default}
  - 'target'    : nom de la colonne cible (ex: 'default_24m')

IMPORTANT :
  - Les bornes de risk_buckets.json sont supposées définies sur l'échelle DU SCORE
    (best_lr.decision_function), pas sur les probas.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from joblib import load

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


# ============== WOE utils =================
def resolve_bin_col(df: pd.DataFrame, raw: str, tag: str) -> Optional[str]:
    """Trouve la colonne BIN correspondant à une variable brute `raw` et un tag (__BIN ou préfixe)."""
    pref = f"{tag}{raw}"
    suff = f"{raw}{tag}"
    if pref in df.columns:
        return pref
    if suff in df.columns:
        return suff
    return None


def strip_bin_tag(name: str, bin_tag: str) -> str:
    """Enlève le suffixe/prefixe de binning pour retrouver le nom 'raw'."""
    if bin_tag and name.endswith(bin_tag):
        return name[: -len(bin_tag)]
    if bin_tag and name.startswith(bin_tag):
        return name[len(bin_tag) :]
    return name


# ============== Bucketing =================
def assign_bucket(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Buckets 1..K en fonction du SCORE (décision LR)."""
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


def compute_metrics_if_target(df_scored: pd.DataFrame, target: str, proba_col: str) -> Dict[str, Any]:
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

    if target not in df_scored.columns:
        return {}
    y = df_scored[target].astype(int).values
    p = df_scored[proba_col].astype(float).values

    out: Dict[str, Any] = {}
    out["n"] = int(len(y))
    out["base_rate"] = float(np.mean(y)) if len(y) else None
    out["proba_mean"] = float(np.mean(p)) if len(p) else None
    out["proba_std"] = float(np.std(p)) if len(p) else None

    try:
        out["auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["auc"] = None
    try:
        out["brier"] = float(brier_score_loss(y, p))
    except Exception:
        out["brier"] = None
    try:
        out["logloss"] = float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)))
    except Exception:
        out["logloss"] = None

    ks, thr = ks_best_threshold(y, p)
    out.update({"ks": ks, "ks_threshold": thr})
    return out


# ============== CLI =================
def parse_args():
    p = argparse.ArgumentParser(
        "Apply imputer → binning → (WOE ou BIN selon training) → modèle → PD + risk buckets (sur score) sur OOS."
    )
    p.add_argument("--data", required=True, help="Chemin du OOS brut (parquet/csv).")
    p.add_argument("--out", required=True, help="Fichier de sortie scoré (parquet/csv).")
    # artefacts
    p.add_argument(
        "--imputer",
        default="artifacts/imputer/imputer.joblib",
        help="Imputer sklearn (obligatoire si données brutes).",
    )
    p.add_argument(
        "--imputer-meta",
        default="artifacts/imputer/imputer_meta.json",
        help="Meta imputer (optionnel, utilisé s’il existe, pour les noms de colonnes out).",
    )
    p.add_argument(
        "--bins",
        default="artifacts/binning_maxgini/bins.json",
        help="bins.json appris.",
    )
    p.add_argument(
        "--model",
        default="artifacts/model_from_binned/model_best.joblib",
        help="Modèle entraîné (joblib) avec clés model / best_lr / kept_woe / computed_woe / woe_maps / target.",
    )
    p.add_argument(
        "--buckets",
        default="artifacts/model_from_binned/risk_buckets.json",
        help="JSON avec 'edges' (sur SCORE) pour les classes (1..K).",
    )
    # colonnes & options
    p.add_argument(
        "--bin-suffix",
        default="__BIN",
        help="Suffixe/prefixe des colonnes de binning.",
    )
    p.add_argument(
        "--target",
        default="default_24m",
        help="Nom de la cible si présente dans OOS.",
    )
    p.add_argument(
        "--id-cols",
        default="loan_sequence_number,vintage",
        help="Colonnes à recopier dans la sortie (séparées par virgules).",
    )
    p.add_argument(
        "--drop-missing-flags",
        action="store_true",
        help="Supprime *_missing / was_missing_* avant traitement.",
    )
    return p.parse_args()


# ============== MAIN =================
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

    # 3) Binning appris (toujours sur les données imputées)
    learned = load_bins_json(args.bins)
    df_binned = transform_with_learned_bins(df_imp, learned)

    # 4) Modèle & artefacts
    art = load(args.model)
    if isinstance(art, dict):
        model_cal = art.get("model", None)
        best_lr = art.get("best_lr", None)
        kept_woe = art.get("kept_woe", None)
        computed_woe = bool(art.get("computed_woe", False))
        woe_maps = art.get("woe_maps", None)
        target_col = art.get("target", args.target)
    else:
        # fallback: artefact = estimator brut
        model_cal = art
        best_lr = None
        kept_woe = getattr(art, "feature_names_in_", None)
        computed_woe = False
        woe_maps = None
        target_col = args.target

    if kept_woe is None or len(kept_woe) == 0:
        raise SystemExit("L'artefact modèle ne contient pas 'kept_woe' non vide.")

    kept_woe = list(kept_woe)
    bin_tag = args.bin_suffix

    # 5) Construction de X cohérente avec le training
    any_woe_like = any(c.endswith("_WOE") for c in kept_woe)
    all_bin_like = bin_tag and all(c.endswith(bin_tag) or c.startswith(bin_tag) for c in kept_woe)

    if computed_woe:
        if not isinstance(woe_maps, dict):
            raise SystemExit("computed_woe=True mais 'woe_maps' est absent ou invalide dans l'artefact.")

        X = pd.DataFrame(index=df_binned.index)

        if any_woe_like:
            # Cas 1 : kept_woe = ['credit_score_WOE', ...]
            for feat in kept_woe:
                raw = feat[:-4]  # strip '_WOE'
                wm = woe_maps.get(raw)
                if wm is None:
                    continue
                bcol = resolve_bin_col(df_binned, raw, bin_tag)
                if bcol is None or bcol not in df_binned.columns:
                    continue
                ser = df_binned[bcol].astype("Int64")
                wmap = wm.get("map", {})
                wdef = float(wm.get("default", 0.0))
                X[feat] = ser.map(wmap).astype(float).fillna(wdef)

        elif all_bin_like:
            # Cas 2 : kept_woe = ['credit_score__BIN', ...] mais ces colonnes
            # contenaient des WOE pendant le fit. On reconstruit WOE avec les MÊMES noms.
            for feat in kept_woe:
                raw = strip_bin_tag(feat, bin_tag)
                wm = woe_maps.get(raw)
                if wm is None:
                    continue
                bcol = resolve_bin_col(df_binned, raw, bin_tag)
                if bcol is None or bcol not in df_binned.columns:
                    continue
                ser = df_binned[bcol].astype("Int64")
                wmap = wm.get("map", {})
                wdef = float(wm.get("default", 0.0))
                X[feat] = ser.map(wmap).astype(float).fillna(wdef)
        else:
            raise SystemExit(
                "computed_woe=True mais kept_woe ne ressemble ni à des *_WOE ni à des colonnes BIN.\n"
                f"Exemple kept_woe[:10] = {kept_woe[:10]}"
            )

        missing = [c for c in kept_woe if c not in X.columns]
        if missing:
            raise SystemExit(
                "Impossible de reconstruire toutes les features WOE attendues.\n"
                f"Manquantes (extrait) : {missing[:10]}"
            )

        # On essaie d'aligner sur feature_names_in_ si présent
        estimator_for_names = best_lr or model_cal
        feat_in = getattr(estimator_for_names, "feature_names_in_", None) if estimator_for_names is not None else None
        if feat_in is not None:
            feat_in = list(feat_in)
            missing_from_X = [c for c in feat_in if c not in X.columns]
            if missing_from_X:
                raise SystemExit(
                    "Colonnes attendues par l'estimateur manquantes dans X (WOE).\n"
                    f"Manquantes (extrait) : {missing_from_X[:10]}"
                )
            X = X[feat_in]
        else:
            X = X[kept_woe]

    else:
        # Training direct sur BIN (ou features déjà WOE dans les données) :
        # on aligne X sur les colonnes attendues par l'estimateur.
        estimator_for_names = best_lr or model_cal
        feat_in = getattr(estimator_for_names, "feature_names_in_", None) if estimator_for_names is not None else None

        if feat_in is not None:
            feat_names = list(feat_in)
        else:
            feat_names = kept_woe

        missing_bin = [c for c in feat_names if c not in df_binned.columns]
        if missing_bin:
            raise SystemExit(
                "Colonnes de features attendues par le modèle manquantes dans df_binned.\n"
                f"Manquantes (extrait) : {missing_bin[:10]}"
            )
        X = df_binned[feat_names].astype(float).fillna(0.0)

    # 6) PD (proba) : on privilégie le modèle calibré si dispo, sinon best_lr
    if model_cal is not None and hasattr(model_cal, "predict_proba"):
        proba = model_cal.predict_proba(X)[:, 1]
    elif best_lr is not None and hasattr(best_lr, "predict_proba"):
        proba = best_lr.predict_proba(X)[:, 1]
    else:
        est = best_lr or model_cal
        if est is None:
            raise SystemExit("Artefact modèle invalide : ni 'best_lr' ni 'model' n'est défini.")
        if hasattr(est, "decision_function"):
            z = est.decision_function(X)
        else:
            raise SystemExit("Impossible de calculer des probas: ni predict_proba ni decision_function disponibles.")
        proba = 1.0 / (1.0 + np.exp(-z))

    # 7) Score structurel pour le bucketing (LR brute en priorité)
    if best_lr is not None and hasattr(best_lr, "decision_function"):
        score = best_lr.decision_function(X)
    elif model_cal is not None and hasattr(model_cal, "decision_function"):
        score = model_cal.decision_function(X)
    else:
        # fallback : logit(proba)
        score = np.log(
            np.clip(proba, 1e-15, 1 - 1e-15) / np.clip(1.0 - proba, 1e-15, 1.0)
        )

    df_scored = pd.DataFrame({"proba": proba}, index=df_binned.index)

    # 8) Buckets (sur SCORE)
    if args.buckets and Path(args.buckets).exists():
        spec = json.loads(Path(args.buckets).read_text(encoding="utf-8"))
        edges = np.asarray(spec["edges"], dtype=float)
        df_scored["risk_bucket"] = assign_bucket(score, edges)

    # 9) Id-cols + cible (si disponible dans df_raw)
    if args.id_cols:
        for c in [c.strip() for c in args.id_cols.split(",") if c.strip()]:
            if c in df_raw.columns and c not in df_scored.columns:
                df_scored[c] = df_raw[c].values

    if target_col and target_col in df_raw.columns:
        df_scored[target_col] = df_raw[target_col].astype(int).values
        metrics = compute_metrics_if_target(df_scored, target_col, "proba")
    else:
        metrics = {}

    # 10) Save
    save_any(df_scored.reset_index(drop=True), out_path)

    # 11) Log
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
















fit_binning.py :

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
from pathlib import Path
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features.binning import (
    run_binning_maxgini_on_df, save_bins_json,
    DENYLIST_STRICT_DEFAULT, EXCLUDE_IDS_DEFAULT
)

def parse_args():
    p = argparse.ArgumentParser(description="Fit max|Gini| bins on TRAIN and apply to VALIDATION; save datasets + bins.json")
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--outdir", default="data/processed/merged/binned")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--format", choices=["parquet","csv"], default="parquet")
    # options binning
    p.add_argument("--bin-col-suffix", default="__BIN")
    p.add_argument("--include-missing", action="store_true")
    p.add_argument("--missing-label", default="__MISSING__")
    p.add_argument("--max-bins-categ", type=int, default=6)
    p.add_argument("--min-bin-size-categ", type=int, default=200)
    p.add_argument("--max-bins-num", type=int, default=6)
    p.add_argument("--min-bin-size-num", type=int, default=200)
    p.add_argument("--n-quantiles-num", type=int, default=50)
    p.add_argument("--min-gini-keep", type=float, default=None)
    # nouveautés : denylist + drop flags + parallélisme
    p.add_argument("--no-denylist", action="store_true", help="Ne pas appliquer la denylist stricte")
    p.add_argument("--drop-missing-flags", action="store_true", help="Supprimer les colonnes *_missing / was_missing_*")
    p.add_argument("--n-jobs-categ", type=int, default=-1, help="Nombre de jobs pour le binning catégoriel (joblib)")
    p.add_argument("--n-jobs-num", type=int, default=-1, help="Nombre de jobs pour le binning numérique (joblib)")
    return p.parse_args()

def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet",".pq"): return pd.read_parquet(p)
    return pd.read_csv(p)

def save_any(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".parquet",".pq"): df.to_parquet(path, index=False)
    else: df.to_csv(path, index=False)

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    artifacts = Path(args.artifacts); artifacts.mkdir(parents=True, exist_ok=True)

    tr = load_any(args.train); va = load_any(args.validation)
    if args.target not in tr.columns or args.target not in va.columns:
        raise SystemExit(f"Cible '{args.target}' absente du train/validation.")

    learned, tr_enriched, tr_binned = run_binning_maxgini_on_df(
        df=tr, target_col=args.target,
        include_missing=args.include_missing, missing_label=args.missing_label,
        max_bins_categ=args.max_bins_categ, min_bin_size_categ=args.min_bin_size_categ,
        max_bins_num=args.max_bins_num,   min_bin_size_num=args.min_bin_size_num,
        n_quantiles_num=args.n_quantiles_num,
        bin_col_suffix=args.bin_col_suffix,
        min_gini_keep=args.min_gini_keep,
        # nouveautés
        denylist_strict=([] if args.no_denylist else list(DENYLIST_STRICT_DEFAULT)),
        drop_missing_flags=bool(args.drop_missing_flags),
        n_jobs_categ=int(args.n_jobs_categ if args.n_jobs_categ != 0 else 1),
        n_jobs_num=int(args.n_jobs_num if args.n_jobs_num != 0 else 1),
        exclude_ids=EXCLUDE_IDS_DEFAULT
    )

    # Applique aux données de validation
    from features.binning import transform_with_learned_bins
    va_binned = transform_with_learned_bins(va, learned)

    # Sauvegardes datasets
    if args.format=="parquet":
        save_any(tr_binned, outdir / "train.parquet")
        save_any(va_binned, outdir / "validation.parquet")
    else:
        save_any(tr_binned, outdir / "train.csv")
        save_any(va_binned, outdir / "validation.csv")

    # bins.json
    save_bins_json(learned, artifacts / "bins.json")

    print("✔ Saved bins:", artifacts / "bins.json")
    print("✔ Datasets :", outdir)

if __name__ == "__main__":
    main()

















impute_and_save.py :


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Impute train/validation with DataImputer, then save either:
- Parquet files (types conservés nativement), or
- CSV + pickles des dtypes (parse_dates, cat_dtypes, other_dtypes)

Nouveautés :
- --labels-window-dir + --use-splits : lit window=.../_splits.json pour construire
  automatiquement train (pooled.parquet) et validation (liste de quarters ou oos).
- Supporte plusieurs quarters de validation (concat).
- Rétro-compatible avec --train-csv / --validation-csv.

Sorties :
- data/processed/imputed/{train.parquet, validation.parquet} (ou CSV)
- artifacts/imputer/imputer.joblib + imputer_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from joblib import dump

# Rendez le package importable depuis ./src
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from features.impute import DataImputer  # adapte si besoin


# ----------------------------- CLI -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Impute & save datasets + persist imputer")

    # Mode explicite (rétro-compatible)
    p.add_argument("--train-csv", help="Fichier d'entraînement (CSV/Parquet)")
    p.add_argument("--validation-csv", help="Fichier de validation (CSV/Parquet)")

    # Mode basé sur les splits produits par make_labels.py
    p.add_argument(
        "--labels-window-dir",
        help="Chemin du dossier window=XXm (ex: data/processed/default_labels/window=24m)"
    )
    p.add_argument(
        "--use-splits",
        action="store_true",
        help="Utiliser window/_splits.json pour construire train/validation"
    )

    p.add_argument("--target", default=None, help="Nom de la colonne cible (optionnel)")
    p.add_argument("--outdir", default="data/processed/imputed", help="Dossier de sortie")
    p.add_argument(
        "--artifacts",
        default="artifacts/imputer",
        help="Dossier où sauvegarder l'imputer et la méta"
    )
    p.add_argument(
        "--format", choices=["parquet", "csv"], default="parquet",
        help="Format de sauvegarde des jeux imputés"
    )
    p.add_argument("--use-cohort", action="store_true", help="Imputation par cohortes")
    p.add_argument("--missing-flag", action="store_true", help="Ajouter les indicateurs was_missing_*")
    p.add_argument("--fail-on-nan", action="store_true", help="Échouer si des NaN subsistent après imputation")

    return p.parse_args()


# ----------------------------- I/O helpers -----------------------------
def load_any(path: str | Path) -> pd.DataFrame:
    """Charge un DataFrame depuis Parquet ou CSV (déduit de l'extension)."""
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def concat_parquet(paths: List[Path]) -> pd.DataFrame:
    """Concatène une liste de fichiers parquet (schema-align si nécessaire)."""
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        if p.exists():
            dfs.append(pd.read_parquet(p))
    if not dfs:
        return pd.DataFrame()
    all_cols = sorted(set().union(*[df.columns for df in dfs]))
    dfs = [df.reindex(columns=all_cols) for df in dfs]
    return pd.concat(dfs, ignore_index=True)


def save_parquet(df_train_imp: pd.DataFrame, df_val_imp: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "train.parquet").unlink(missing_ok=True)
    (outdir / "validation.parquet").unlink(missing_ok=True)
    df_train_imp.to_parquet(outdir / "train.parquet", index=False)
    df_val_imp.to_parquet(outdir / "validation.parquet", index=False)
    print(f"✔ Parquet écrit dans {outdir}")


def save_csv_with_dtypes(df_train_imp: pd.DataFrame, df_val_imp: pd.DataFrame, outdir: Path):
    from pandas.api.types import CategoricalDtype  # import local pour cohérence pickle

    outdir.mkdir(parents=True, exist_ok=True)

    dtypes = df_train_imp.dtypes.to_dict()
    parse_dates = [c for c, dt in dtypes.items() if str(dt).startswith("datetime64")]
    cat_dtypes = {c: dt for c, dt in dtypes.items() if isinstance(dt, CategoricalDtype)}
    other_dtypes = {
        c: ("Int64" if str(dt).startswith("int") and df_train_imp[c].isna().any() else dt)
        for c, dt in dtypes.items()
        if c not in parse_dates and c not in cat_dtypes
    }

    with open(outdir / "parse_dates.pkl", "wb") as f:
        pickle.dump(parse_dates, f)
    with open(outdir / "cat_dtypes.pkl", "wb") as f:
        pickle.dump(cat_dtypes, f)
    with open(outdir / "other_dtypes.pkl", "wb") as f:
        pickle.dump(other_dtypes, f)

    df_train_imp.to_csv(outdir / "train_imputed.csv", index=False)
    df_val_imp.to_csv(outdir / "validation_imputed.csv", index=False)
    print(f"✔ CSV + dtypes picklés écrits dans {outdir}")


# ----------------------------- Splits loader -----------------------------
def _pull(d: dict, key: str, default=None):
    """Cherche une clé à la racine, sinon dans d['splits'], sinon default."""
    if key in d:
        return d[key]
    s = d.get("splits", {})
    return s.get(key, default)


def resolve_splits(labels_window_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Construit (train_df, val_df) à partir de window/_splits.json :
      - train = pooled.parquet
      - validation = concat(quartiers listés) OU oos.parquet selon 'validation_mode'
    Supporte les manifests où les clés sont à la racine OU sous "splits".
    """
    splits_path = labels_window_dir / "_splits.json"
    pooled_path = labels_window_dir / "pooled.parquet"
    oos_path = labels_window_dir / "oos.parquet"

    if not splits_path.exists():
        raise SystemExit(f"[ERR] Splits file not found: {splits_path}")

    manifest = json.loads(splits_path.read_text(encoding="utf-8"))

    # Train
    if not pooled_path.exists():
        raise SystemExit(f"[ERR] Train 'pooled.parquet' not found at {pooled_path}")
    df_train = load_any(pooled_path)

    # Récupération robuste des infos de validation
    mode = (_pull(manifest, "validation_mode", None) or "quarters").lower()

    # 1) validation_quarters normalisés
    val_quarters = _pull(manifest, "validation_quarters", [])
    # 2) rétro-compat via splits.explicit
    explicit = _pull(manifest, "explicit", {}) or {}
    if not val_quarters:
        val_quarters = list(explicit.get("validation_quarters", []))
    if not val_quarters:
        val_quarters = list(explicit.get("default_val_quarters", []))
    if not val_quarters and explicit.get("default_val_quarter"):
        val_quarters = [explicit["default_val_quarter"]]

    # 3) oos_quarters
    oos_quarters = _pull(manifest, "oos_quarters", [])
    if not oos_quarters:
        oos_quarters = list(explicit.get("oos_quarters", []))

    # Si mode=quarters mais aucune liste fournie, on bascule intelligemment
    if mode == "quarters" and not val_quarters:
        if oos_path.exists() and oos_quarters:
            print("[WARN] Pas de validation_quarters trouvés → fallback sur OOS.")
            mode = "oos"
        else:
            raise SystemExit("[ERR] No 'validation_quarters' found and no oos fallback available.")

    # Build validation
    if mode == "oos":
        if not oos_path.exists():
            raise SystemExit(f"[ERR] Validation 'oos.parquet' not found at {oos_path}")
        df_val = load_any(oos_path)
        used_quarters = oos_quarters
    else:
        files = [labels_window_dir / f"quarter={q}" / "data.parquet" for q in val_quarters]
        missing = [str(p) for p in files if not p.exists()]
        if missing:
            raise SystemExit(f"[ERR] Missing validation quarter files:\n  " + "\n  ".join(missing))
        df_val = concat_parquet(files)
        used_quarters = val_quarters

    # Align colonnes
    all_cols = sorted(set(df_train.columns).union(df_val.columns))
    df_train = df_train.reindex(columns=all_cols)
    df_val = df_val.reindex(columns=all_cols)

    # Logs utiles
    print(f"[SPLITS] mode: {mode}")
    if mode == "quarters":
        print(f"[SPLITS] validation_quarters: {used_quarters}")
    else:
        print(f"[SPLITS] oos_quarters: {used_quarters}")

    # Méta enrichie
    meta = {
        "train_file": str(pooled_path),
        "validation_mode": mode,
        "validation_used_quarters": used_quarters,
        "validation_paths": (
            [str(labels_window_dir / f"quarter={q}" / "data.parquet") for q in used_quarters]
            if mode != "oos" else [str(oos_path)]
        )
    }
    # Conserver le manifest brut pour traçabilité
    manifest["_resolved"] = meta
    return df_train, df_val, manifest


# ----------------------------- Main -----------------------------
def main():
    args = parse_args()

    outdir = Path(args.outdir)
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    # ----------------- Choix de la source (explicite vs splits) -----------------
    df_train: Optional[pd.DataFrame] = None
    df_val: Optional[pd.DataFrame] = None
    splits_meta: Optional[dict] = None

    if args.use_splits:
        if not args.labels_window_dir:
            raise SystemExit("[ERR] --use-splits nécessite --labels-window-dir")
        labels_dir = Path(args.labels_window_dir)
        if not labels_dir.exists():
            raise SystemExit(f"[ERR] labels_window_dir introuvable : {labels_dir}")
        df_train, df_val, splits_meta = resolve_splits(labels_dir)
    else:
        if not args.train_csv or not args.validation_csv:
            raise SystemExit("[ERR] Fournir --train-csv et --validation-csv, ou utiliser --use-splits avec --labels-window-dir")
        df_train = load_any(args.train_csv)
        df_val = load_any(args.validation_csv)

    # ----------------- Séparer (optionnel) la cible -----------------
    y_train = y_val = None
    if args.target and args.target in df_train.columns:
        y_train = df_train[args.target].copy()
        df_train = df_train.drop(columns=[args.target])
    if args.target and args.target in df_val.columns:
        y_val = df_val[args.target].copy()
        df_val = df_val.drop(columns=[args.target])

    # ----------------- Imputation -----------------
    imputer = DataImputer(use_cohort=args.use_cohort, missing_flag=args.missing_flag)
    imputer.fit(df_train)
    df_train_imp = imputer.transform(df_train)
    df_val_imp = imputer.transform(df_val)

    # Ré-attacher la cible si présente
    if y_train is not None:
        df_train_imp[args.target] = y_train.values
    if y_val is not None:
        df_val_imp[args.target] = y_val.values

    # Optionnel : garde-fou qualité
    if args.fail_on_nan:
        bad_train = df_train_imp.columns[df_train_imp.isna().any()].tolist()
        bad_val = df_val_imp.columns[df_val_imp.isna().any()].tolist()
        if bad_train or bad_val:
            raise SystemExit(
                f"NaN résiduels après imputation.\n"
                f"Train: {bad_train}\nValidation: {bad_val}"
            )

    # ----------------- Sauvegarde des datasets -----------------
    if args.format == "parquet":
        save_parquet(df_train_imp, df_val_imp, outdir)
    else:
        save_csv_with_dtypes(df_train_imp, df_val_imp, outdir)

    # ----------------- Persistance imputer + méta -----------------
    dump(imputer, artifacts / "imputer.joblib")

    meta = {
        "target": args.target,
        "use_cohort": args.use_cohort,
        "missing_flag": args.missing_flag,
        "features": list(df_train_imp.drop(columns=[args.target]).columns) if args.target else list(df_train_imp.columns),
        "output_dir": str(outdir.resolve()),
        "format": args.format,
        "mode": "splits" if args.use_splits else "explicit",
    }

    if args.use_splits and splits_meta is not None:
        meta["splits"] = splits_meta
    else:
        meta["train_path"] = str(Path(args.train_csv).resolve())
        meta["validation_path"] = str(Path(args.validation_csv).resolve())

    (artifacts / "imputer_meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"✔ Imputer sauvegardé dans {artifacts / 'imputer.joblib'}")


if __name__ == "__main__":
    main()





















make_labels :

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build default labels (default_{T}m) per quarter, in parallel.
- Lit TOUTE la config dans config.yml (data.*, labels.*, output.*, splits.*)
- Écrit : quarter=YYYYQn/data.{fmt}, pooled.{fmt} (design), oos.{fmt} (hold-out)
- 'vintage' est calculé selon labels.vintage_basis :
    * "first_payment_date" (historique, FPD → Quarter)
    * "origination_quarter" (RECOMMANDÉ ici : le quarter du fichier source)
- IMPORTANT : le manifest _splits.json inclut
    * validation_mode: "quarters"
    * validation_quarters: [...]
    * oos_quarters: [...]
  pour être compatible avec impute_and_save.resolve_splits().
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------------------------------------------------------------
# Import du builder de labels
# ------------------------------------------------------------------
from features.labels import build_default_labels


# ========================= Helpers I/O ============================
def qpaths(root: Path, q: str) -> Tuple[Path, Path]:
    d = root / f"historical_data_{q}"
    return d / f"historical_data_{q}.txt", d / f"historical_data_time_{q}.txt"


def _to_parquet_safe(df: pd.DataFrame, path: Path):
    """Parquet ne supporte pas Period → cast en timestamps fin de mois."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_period_dtype(out[c]):
            out[c] = out[c].dt.to_timestamp("M")
    out.to_parquet(path, index=False)


def _to_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_df(df: pd.DataFrame, path: Path, fmt: str = "parquet"):
    if fmt == "parquet":
        _to_parquet_safe(df, path)
    else:
        _to_csv(df, path)


def _concat_parquet(paths: List[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow.parquet as pq
        import pyarrow as pa  # noqa: F401  (utile si cast)
        writer = None
        schema = None
        wrote = False
        for p in paths:
            if not p.exists():
                continue
            tbl = pq.read_table(p)
            if writer is None:
                schema = tbl.schema
                writer = pq.ParquetWriter(str(out_path), schema)
            else:
                if tbl.schema != schema:
                    tbl = tbl.cast(schema)
            writer.write_table(tbl)
            wrote = True
        if writer is not None:
            writer.close()
        if not wrote:
            pd.DataFrame().to_parquet(out_path, index=False)
    except Exception:
        dfs = [pd.read_parquet(p) for p in paths if p.exists()]
        if dfs:
            pd.concat(dfs, ignore_index=True).to_parquet(out_path, index=False)
        else:
            pd.DataFrame().to_parquet(out_path, index=False)


def _concat_csv(paths: List[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with open(out_path, "w", newline="") as fout:
        for p in paths:
            if not p.exists():
                continue
            with open(p, "r") as fin:
                if wrote_header:
                    next(fin, None)  # skip header
                for line in fin:
                    fout.write(line)
            wrote_header = True


def concat_outputs(paths: List[Path], out_path: Path, fmt: str):
    if fmt == "parquet":
        _concat_parquet(paths, out_path)
    else:
        _concat_csv(paths, out_path)


# ===================== Vintage computation =======================
def _infer_vintage_fpd(df: pd.DataFrame) -> pd.Series:
    """Vintage basé sur first_payment_date (FPD)."""
    fpd = df["first_payment_date"]
    qnum = ((fpd.dt.month - 1) // 3 + 1).astype("int")
    return fpd.dt.year.astype("string") + "Q" + qnum.astype("string")


def _infer_vintage_orig(df: pd.DataFrame) -> pd.Series:
    """
    Vintage = quarter d'ORIGINATION. On utilise la colonne __file_quarter,
    ajoutée par le worker en fonction du dossier source.
    """
    if "__file_quarter" not in df.columns:
        # Safety net: si absent, tenter origination_date si dispo
        if "origination_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["origination_date"]):
            od = df["origination_date"]
            qnum = ((od.dt.month - 1) // 3 + 1).astype("int")
            return od.dt.year.astype("string") + "Q" + qnum.astype("string")
        # Sinon, fallback FPD
        return _infer_vintage_fpd(df)
    return df["__file_quarter"].astype("string")


def compute_vintage(df: pd.DataFrame, basis: str) -> pd.Series:
    basis = (basis or "first_payment_date").lower().strip()
    if basis in ("first_payment_date", "fpd"):
        return _infer_vintage_fpd(df)
    elif basis in ("origination_quarter", "orig", "origination"):
        return _infer_vintage_orig(df)
    else:
        # fallback sûr
        return _infer_vintage_fpd(df)


# ========================= Worker ================================
def quarter_worker(
    q: str,
    root: str,
    window_dir: str,
    fmt: str,
    window_months: int,
    delinquency_threshold: int,
    liquidation_codes: Tuple[str, ...],
    include_ra: bool,
    require_full_window: bool,
    vintage_basis: str,
) -> Dict[str, Any]:
    """
    Calcule les labels pour un quarter, sauvegarde le fichier, et renvoie un récap.
    """
    try:
        root_p = Path(root)
        window_dir_p = Path(window_dir)
        orig, perf = qpaths(root_p, q)
        if not orig.exists() or not perf.exists():
            return {"quarter": q, "ok": False, "error": f"missing files: {orig} / {perf}"}

        df = build_default_labels(
            path_orig=str(orig),
            path_perf=str(perf),
            window_months=window_months,
            delinquency_threshold=delinquency_threshold,
            liquidation_codes=liquidation_codes,
            include_ra=include_ra,
            require_full_window=require_full_window,
        )

        # Trace du quarter source pour retrouver "vintage=origination_quarter"
        df["__file_quarter"] = q

        # Vintage selon la base demandée
        df["vintage"] = compute_vintage(df, vintage_basis)

        q_dir = window_dir_p / f"quarter={q}"
        out_path = q_dir / f"data.{fmt}"
        _save_df(df, out_path, fmt)

        lbl_col = f"default_{window_months}m"
        return {
            "quarter": q,
            "ok": True,
            "path": str(out_path),
            "n_rows": int(len(df)),
            "n_unique_loans": int(df["loan_sequence_number"].nunique()) if "loan_sequence_number" in df.columns else None,
            "default_rate": float(df[lbl_col].mean()) if lbl_col in df.columns and len(df) > 0 else None,
        }
    except Exception as e:
        return {"quarter": q, "ok": False, "error": repr(e)}


# ========================= CLI / MAIN ============================
def parse_args():
    p = argparse.ArgumentParser(description="Build default labels per quarter (parallel), splits via YAML.")
    p.add_argument("--config", default="config.yml", help="Chemin du YAML (data.*, labels.*, output.*, splits.*)")
    p.add_argument("--workers", type=int, default=None, help="Nombre de workers (défaut = CPU-1)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    # ------------- Lecture config -------------
    data_cfg = cfg.get("data", {})
    labels_cfg = cfg.get("labels", {})
    output_cfg = cfg.get("output", {})
    splits_cfg = cfg.get("splits", {})

    root = Path(data_cfg["root"])
    quarters_all: List[str] = list(data_cfg.get("quarters", []))

    window_months = int(labels_cfg.get("window_months", 24))
    delinquency_threshold = int(labels_cfg.get("delinquency_threshold", 3))
    liquidation_codes = tuple(labels_cfg.get("liquidation_codes", ["02", "03", "09"]))
    include_ra = bool(labels_cfg.get("include_ra", True))
    require_full_window = bool(labels_cfg.get("require_full_window", False))
    vintage_basis = str(labels_cfg.get("vintage_basis", "origination_quarter"))

    out_dir = Path(output_cfg.get("dir", "data/processed/default_labels"))
    out_fmt = str(output_cfg.get("format", "parquet")).lower()
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = (splits_cfg.get("mode", "explicit") or "explicit").lower()
    explicit_cfg = splits_cfg.get("explicit", {}) if mode == "explicit" else {}

    # Clés normalisées + rétro-compat
    design_quarters: List[str] = list(explicit_cfg.get("design_quarters", []))
    validation_quarters: List[str] = list(explicit_cfg.get("validation_quarters", []))
    if not validation_quarters:
        # rétro-compat : default_val_quarters / default_val_quarter
        validation_quarters = list(explicit_cfg.get("default_val_quarters", []))
        if not validation_quarters and explicit_cfg.get("default_val_quarter"):
            validation_quarters = [explicit_cfg["default_val_quarter"]]
    oos_quarters: List[str] = list(explicit_cfg.get("oos_quarters", []))

    # ------------- Sanity checks / logs -------------
    def _warn(msg: str):
        print(f"[WARN] {msg}")

    requested = sorted(set(design_quarters) | set(oos_quarters))
    if quarters_all:
        missing = [q for q in requested if q not in quarters_all]
        if missing:
            _warn(f"Les quarters {missing} sont demandés par splits.explicit mais absents de data.quarters.")
        quarters_to_build = [q for q in requested if q in (quarters_all or requested)]
    else:
        quarters_to_build = requested

    if any(q not in design_quarters for q in validation_quarters):
        diff = [q for q in validation_quarters if q not in design_quarters]
        _warn(f"validation_quarters contient des quarters hors design_quarters: {diff}")

    if not design_quarters:
        _warn("design_quarters est vide → pooled sera vide.")
    if not validation_quarters:
        _warn("validation_quarters est vide → validation interne vide (imputer retombera sur OOS si présent).")
    if not oos_quarters:
        print("[INFO] Aucun oos_quarters renseigné (oos.{fmt} sera vide).")

    print(f"[INFO] design_quarters: {design_quarters}")
    print(f"[INFO] validation_quarters: {validation_quarters}")
    print(f"[INFO] oos_quarters: {oos_quarters}")

    window_dir = out_dir / f"window={window_months}m"
    window_dir.mkdir(parents=True, exist_ok=True)

    # ------------- Build per quarter (parallel) -------------
    t0 = time.time()
    results: List[Dict[str, Any]] = []
    produced_quarters: List[str] = []

    if quarters_to_build:
        workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
        print(f"→ Building {len(quarters_to_build)} quarters with {workers} workers...")
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(
                    quarter_worker,
                    q,
                    str(root),
                    str(window_dir),
                    out_fmt,
                    window_months,
                    delinquency_threshold,
                    liquidation_codes,
                    include_ra,
                    require_full_window,
                    vintage_basis,
                )
                for q in quarters_to_build
            ]
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                if r.get("ok"):
                    produced_quarters.append(r["quarter"])
                    print(f"✔ {r['quarter']} saved ({r.get('n_rows','?')} rows)")
                else:
                    _warn(f"{r['quarter']} failed: {r.get('error')}")
    else:
        raise SystemExit("[ERR] Aucun quarter à produire : vérifiez data.quarters et splits.explicit.*")

    # ------------- Summary -------------
    lbl_col = f"default_{window_months}m"
    summary_rows = [
        {k: r.get(k) for k in ("quarter", "n_rows", "n_unique_loans", "default_rate", "ok", "path", "error")}
        for r in sorted(results, key=lambda x: x["quarter"])
    ]
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(window_dir / "_summary.csv", index=False)

    # ------------- pooled (design) & oos concatenations -------------
    ok_map: Dict[str, Path] = {
        r["quarter"]: Path(r["path"]) for r in results if r.get("ok") and r.get("path")
    }

    # pooled = concat(design_quarters présents)
    pooled_list = [q for q in design_quarters if q in ok_map]
    if pooled_list:
        concat_outputs([ok_map[q] for q in pooled_list], window_dir / f"pooled.{out_fmt}", out_fmt)
        print(f"✔ pooled -> {window_dir / f'pooled.{out_fmt}'} "
              f"(quarters: {pooled_list[:3]}{'...' if len(pooled_list) > 3 else ''})")
    else:
        _warn("Aucun quarter pour pooled (design_quarters vide ou non produits).")

    # oos = concat(oos_quarters présents)
    oos_list = [q for q in oos_quarters if q in ok_map]
    if oos_list:
        concat_outputs([ok_map[q] for q in oos_list], window_dir / f"oos.{out_fmt}", out_fmt)
        print(f"✔ oos -> {window_dir / f'oos.{out_fmt}'} "
              f"(quarters: {oos_list[:3]}{'...' if len(oos_list) > 3 else ''})")
    else:
        print("[INFO] Pas de oos (liste vide ou non produits).")

    # ------------- Manifest (compatible resolve_splits) -------------
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_dir": str(out_dir.resolve()),
        "window_months": window_months,
        "format": out_fmt,
        "vintage_basis": vintage_basis,
        "quarters_requested": quarters_to_build,
        "quarters_produced": produced_quarters,
        # Bloc "splits" minimal + rétro-compat :
        "splits": {
            "mode": mode,
            "explicit": {
                "design_quarters": design_quarters,
                # Rétro-compat d'affichage : on garde les anciennes clés aussi
                "default_val_quarters": validation_quarters,
                "oos_quarters": oos_quarters,
            },
            # CLÉS NORMALISÉES attendues par resolve_splits()
            "validation_mode": "quarters",
            "validation_quarters": validation_quarters,
            "oos_quarters": oos_quarters,
        },
        "elapsed_sec": round(time.time() - t0, 2),
        "config_snapshot": cfg,
    }

    (window_dir / "_splits.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("✔ Manifest written:", window_dir / "_splits.json")


if __name__ == "__main__":
    main()

















results.py :

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
results.py — Vintage × Risk Grade report

Pour chaque vintage et chaque grade (classe de risque), le script retourne :
- grade (label)
- bornes de la classe [lower, upper]
- effectif (n)
- nb de défauts (n_default) si la colonne cible est fournie
- taux de défaut (default_rate) si la colonne cible est fournie
- probabilité de défaut de la classe issue de l'entraînement (class_pd_train), si présente dans risk_buckets.json
- moyenne des probabilités prédites (mean_proba)
- (optionnel) n_train : effectif d'entraînement par grade si présent dans risk_buckets.json

Entrées attendues :
- un dataset "scored" avec au minimum : 'proba' et 'vintage'
- un fichier JSON de buckets (edges/labels/etc.) sauvegardé à l'entraînement

Exemple :
    python results.py \
        --data data/processed/scored/test_scored.parquet \
        --buckets artifacts/model/risk_buckets.json \
        --target default_24m \
        --out reports/by_vintage_grades.csv
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------ I/O utils ------------------------------
def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def save_any(df: pd.DataFrame, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(p, index=False)
    else:
        df.to_csv(p, index=False)


# ------------------------------ Buckets utils ------------------------------
def load_buckets(buckets_path: str) -> Dict:
    """
    Attend un JSON de la forme (schéma souple, champs optionnels) :
    {
      "edges": [ ... ],                   # len = K+1 (obligatoire)
      "labels": ["G01", ...],             # len = K (optionnel)
      "bucket_pd_train": [p1..pK],        # PD de classe sur TRAIN/OOF (optionnel)
      "bucket_pd": [p1..pK],              # alias possible (optionnel)
      "pd_train": [p1..pK],               # alias possible (optionnel)
      "class_pd": [p1..pK],               # alias possible (optionnel)
      "bucket_n_train": [n1..nK]          # effectifs TRAIN par classe (optionnel)
    }
    """
    with open(buckets_path, "r") as f:
        cfg = json.load(f)

    if "edges" not in cfg or not isinstance(cfg["edges"], list) or len(cfg["edges"]) < 2:
        raise ValueError("risk_buckets.json: 'edges' manquant ou invalide.")

    edges = np.asarray(cfg["edges"], dtype=float)
    if not np.all(np.diff(edges) >= 0):
        raise ValueError("risk_buckets.json: 'edges' doivent être triés et non décroissants.")

    K = len(edges) - 1

    labels = cfg.get("labels")
    if labels is None or len(labels) != K:
        labels = [f"G{str(i+1).zfill(2)}" for i in range(K)]

    # PD de classe issue de l'entraînement (accepte plusieurs clés)
    pd_train = (
        cfg.get("bucket_pd_train")
        or cfg.get("bucket_pd")
        or cfg.get("pd_train")
        or cfg.get("class_pd")
    )
    if pd_train is not None:
        if len(pd_train) != K:
            raise ValueError("risk_buckets.json: longueur de 'bucket_pd*_train' incompatible avec edges.")
        pd_train = np.asarray(pd_train, dtype=float)

    # Effectifs d'entraînement par classe (optionnel)
    n_train = cfg.get("bucket_n_train")
    if n_train is not None:
        if len(n_train) != K:
            raise ValueError("risk_buckets.json: longueur de 'bucket_n_train' incompatible avec edges.")
        n_train = np.asarray(n_train, dtype=int)

    return {
        "edges": edges,
        "labels": labels,
        "pd_train": pd_train,   # peut être None
        "n_train": n_train,     # peut être None
        "n_bins": K,
    }


def assign_bucket_indices(proba: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Assigne chaque probabilité au bin correspondant, avec la convention right=True :
        bins[i-1] < x <= bins[i]
    Retourne des indices de 0 à K-1.
    """
    idx = np.digitize(proba, edges, right=True) - 1
    # clamp (sécurité pour valeurs extrêmes ou proba==edges[0])
    idx = np.clip(idx, 0, len(edges) - 2)
    return idx


# ------------------------------ Vintage sorting ------------------------------
def _parse_vintage(v: str) -> Tuple[int, int]:
    """
    Convertit 'YYYYQn' en (YYYY, n) pour trier correctement.
    Si parsing impossible, renvoie (très grand) pour pousser en bas.
    """
    try:
        year = int(v[:4])
        q = int(v[-1])
        return year, q
    except Exception:
        return (10**9, 9)


# ------------------------------ Core reporting ------------------------------
def build_bucket_meta(bkt: Dict) -> pd.DataFrame:
    edges = bkt["edges"]
    labels = bkt["labels"]
    pd_train = bkt.get("pd_train")
    n_train = bkt.get("n_train")
    K = len(labels)

    lower = edges[:-1]
    upper = edges[1:]

    meta = pd.DataFrame({
        "bucket_index": np.arange(K, dtype=int),
        "grade": labels,
        "lower": lower,
        "upper": upper
    })

    meta["class_pd_train"] = np.nan if pd_train is None else pd_train
    if n_train is not None:
        meta["n_train"] = n_train

    return meta


def make_report(
    df: pd.DataFrame,
    bkt: Dict,
    proba_col: str = "proba",
    vintage_col: str = "vintage",
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Produit un DataFrame agrégé :
    [vintage, grade, bucket_index, lower, upper, class_pd_train, (n_train?), n, n_default, default_rate, mean_proba]
    """
    if proba_col not in df.columns:
        raise ValueError(f"Colonne '{proba_col}' absente du dataset.")

    if vintage_col not in df.columns:
        raise ValueError(f"Colonne '{vintage_col}' absente du dataset.")

    edges = bkt["edges"]
    idx = assign_bucket_indices(df[proba_col].to_numpy(dtype=float), edges)

    meta = build_bucket_meta(bkt)

    work = df.copy()
    work["bucket_index"] = idx

    if target_col and target_col in work.columns:
        y = work[target_col].astype(int)
        work = work.assign(_y=y)
        agg = work.groupby([vintage_col, "bucket_index"], as_index=False).agg(
            n=("bucket_index", "size"),
            n_default=("_y", "sum"),
            mean_proba=(proba_col, "mean"),
        )
        agg["default_rate"] = agg["n_default"] / agg["n"]
    else:
        agg = work.groupby([vintage_col, "bucket_index"], as_index=False).agg(
            n=("bucket_index", "size"),
            mean_proba=(proba_col, "mean"),
        )
        agg["n_default"] = np.nan
        agg["default_rate"] = np.nan

    # jointure avec les métadonnées de classe
    rep = agg.merge(meta, on="bucket_index", how="left")

    # tri par vintage chronologique puis par bucket
    rep["__v_key"] = rep[vintage_col].astype(str).map(_parse_vintage)
    rep = rep.sort_values(["__v_key", "bucket_index"]).drop(columns="__v_key")

    # colonnes finales (inclut n_train si présent)
    base_cols = [
        vintage_col, "grade", "bucket_index",
        "lower", "upper", "class_pd_train",
    ]
    if "n_train" in rep.columns:
        base_cols.append("n_train")
    base_cols += ["n", "n_default", "default_rate", "mean_proba"]
    rep = rep[base_cols]

    # arrondis élégants
    for c in ("lower", "upper", "class_pd_train", "default_rate", "mean_proba"):
        if c in rep.columns:
            rep[c] = rep[c].astype(float).round(6)

    return rep


# ------------------------------ CLI ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Vintage × grade report à partir d'un dataset scoré et d'un fichier de buckets.")
    p.add_argument("--data", required=True, help="CSV/Parquet contenant au minimum 'proba' et 'vintage'.")
    p.add_argument("--buckets", required=True, help="risk_buckets.json sauvegardé à l'entraînement (edges, labels, bucket_pd_train...).")
    p.add_argument("--target", default=None, help="Nom de la colonne cible binaire (ex: default_24m).")
    p.add_argument("--proba-col", default="proba", help="Nom de la colonne des probabilités.")
    p.add_argument("--vintage-col", default="vintage", help="Nom de la colonne vintage (format 'YYYYQn').")
    p.add_argument("--out", default=None, help="Chemin de sortie (CSV/Parquet). Si non fourni, imprime sur stdout.")
    return p.parse_args()


def main():
    args = parse_args()

    df = load_any(args.data)
    buckets = load_buckets(args.buckets)

    report = make_report(
        df=df,
        bkt=buckets,
        proba_col=args.proba_col,
        vintage_col=args.vintage_col,
        target_col=args.target,
    )

    if args.out:
        save_any(report, args.out)
        print(f"✔ Report written to: {args.out}")
    else:
        with pd.option_context("display.max_rows", None, "display.width", 160, "display.max_columns", None):
            print(report)


if __name__ == "__main__":
    main()






















train_model.py :

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, time, contextlib, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from joblib import dump

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, roc_curve
from sklearn.model_selection import StratifiedKFold, GridSearchCV

# -----------------------------
# I/O helpers
# -----------------------------
def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")

# -----------------------------
# Timing utils (overhead négligeable)
# -----------------------------
class Timer:
    def __init__(self, live: bool = False, stream=None):
        self.records: Dict[str, float] = {}
        self._stack: List[str] = []
        self.live = bool(live)
        self.stream = stream or sys.stdout

    @contextlib.contextmanager
    def section(self, name: str):
        t0 = time.perf_counter()
        self._stack.append(name)
        if self.live:
            print(f"▶ {name} ...", file=self.stream, flush=True)
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.records[name] = self.records.get(name, 0.0) + dt
            self._stack.pop()
            if self.live:
                print(f"  ✓ {name:22s} {dt:8.3f}s", file=self.stream, flush=True)

    def add(self, name: str, dt: float):
        self.records[name] = self.records.get(name, 0.0) + dt

# -----------------------------
# Metrics & utils
# -----------------------------
def ks_best_threshold(y, p):
    y = pd.Series(y).astype(int).to_numpy()
    p = pd.Series(p).astype(float).to_numpy()
    if np.unique(y).size < 2:
        return np.nan, np.nan
    fpr, tpr, thr = roc_curve(y, p)
    ks_arr = tpr - fpr
    i = int(np.nanargmax(ks_arr))
    return float(ks_arr[i]), float(thr[i])

def decile_table(y, p, q=10):
    df = pd.DataFrame({"y": pd.Series(y).astype(int), "p": pd.Series(p).astype(float)})
    try:
        df["decile"] = pd.qcut(df["p"], q=q, labels=False, duplicates="drop")
    except Exception:
        n = len(df)
        ranks = df["p"].rank(method="first") / max(n, 1)
        df["decile"] = pd.cut(ranks, bins=np.linspace(0, 1, q + 1), labels=False, include_lowest=True)
    tab = (
        df.groupby("decile", dropna=True)
        .agg(events=("y", "sum"), count=("y", "size"), avg_p=("p", "mean"))
        .sort_index(ascending=False)
    )
    if tab.empty:
        return tab
    tab["rate"] = tab["events"] / tab["count"].where(tab["count"] > 0, 1)
    tab["cum_events"] = tab["events"].cumsum()
    tab["cum_count"] = tab["count"].cumsum()
    tot_e = float(tab["events"].sum())
    cum_good = tab["count"] - tab["events"]
    denom_good = float(cum_good.sum()) if float(cum_good.sum()) > 0 else 1.0
    tab["TPR"] = tab["cum_events"] / (tot_e if tot_e > 0 else 1.0)
    tab["FPR"] = cum_good.cumsum() / denom_good
    tab["KS"] = tab["TPR"] - tab["FPR"]
    return tab

def psi(a, b, bins=10, eps=1e-9):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    q = np.quantile(a, np.linspace(0, 1, bins + 1))
    q = np.unique(q)
    if q.size < 2:
        return 0.0
    q[0], q[-1] = -np.inf, np.inf
    for i in range(1, len(q)):
        if not (q[i] > q[i - 1]):
            q[i] = np.nextafter(q[i - 1], np.inf)
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    pa, pb = ca / max(ca.sum(), 1), cb / max(cb.sum(), 1)
    return float(np.sum((pa - pb) * np.log((pa + eps) / (pb + eps))))

def psi_by_feature(a, b, bins=10, eps=1e-9):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    q = np.quantile(a, np.linspace(0, 1, bins + 1))
    q = np.unique(q); q[0], q[-1] = -np.inf, np.inf
    for i in range(1, len(q)):
        if not (q[i] > q[i - 1]):
            q[i] = np.nextafter(q[i - 1], np.inf)
    ca, _ = np.histogram(a, bins=q)
    cb, _ = np.histogram(b, bins=q)
    pa, pb = ca / max(ca.sum(), 1), cb / max(cb.sum(), 1)
    return float(np.sum((pa - pb) * np.log((pa + eps) / (pb + eps))))

# -----------------------------
# BIN helpers (préfixe/suffixe)
# -----------------------------
def find_bin_columns(df: pd.DataFrame, tag: str) -> List[str]:
    cols = []
    for c in df.columns:
        if c.startswith(tag) or c.endswith(tag):
            cols.append(c)
    return sorted(cols)

def raw_name_from_bin(col: str, tag: str) -> str:
    if col.startswith(tag): return col[len(tag):]
    if col.endswith(tag):  return col[:-len(tag)]
    return col

def resolve_bin_col(df: pd.DataFrame, raw: str, tag: str) -> Optional[str]:
    cand_prefix = f"{tag}{raw}"
    cand_suffix = f"{raw}{tag}"
    if cand_prefix in df.columns: return cand_prefix
    if cand_suffix in df.columns: return cand_suffix
    return None

# -----------------------------
# WOE from BIN
# -----------------------------
def build_woe_maps_from_bins(
    df_enrichi: pd.DataFrame,
    target: str,
    raw_to_bin: Dict[str, str],
    smooth: float = 0.5
) -> Dict[str, Dict]:
    maps = {}
    y = df_enrichi[target].astype(int)
    B_all = float(y.sum())
    G_all = float(len(y) - y.sum())
    global_woe = float(np.log((B_all + smooth) / (G_all + smooth)))
    for raw, bcol in raw_to_bin.items():
        tab = df_enrichi.groupby(bcol, dropna=True)[target].agg(["sum", "count"])
        if tab.empty:
            maps[raw] = {"map": {}, "default": global_woe}
            continue
        tab["good"] = tab["count"] - tab["sum"]
        B = float(tab["sum"].sum())
        G = float(tab["good"].sum())
        K = max(int(len(tab)), 1)
        denom_bad  = (B + smooth * K) if (B + smooth * K) > 0 else 1.0
        denom_good = (G + smooth * K) if (G + smooth * K) > 0 else 1.0
        w = np.log(
            ((tab["sum"] + smooth) / denom_bad) /
            ((tab["good"] + smooth) / denom_good)
        ).replace([np.inf, -np.inf], np.nan)
        maps[raw] = {
            "map": {int(k) if pd.notna(k) else -9999: float(v) for k, v in w.items()},
            "default": global_woe
        }
    return maps

def apply_woe_with_maps(
    df_any: pd.DataFrame,
    maps: Dict[str, Dict],
    kept_vars_raw: List[str],
    bin_tag: str
) -> pd.DataFrame:
    cols = []
    for raw in kept_vars_raw:
        bcol = resolve_bin_col(df_any, raw, bin_tag)
        if bcol is None or raw not in maps:
            continue
        ser = df_any[bcol].astype("Int64")
        wmap = maps[raw]["map"]
        wdef = float(maps[raw]["default"])
        x = ser.map(wmap).astype(float).fillna(wdef)
        cols.append((f"{raw}_WOE", x))
    if not cols:
        return pd.DataFrame(index=df_any.index)
    return pd.concat([s for _, s in cols], axis=1)

# -----------------------------
# Sélection par anti-colinéarité
# -----------------------------
def select_woe_columns(X_woe: pd.DataFrame, order_hint: List[str], corr_thr: float = 0.85) -> List[str]:
    cols = [c for c in order_hint if c in X_woe.columns] or list(X_woe.columns)
    corr = X_woe[cols].corr().abs().fillna(0.0)
    selected = []
    for c in cols:
        if not selected:
            selected.append(c)
            continue
        mc = corr.loc[c, corr.columns.intersection(selected)]
        max_corr = float(mc.max()) if len(mc) else 0.0
        if not np.isfinite(max_corr) or np.isnan(max_corr):
            max_corr = 0.0
        if max_corr < corr_thr:
            selected.append(c)
    return selected

# -----------------------------
# Entraînement principal (sans ablation ni prior-shift)
# -----------------------------
def train_from_binned(
    df_tr: pd.DataFrame,
    df_va: pd.DataFrame,
    target: str,
    bin_suffix: str = "__BIN",
    corr_threshold: float = 0.85,
    cv_folds: int = 5,
    calibration: str = "none",            # "none", "sigmoid", "isotonic"
    # options PSI / proxies
    drop_proxy_cutoff: Optional[float] = None,
    conditional_proxies: Optional[List[str]] = None,
    timer: Optional[Timer] = None
) -> Dict[str, Any]:
    """
    Entraîne un modèle de PD en suivant :
      BIN -> WOE -> sélection anti-colinéarité -> LR (CV) -> calibration optionnelle.

    Pratique "bancaire" visée :
      - La grille de notation (master scale) est définie sur le SCORE LR brut
        (best_lr.decision_function), indépendant de la calibration.
      - La calibration (none/sigmoid/isotonic) sert à produire des PD, pas à définir les grades.

    La validation est utilisée uniquement pour :
      - mesurer la performance (AUC, Brier, logloss, KS, PSI train/val),
      - construire les déciles.
    """

    nullctx = contextlib.nullcontext()
    tctx = (lambda name: timer.section(name)) if timer else (lambda name: nullctx)

    # 1) Y
    with tctx("prep_y"):
        y_tr = df_tr[target].astype(int).values
        y_va = df_va[target].astype(int).values if target in df_va.columns else None

    # 2) BIN -> WOE
    with tctx("detect_bin_cols"):
        bin_cols = [c for c in df_tr.columns if c.endswith(bin_suffix) or c.startswith(bin_suffix)]
    if not bin_cols:
        raise SystemExit("Aucune colonne BIN détectée (__BIN en suffixe ou préfixe).")

    with tctx("woe_build"):
        raw_to_bin = {raw_name_from_bin(c, bin_suffix): c for c in bin_cols}
        keep_raw = sorted(raw_to_bin.keys())
        woe_maps = build_woe_maps_from_bins(df_tr, target, raw_to_bin, smooth=0.5)
        Xtr_woe_full = apply_woe_with_maps(df_tr, woe_maps, keep_raw, bin_tag=bin_suffix)
        Xva_woe_full = apply_woe_with_maps(df_va, woe_maps, keep_raw, bin_tag=bin_suffix) if y_va is not None else None
        if Xva_woe_full is not None:
            Xva_woe_full = Xva_woe_full.reindex(columns=Xtr_woe_full.columns, fill_value=0.0)

    computed_woe = True

    # 3) Drop proxies instables via PSI(feature) train / val (optionnel)
    if drop_proxy_cutoff is not None and conditional_proxies and Xva_woe_full is not None:
        with tctx("psi_drop"):
            to_drop = []
            for raw in conditional_proxies:
                c = f"{raw}_WOE"
                if c in Xtr_woe_full.columns and c in Xva_woe_full.columns:
                    v = psi_by_feature(Xtr_woe_full[c], Xva_woe_full[c])
                    if v is not None and np.isfinite(v) and v > float(drop_proxy_cutoff):
                        to_drop.append(c)
            if to_drop:
                Xtr_woe_full = Xtr_woe_full.drop(columns=to_drop, errors="ignore")
                Xva_woe_full = Xva_woe_full.drop(columns=to_drop, errors="ignore")

    # 4) Sélection anti-colinéarité (ordre = variance décroissante)
    with tctx("feature_selection"):
        order_hint = list(Xtr_woe_full.var().sort_values(ascending=False).index)
        kept_woe = select_woe_columns(Xtr_woe_full, order_hint, corr_thr=corr_threshold)
        Xtr = Xtr_woe_full[kept_woe].copy()
        Xva = Xva_woe_full[kept_woe].copy() if Xva_woe_full is not None else None

    # 5) GridSearch sur LR
    with tctx("gridsearch"):
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid = {
            # plage de C conservatrice pour limiter l’overfitting
            "C": [0.01, 0.1, 1.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
            "class_weight": [None],
            "max_iter": [2000],
        }
        base_lr = LogisticRegression(random_state=42)
        gs = GridSearchCV(base_lr, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True).fit(Xtr, y_tr)
        best_lr = gs.best_estimator_

    # 6) Scores LR (structurels, pour la grille de notation)
    with tctx("scores_lr"):
        score_tr = best_lr.decision_function(Xtr)
        score_va = best_lr.decision_function(Xva) if Xva is not None else None

    # 7) Calibration optionnelle (pour PD, pas pour la grille)
    with tctx("calibration_fit"):
        if calibration is None or calibration.lower() == "none":
            model = best_lr
        else:
            method = calibration.lower()
            if method not in ("sigmoid", "isotonic"):
                raise ValueError(f"Calibration inconnue: {calibration} (attendu: none, sigmoid, isotonic)")
            model = CalibratedClassifierCV(best_lr, method=method, cv=cv).fit(Xtr, y_tr)

    # 8) Metrics train / val (sur les probas du modèle final)
    with tctx("metrics_train"):
        p_tr = model.predict_proba(Xtr)[:, 1]
        metrics = {
            "train_auc": float(roc_auc_score(y_tr, p_tr)),
            "train_brier": float(brier_score_loss(y_tr, p_tr)),
            "train_logloss": float(log_loss(y_tr, p_tr)),
            "n_kept_features": int(len(kept_woe)),
            "calibration": calibration,
        }

    dec = pd.DataFrame()
    p_va = None

    if Xva is not None and y_va is not None:
        with tctx("metrics_val"):
            p_va = model.predict_proba(Xva)[:, 1]
            ks, thr = ks_best_threshold(y_va, p_va)
            metrics.update({
                "val_auc": float(roc_auc_score(y_va, p_va)),
                "val_brier": float(brier_score_loss(y_va, p_va)),
                "val_logloss": float(log_loss(y_va, p_va)),
                "val_ks": float(ks),
                "val_ks_threshold": float(thr),
                "psi_train_to_val_proba": float(psi(p_tr, p_va, bins=10))
            })
            dec = decile_table(y_va, p_va, q=10)

    # 9) Importance standardisée (sur best_lr)
    imp_df = None
    with tctx("importance"):
        lr_final = best_lr
        if hasattr(lr_final, "coef_"):
            beta = np.asarray(lr_final.coef_).ravel()
            feat_cols = list(Xtr.columns)
            if len(beta) != len(feat_cols):
                m = min(len(beta), len(feat_cols))
                beta = beta[:m]
                feat_cols = feat_cols[:m]
            stds = Xtr[feat_cols].std(ddof=0).replace(0, np.nan)
            std_coef = pd.Series(beta, index=feat_cols) * stds
            imp_df = (
                pd.DataFrame({"feature": feat_cols, "coef": beta, "std": stds.values, "std_coef": std_coef.values})
                .dropna(subset=["std_coef"])
                .sort_values("std_coef", key=lambda s: s.abs(), ascending=False)
            )

    out: Dict[str, Any] = {
        "model": model,                 # LR ou CalibratedClassifierCV (pour PD)
        "best_lr": lr_final,            # LogisticRegression brute (pour le score & la grille)
        "kept_woe": list(Xtr.columns),
        "computed_woe": bool(computed_woe),
        "woe_maps": woe_maps,
        "metrics": metrics,
        "deciles_val": dec,
        "importance": imp_df,
        # sorties supplémentaires
        "p_tr_final": p_tr,             # probas train (pour diagnostics)
        "p_va_final": p_va,             # probas val (pour diagnostics)
        "score_tr_final": score_tr,     # scores LR train (pour grille)
        "score_va_final": score_va,     # scores LR val (pour grille)
        "y_va": y_va,
    }
    return out

# -----------------------------
# Bucketing helpers (quantiles robustes)
# -----------------------------
def safe_quantile_edges(scores: np.ndarray, n: int = 10) -> np.ndarray:
    """
    Bornes strictement croissantes à partir des quantiles du score.
    Indépendant de l'échelle (score LR non borné, proba, etc.).
    """
    s = np.asarray(scores, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    qs = np.linspace(0.0, 1.0, n + 1)
    edges = np.quantile(s, qs, method="linear")
    for i in range(1, len(edges)):
        if not (edges[i] > edges[i - 1]):
            edges[i] = np.nextafter(edges[i - 1], np.inf)
    return edges

def assign_bucket(scores: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Retourne des buckets 1..K-1 en utilisant les bornes `edges` sur le score."""
    inner = edges[1:-1]
    return (np.digitize(scores, inner, right=False) + 1).astype(int)

def bucket_stats(scores: np.ndarray, y: Optional[np.ndarray], edges: np.ndarray) -> pd.DataFrame:
    """
    Statistiques par bucket sur l'échelle de score.
    - `score_mean` = moyenne du score dans le bucket.
    - `pd` = taux de défaut observé (si y fourni).
    """
    b = assign_bucket(scores, edges)
    dfb = pd.DataFrame({"bucket": b, "score": scores})
    if y is not None and len(y) == len(scores):
        dfb["y"] = y.astype(int)
        tab = (
            dfb.groupby("bucket", as_index=True)
               .agg(count=("y", "size"),
                    events=("y", "sum"),
                    score_mean=("score", "mean"))
               .sort_index()
        )
        tab["pd"] = tab["events"] / tab["count"].where(tab["count"] > 0, 1)
    else:
        tab = (
            dfb.groupby("bucket", as_index=True)
               .agg(count=("score", "size"),
                    score_mean=("score", "mean"))
               .sort_index()
        )
        tab["events"] = np.nan
        tab["pd"] = np.nan
    tab.reset_index(inplace=True)
    return tab

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        "Train depuis données binners (BIN → WOE) + LR (CV) + calibration optionnelle, sans ablation ni prior-shift."
    )
    p.add_argument("--train", required=True)
    p.add_argument("--validation", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--artifacts", default="artifacts/model_from_binned")
    p.add_argument("--bin-suffix", default="__BIN")     # tag (préfixe OU suffixe)
    p.add_argument("--corr-threshold", type=float, default=0.85)
    p.add_argument("--cv-folds", type=int, default=5)
    # calibration: none / sigmoid / isotonic
    p.add_argument("--calibration", default="none",
                   help="Méthode de calibration pour PD: none (défaut), sigmoid, isotonic.")
    # PSI / proxies (optionnel)
    p.add_argument("--drop-proxy-cutoff", type=float, default=None)
    p.add_argument("--conditional-proxies", default="")
    # timing
    p.add_argument("--timing", action="store_true", help="Sauve timings.json + résumé final en stdout")
    p.add_argument("--timing-live", action="store_true", help="Affiche le timing de chaque section en live")
    # bucketing
    p.add_argument("--n-buckets", type=int, default=10, help="Nombre de classes risque (quantiles)")
    p.add_argument(
        "--bucket-base",
        choices=["train", "validation", "train+validation"],
        default="validation",
        help="Échantillon utilisé pour construire la grille de buckets lorsqu'on ne réutilise pas une grille existante."
    )
    p.add_argument(
        "--risk-buckets-in",
        default=None,
        help="JSON existant avec les bornes de buckets (edges de score). Si fourni, on réutilise cette grille et on ne la recrée pas."
    )
    return p.parse_args()

def main():
    args = parse_args()
    artifacts = Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    timer = Timer(live=args.timing_live) if (args.timing or args.timing_live) else None
    nullctx = contextlib.nullcontext()
    tctx = (lambda name: timer.section(name)) if timer else (lambda name: nullctx)

    with tctx("load_data"):
        df_tr = load_any(args.train)
        df_va = load_any(args.validation)
        if args.target not in df_tr.columns or args.target not in df_va.columns:
            raise SystemExit(f"Target '{args.target}' absente de train/validation.")

    with tctx("parse_args"):
        cond_proxies = [s.strip() for s in args.conditional_proxies.split(",") if s.strip()] if args.conditional_proxies else None

    with tctx("train_from_binned"):
        out = train_from_binned(
            df_tr=df_tr, df_va=df_va, target=args.target,
            bin_suffix=args.bin_suffix,
            corr_threshold=args.corr_threshold, cv_folds=args.cv_folds,
            calibration=args.calibration,
            drop_proxy_cutoff=args.drop_proxy_cutoff, conditional_proxies=cond_proxies,
            timer=timer
        )

    # ---------------- Buckets (bornes + stats) ----------------
    with tctx("save_buckets"):
        score_tr_final = np.asarray(out.get("score_tr_final", []), float)
        score_va_final = out.get("score_va_final", None)

        # échantillons cibles pour PD observée
        y_tr = df_tr[args.target].astype(int).values
        y_va_arr = df_va[args.target].astype(int).values

        # 1) Si on réutilise une grille existante (mode "banque" : grille gelée)
        if args.risk_buckets_in:
            cfg = json.loads(Path(args.risk_buckets_in).read_text(encoding="utf-8"))
            edges = np.asarray(cfg["edges"], float)

            # On calcule les stats sur la validation (par défaut) avec cette grille
            if score_va_final is not None:
                base_scores = np.asarray(score_va_final, float)
                base_y = y_va_arr
            else:
                base_scores = score_tr_final
                base_y = y_tr

            stats_df = bucket_stats(
                scores=base_scores,
                y=base_y,
                edges=edges
            )

            # On NE recrée PAS la grille : on la recopie juste dans les artefacts
            save_json({"edges": edges.tolist()}, artifacts / "risk_buckets.json")
            save_json({
                "n_buckets": int(len(edges) - 1),
                "edges": edges.tolist(),
                "by_bucket": stats_df.to_dict(orient="records")
            }, artifacts / "bucket_stats.json")
            print(f"✔ Buckets réutilisés depuis {args.risk_buckets_in} → {artifacts/'risk_buckets.json'}")
            print(f"✔ Stats buckets (grille existante, sur score LR) → {artifacts/'bucket_stats.json'}")

        # 2) Sinon, on construit la master scale une fois (mode "construction de grille")
        else:
            # Construction de la base pour les quantiles (sur score LR)
            if args.bucket_base == "train":
                base_scores = score_tr_final
                base_y = y_tr
            elif args.bucket_base == "train+validation" and score_va_final is not None:
                base_scores = np.concatenate([score_tr_final, np.asarray(score_va_final, float)])
                base_y = np.concatenate([y_tr, y_va_arr])
            else:  # "validation" par défaut ou si score_va_final dispo
                if score_va_final is not None:
                    base_scores = np.asarray(score_va_final, float)
                    base_y = y_va_arr
                else:
                    base_scores = score_tr_final
                    base_y = y_tr

            edges = safe_quantile_edges(base_scores, n=args.n_buckets)

            # Stats sur la validation si dispo, sinon sur la base (train ou train+val)
            if score_va_final is not None and args.bucket_base != "train":
                stats_scores = np.asarray(score_va_final, float)
                stats_y = y_va_arr
            else:
                stats_scores = base_scores
                stats_y = base_y

            stats_df = bucket_stats(
                scores=stats_scores,
                y=stats_y,
                edges=edges
            )

            save_json({"edges": edges.tolist()}, artifacts / "risk_buckets.json")
            save_json({
                "n_buckets": int(len(edges) - 1),
                "edges": edges.tolist(),
                "by_bucket": stats_df.to_dict(orient="records")
            }, artifacts / "bucket_stats.json")
            print(f"✔ Buckets construits (base={args.bucket_base}, sur score LR) → {artifacts/'risk_buckets.json'}")
            print(f"✔ Stats buckets (échantillon dev, sur score LR) → {artifacts/'bucket_stats.json'}")

    with tctx("save_artifacts"):
        dump({
            "model": out["model"],           # LR ou CalibratedClassifierCV (pour PD)
            "best_lr": out["best_lr"],       # LogisticRegression brute (pour score & grille)
            "kept_woe": out["kept_woe"],
            "computed_woe": out["computed_woe"],
            "woe_maps": out["woe_maps"],
            "target": args.target
        }, artifacts / "model_best.joblib")

        pd.DataFrame([out["metrics"]]).to_csv(artifacts / "reports.csv", index=False)

        save_json({
            "train_path": str(Path(args.train).resolve()),
            "validation_path": str(Path(args.validation).resolve()),
            "artifacts_dir": str(artifacts.resolve()),
            "target": args.target,
            "bin_suffix": args.bin_suffix,
            "cv_folds": int(args.cv_folds),
            "corr_threshold": float(args.corr_threshold),
            "calibration": args.calibration,
            "kept_woe": out["kept_woe"],
            "computed_woe": bool(out["computed_woe"]),
            "drop_proxy_cutoff": args.drop_proxy_cutoff,
            "conditional_proxies": cond_proxies,
            "metrics": out["metrics"],
            "n_buckets": int(args.n_buckets),
            "bucket_base": args.bucket_base,
            "risk_buckets_in": args.risk_buckets_in,
        }, artifacts / "meta.json")

        if isinstance(out["deciles_val"], pd.DataFrame) and not out["deciles_val"].empty:
            out["deciles_val"].to_csv(artifacts / "deciles_val.csv", index=False)
        if isinstance(out["importance"], pd.DataFrame) and not out["importance"].empty:
            out["importance"].to_csv(artifacts / "importance.csv", index=False)

    # Affichage + export timings
    if timer:
        total = sum(timer.records.values())
        print("\n⏱ Timings (s):")
        for k, v in sorted(timer.records.items(), key=lambda kv: -kv[1]):
            pct = (v / total * 100.0) if total > 0 else 0.0
            print(f"  - {k:22s} {v:8.3f}s  ({pct:5.1f}%)")
        save_json({"timings_seconds": timer.records, "total_seconds": total}, artifacts / "timings.json")

    print("✔ Saved:", artifacts / "model_best.joblib")
    print("✔ Saved:", artifacts / "reports.csv")
    print("✔ Saved:", artifacts / "meta.json")
    if (artifacts / "deciles_val.csv").exists():
        print("✔ Saved:", artifacts / "deciles_val.csv")
    if (artifacts / "importance.csv").exists():
        print("✔ Saved:", artifacts / "importance.csv")
    if (artifacts / "risk_buckets.json").exists():
        print("✔ Saved:", artifacts / "risk_buckets.json")
    if (artifacts / "bucket_stats.json").exists():
        print("✔ Saved:", artifacts / "bucket_stats.json")

if __name__ == "__main__":
    main()
















Maintenant dans src/features :

binning.py :

# src/features/binning.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import json, math, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Parallélisation
try:
    from joblib import Parallel, delayed
except Exception:  # fallback si joblib indisponible
    Parallel = None
    def delayed(f): return f  # type: ignore

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*is_period_dtype is deprecated.*")

# =============================
# Réglages "pipeline.py"-like
# =============================
DENYLIST_STRICT_DEFAULT = [
    "first_payment_date",       # proxy temporel
    "maturity_date",            # proxy temporel redondant
    "vintage",                  # proxy temporel
    "mi_cancellation_indicator" # post-événement → fuite
]

EXCLUDE_IDS_DEFAULT: Tuple[str, ...] = (
    "loan_sequence_number", "postal_code", "seller_name", "servicer_name", "msa_md"
)

# -----------------------------
# Utilitaires généraux
# -----------------------------
def gini_trapz(df_cum, y_col="bad_client_share_cumsum", x_col="good_client_share_cumsum", signed=False):
    df = df_cum[[x_col, y_col]].astype(float).copy().sort_values(x_col)
    df[x_col] = df[x_col].clip(0, 1)
    df[y_col] = df[y_col].clip(0, 1)
    if df[x_col].iloc[0] > 0 or df[y_col].iloc[0] > 0:
        df = pd.concat([pd.DataFrame({x_col: [0.0], y_col: [0.0]}), df], ignore_index=True)
    if df[x_col].iloc[-1] < 1 - 1e-12 or df[y_col].iloc[-1] < 1 - 1e-12:
        df = pd.concat([df, pd.DataFrame({x_col: [1.0], y_col: [1.0]})], ignore_index=True)
    area = np.trapezoid(df[y_col].to_numpy(), df[x_col].to_numpy()) if hasattr(np, "trapezoid") else np.trapz(df[y_col].to_numpy(), df[x_col].to_numpy())
    g = 1 - 2 * area
    return g if signed else abs(g)

def _is_period_dtype(dt):  # compat
    try:
        return pd.api.types.is_period_dtype(dt)
    except Exception:
        return False

def to_float_series(s: pd.Series) -> pd.Series:
    if _is_period_dtype(s.dtype):
        ts = s.dt.to_timestamp(how="start")
        days = (ts.astype("int64") // 86_400_000_000_000)
        return days.astype("float64")
    if pd.api.types.is_datetime64_any_dtype(s):
        s_dt = s
        try:
            if getattr(s_dt.dt, "tz", None) is not None:
                s_dt = s_dt.dt.tz_convert(None)
        except Exception:
            pass
        s_dt = s_dt.astype("datetime64[ns]")
        days = (s_dt.astype("int64") // 86_400_000_000_000)
        return days.astype("float64")
    return pd.to_numeric(s, errors="coerce").astype("float64")

# -----------------------------
# Helpers pré-traitements
# -----------------------------
def drop_missing_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    mask = cols.str.startswith("was_missing_") | cols.str.endswith("_missing")
    to_drop = cols[mask].tolist()
    if to_drop:
        return df.drop(columns=to_drop, errors="ignore")
    return df

def apply_denylist(df: pd.DataFrame, denylist: List[str]) -> pd.DataFrame:
    if not denylist:
        return df
    return df.drop(columns=[c for c in denylist if c in df.columns], errors="ignore")

# -----------------------------
# Dé-one-hot + denylist
# -----------------------------
def detect_onehot_groups(df, exclude_cols=None, exclusivity_thr=0.95):
    exclude = set(exclude_cols or [])
    groups = {}
    for c in df.columns:
        if c in exclude or "_" not in c:
            continue
        base, lab = c.rsplit("_", 1)
        s = df[c]
        is_ohe = (pd.api.types.is_bool_dtype(s) or (pd.api.types.is_numeric_dtype(s) and s.dropna().isin([0, 1]).all()))
        if is_ohe:
            groups.setdefault(base, []).append((c, lab))
    clean = {}
    for base, items in groups.items():
        cols = [c for c, _ in items]
        vals = df[cols].apply(pd.to_numeric, errors="coerce")
        row_sum = vals.fillna(0).astype("Int64").sum(axis=1)
        excl_rate = float(((row_sum <= 1) | row_sum.isna()).mean())
        if excl_rate >= exclusivity_thr:
            clean[base] = items
    return clean

def deonehot(df, exclude_cols=None, ambiguous_label=None):
    groups = detect_onehot_groups(df, exclude_cols=exclude_cols)
    out = df.copy()
    for base, items in groups.items():
        cols = [c for c, _ in items]
        labels = [lab for _, lab in items]
        gvals = df[cols].apply(pd.to_numeric, errors="coerce")
        row_sum = gvals.fillna(0).astype("Int64").sum(axis=1)
        ser = pd.Series(pd.NA, index=df.index, dtype="object")
        for c, lab in zip(cols, labels):
            ser[df[c] == 1] = (pd.NA if lab == "<NA>" else lab)
        amb = row_sum > 1
        if amb.any():
            ser[amb] = ambiguous_label if ambiguous_label is not None else pd.NA
        out[base] = ser.astype("category")
        out.drop(columns=cols, inplace=True, errors="ignore")
    return out

# -----------------------------
# Catégorielles : max |Gini| par fusion
# -----------------------------
def _cat_stats(df, col, target_col, include_missing=True, missing_label="__MISSING__"):
    y = df[target_col].astype(int)
    s = df[col]
    if include_missing:
        s = s.astype("object").where(s.notna(), missing_label)
    tmp = pd.DataFrame({col: s, target_col: y})
    agg = tmp.groupby(col, dropna=not include_missing)[target_col].agg(["sum", "count"])
    agg.rename(columns={"sum": "n_bad", "count": "n_total"}, inplace=True)
    agg["n_good"] = agg["n_total"] - agg["n_bad"]
    n_bad = int(y.sum())
    n_good = int(len(y) - y.sum())
    denom_bad = n_bad if n_bad > 0 else 1
    denom_good = n_good if n_good > 0 else 1
    agg["bad_rate"] = agg["n_bad"] / agg["n_total"].where(agg["n_total"] > 0, 1)
    agg["bad_share"] = agg["n_bad"] / denom_bad
    agg["good_share"] = agg["n_good"] / denom_good
    return agg.reset_index().rename(columns={col: "modality"})

def _groups_df_from_bins(stats_df, bins):
    rows = []
    for i, mods in enumerate(bins):
        sub = stats_df[stats_df["modality"].isin(mods)]
        n_bad = int(sub["n_bad"].sum())
        n_good = int(sub["n_good"].sum())
        n_tot = int(sub["n_total"].sum())
        br = n_bad / n_tot if n_tot > 0 else 0.0
        rows.append(
            {
                "bin_id": i,
                "modalities": tuple(mods),
                "n_total": n_tot,
                "n_bad": n_bad,
                "n_good": n_good,
                "bad_rate": br,
                "bad_share": sub["bad_share"].sum(),
                "good_share": sub["good_share"].sum(),
            }
        )
    gdf = pd.DataFrame(rows).sort_values("bad_rate", ascending=True, kind="mergesort").reset_index(drop=True)
    gdf["bad_cum"] = gdf["bad_share"].cumsum()
    gdf["good_cum"] = gdf["good_share"].cumsum()
    return gdf

def _gini_from_bins(stats_df, bins):
    gdf = _groups_df_from_bins(stats_df, bins)
    df_cum = gdf.rename(
        columns={"good_cum": "good_client_share_cumsum", "bad_cum": "bad_client_share_cumsum"}
    )[["good_client_share_cumsum", "bad_client_share_cumsum"]]
    return gini_trapz(df_cum)

def maximize_gini_categorical(
    df,
    col,
    target_col,
    include_missing=True,
    missing_label="__MISSING__",
    max_bins=6,
    min_bin_size=200,
    min_bin_frac=None,
    ordered=False,
    explicit_order=None,
):
    stats_df = _cat_stats(df, col, target_col, include_missing, missing_label)
    if ordered and explicit_order is not None:
        order = [m for m in explicit_order if m in set(stats_df["modality"])]
        order += [m for m in stats_df["modality"] if m not in set(order)]
    else:
        order = list(stats_df.sort_values("bad_rate")["modality"])
    groups = [[m] for m in order]
    if len(groups) <= 1:
        mapping = {m: 0 for m in order}
        g = _gini_from_bins(stats_df, groups)
        return {
            "mapping": mapping,
            "gini_before": float(g),
            "gini_after": float(g),
            "bins": [tuple(grp) for grp in groups],
        }

    n_total = int(stats_df["n_total"].sum())
    need = 0
    if min_bin_frac is not None:
        need = max(need, math.ceil(float(min_bin_frac) * max(n_total, 1)))
    if min_bin_size is not None:
        need = max(need, int(min_bin_size))

    def ok(gs):
        if max_bins is not None and len(gs) > max_bins:
            return False
        if need:
            for mods in gs:
                if int(stats_df[stats_df["modality"].isin(mods)]["n_total"].sum()) < need:
                    return False
        return True

    def reorder(gs):
        if ordered:
            return gs

        def br(mods):
            sub = stats_df[stats_df["modality"].isin(mods)]
            nb, nt = sub["n_bad"].sum(), sub["n_total"].sum()
            return (nb / nt) if nt > 0 else 0.0

        return sorted(gs, key=br)

    while not ok(groups):
        best_g, best_i = -np.inf, None
        for i in range(len(groups) - 1):
            merged = groups[:i] + [groups[i] + groups[i + 1]] + groups[i + 2 :]
            merged = reorder(merged)
            g_try = _gini_from_bins(stats_df, merged)
            if g_try > best_g:
                best_g, best_i = g_try, i
        if best_i is None:
            best_i = 0
        groups = groups[:best_i] + [groups[best_i] + groups[best_i + 1]] + groups[best_i + 2 :]
        groups = reorder(groups)

    g_before = _gini_from_bins(stats_df, [[m] for m in order])
    g_after = _gini_from_bins(stats_df, groups)
    bins = [tuple(mods) for mods in groups]
    mapping = {m: i for i, mods in enumerate(bins) for m in mods}
    return {"mapping": mapping, "gini_before": float(g_before), "gini_after": float(g_after), "bins": bins}

# -----------------------------
# Numériques : seuils quantiles (candidats) pour max |Gini|
# -----------------------------
def _safe_edges_for_cut(edges, s_float):
    e = np.array(edges, dtype="float64")
    for i in range(1, len(e)):
        if not (e[i] > e[i - 1]):
            e[i] = np.nextafter(e[i - 1], np.inf)
    arr = s_float.to_numpy()
    if arr.size == 0:
        return e
    s_min = float(np.nanmin(arr)) if np.isfinite(arr).any() else -1.0
    s_max = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    if len(e) >= 2:
        e[0] = min(e[1] - 1e-6 * (abs(e[1]) + 1.0), s_min - 1e-6 * (abs(e[1]) + 1.0))
        e[-1] = max(e[-2] + 1e-6 * (abs(e[-2]) + 1.0), s_max + 1e-6 * (abs(e[-2]) + 1.0))
    return e

def _gini_from_numeric_bins(y_int, x_float, edges, include_missing=True):
    y = y_int.astype(int).to_numpy()
    x = x_float.to_numpy()
    K = len(edges) - 1
    idx = np.digitize(x, edges[1:-1], right=True)
    n_bad = int(y.sum())
    n_good = int(len(y) - y.sum())
    denom_bad = n_bad if n_bad > 0 else 1
    denom_good = n_good if n_good > 0 else 1
    rows = []
    for k in range(K):
        m = (idx == k) & ~np.isnan(x)
        nk = int(m.sum())
        nb = int(y[m].sum())
        ng = nk - nb
        br = nb / nk if nk > 0 else 0.0
        rows.append({"bin": k, "n_total": nk, "n_bad": nb, "n_good": ng, "bad_rate": br})
    if include_missing and np.isnan(x).any():
        m = np.isnan(x)
        nk = int(m.sum())
        nb = int(y[m].sum())
        ng = nk - nb
        br = nb / nk if nk > 0 else 0.0
        rows.append({"bin": K, "n_total": nk, "n_bad": nb, "n_good": ng, "bad_rate": br})
    gdf = pd.DataFrame(rows)
    if gdf.empty:
        return 0.0, gdf
    gdf["bad_share"] = gdf["n_bad"] / denom_bad
    gdf["good_share"] = gdf["n_good"] / denom_good
    gdf = gdf.sort_values("bad_rate").reset_index(drop=True)
    gdf["bad_cum"] = gdf["bad_share"].cumsum()
    gdf["good_cum"] = gdf["good_share"].cumsum()
    df_cum = gdf.rename(
        columns={"good_cum": "good_client_share_cumsum", "bad_cum": "bad_client_share_cumsum"}
    )[["good_client_share_cumsum", "bad_client_share_cumsum"]]
    return gini_trapz(df_cum), gdf

def maximize_gini_numeric(
    df,
    col,
    target_col,
    max_bins=6,
    min_bin_size=200,
    min_bin_frac=None,
    n_quantiles=50,
    q_low=0.02,
    q_high=0.98,
    include_missing=True,
    min_gain=1e-5,
):
    s = to_float_series(df[col])
    y = df[target_col].astype(int)
    # Cas dégénéré
    if s.dropna().nunique() < 2:
        s_f = s[np.isfinite(s)]
        if s_f.empty:
            e_cut = np.array([-1.0, 1.0], dtype="float64")
        else:
            lo, hi = float(np.nanmin(s_f)), float(np.nanmax(s_f))
            eps = 1e-6 * (abs(lo) + abs(hi) + 1.0)
            e_cut = np.array([lo - eps, hi + eps], dtype="float64")
        g0, _ = _gini_from_numeric_bins(y, s, [-np.inf, np.inf], include_missing)
        return {"edges": [-np.inf, np.inf], "edges_for_cut": e_cut, "gini_before": float(g0), "gini_after": float(g0)}

    qs = np.linspace(q_low, q_high, n_quantiles)
    cand_vals = np.unique(s.quantile(qs).dropna().values)
    edges = [-np.inf, np.inf]

    n = len(s)
    need = 0
    if min_bin_frac is not None:
        need = max(need, math.ceil(float(min_bin_frac) * max(n, 1)))
    if min_bin_size is not None:
        need = max(need, int(min_bin_size))

    def edges_ok(e):
        arr = s.to_numpy()
        idx = np.digitize(arr, e[1:-1], right=True)
        for k in range(len(e) - 1):
            if int(((idx == k) & ~np.isnan(arr)).sum()) < need:
                return False
        return True

    # baseline correcte
    g0, _ = _gini_from_numeric_bins(y, s, edges, include_missing)
    best_g = g0

    improved = True
    while improved and (len(edges) - 1) < max_bins:
        improved = False
        best_gain = min_gain
        best_t = None
        g_best = best_g
        for t in cand_vals:
            if t in edges:
                continue
            new_e = sorted([*edges, t])
            if any(np.isclose(new_e[i], new_e[i + 1]) for i in range(len(new_e) - 1)):
                continue
            if not edges_ok(new_e):
                continue
            g_try, _ = _gini_from_numeric_bins(y, s, new_e, include_missing)
            gain = g_try - best_g
            if gain > best_gain:
                best_gain, best_t, g_best = gain, t, g_try
        if best_t is not None:
            edges = sorted([*edges, best_t])
            best_g = g_best
            improved = True

    g_after, _ = _gini_from_numeric_bins(y, s, edges, include_missing)
    e = sorted(edges)
    e_cut = _safe_edges_for_cut(e, s)
    return {"edges": e, "edges_for_cut": e_cut, "gini_before": float(g0), "gini_after": float(g_after)}

# -----------------------------
# Pipeline binning complet (train) + transform (val/test)
# -----------------------------
@dataclass
class LearnedBins:
    target: str
    include_missing: bool
    missing_label: str
    bin_col_suffix: str
    cat_results: Dict[str, dict]
    num_results: Dict[str, dict]

# --- workers parallélisés
def _compute_cat_result(df_small, col, target_col, include_missing, missing_label,
                        max_bins_categ, min_bin_size_categ, min_bin_frac_categ):
    res = maximize_gini_categorical(
        df_small, col, target_col,
        include_missing=include_missing, missing_label=missing_label,
        max_bins=max_bins_categ, min_bin_size=min_bin_size_categ, min_bin_frac=min_bin_frac_categ
    )
    return col, res

def _compute_num_result(df_small, col, target_col, max_bins_num, min_bin_size_num, min_bin_frac_num,
                        n_quantiles_num, include_missing):
    res = maximize_gini_numeric(
        df_small, col, target_col,
        max_bins=max_bins_num, min_bin_size=min_bin_size_num, min_bin_frac=min_bin_frac_num,
        n_quantiles=n_quantiles_num, include_missing=include_missing
    )
    return col, res

def run_binning_maxgini_on_df(
    df: pd.DataFrame,
    target_col: str,
    include_missing: bool = True,
    missing_label: str = "__MISSING__",
    max_bins_categ: int = 6,
    min_bin_size_categ: int = 200,
    min_bin_frac_categ: Optional[float] = None,
    max_bins_num: int = 6,
    min_bin_size_num: int = 200,
    min_bin_frac_num: Optional[float] = None,
    n_quantiles_num: int = 50,
    bin_col_suffix: str = "__BIN",
    exclude_ids: Tuple[str, ...] = EXCLUDE_IDS_DEFAULT,
    min_gini_keep: Optional[float] = None,
    # nouveautés
    denylist_strict: Optional[List[str]] = None,
    drop_missing_flags: bool = False,
    n_jobs_categ: int = 1,
    n_jobs_num: int = 1,
):
    # pré-traitements "pipeline.py"-like
    DF = df.copy()
    if drop_missing_flags:
        DF = drop_missing_flag_columns(DF)
    if denylist_strict:
        DF = apply_denylist(DF, denylist_strict)

    DF = deonehot(DF, exclude_cols=[target_col])
    target = target_col

    # détecte les catégorielles raisonnables (en appliquant exclude_ids aux objets aussi)
    exclude_ids_set = set(exclude_ids or [])
    cat_cols = []
    for c in DF.columns:
        if c == target or c in exclude_ids_set:
            continue
        s = DF[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s):
            cat_cols.append(c)
        elif (pd.api.types.is_object_dtype(s) or str(s.dtype).startswith("string")) and s.nunique(dropna=True) <= 50:
            cat_cols.append(c)
        elif (pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 8 and not any(k in c.lower() for k in ["id", "sequence"])):
            cat_cols.append(c)

    # numériques (mêmes exclusions — IDs, binaires, petits entiers, clés commerciales)
    num_cols = []
    for c in DF.columns:
        if c in (cat_cols + [target]) or c in exclude_ids_set:
            continue
        s = DF[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if s.dropna().isin([0, 1]).all():
                continue
            if pd.api.types.is_integer_dtype(s) and s.dropna().nunique() <= 8:
                continue
            if any(k in c.lower() for k in ["id", "sequence", "postal", "zip", "msa", "code", "seller", "servicer"]):
                continue
            num_cols.append(c)
        elif _is_period_dtype(s.dtype) or pd.api.types.is_datetime64_any_dtype(s):
            num_cols.append(c)

    # --- binning catégoriel (parallélisé)
    cat_results = {}
    if cat_cols:
        if n_jobs_categ != 1 and Parallel is not None:
            tasks = (
                delayed(_compute_cat_result)(
                    DF[[c, target]].copy(), c, target, include_missing, missing_label,
                    max_bins_categ, min_bin_size_categ, min_bin_frac_categ
                )
                for c in cat_cols
            )
            out = Parallel(n_jobs=n_jobs_categ, backend="loky", verbose=0)(list(tasks))
            for c, res in out:
                cat_results[c] = res
        else:
            for c in cat_cols:
                res = maximize_gini_categorical(
                    DF[[c, target]].copy(), c, target,
                    include_missing=include_missing, missing_label=missing_label,
                    max_bins=max_bins_categ, min_bin_size=min_bin_size_categ, min_bin_frac=min_bin_frac_categ
                )
                cat_results[c] = res

    # --- binning numérique (parallélisé)
    num_results = {}
    if num_cols:
        if n_jobs_num != 1 and Parallel is not None:
            tasks = (
                delayed(_compute_num_result)(
                    DF[[c, target]].copy(), c, target,
                    max_bins_num, min_bin_size_num, min_bin_frac_num, n_quantiles_num, include_missing
                )
                for c in num_cols
            )
            out = Parallel(n_jobs=n_jobs_num, backend="loky", verbose=0)(list(tasks))
            for c, res in out:
                num_results[c] = res
        else:
            for c in num_cols:
                res = maximize_gini_numeric(
                    DF[[c, target]].copy(), c, target,
                    max_bins=max_bins_num, min_bin_size=min_bin_size_num, min_bin_frac=min_bin_frac_num,
                    n_quantiles=n_quantiles_num, include_missing=include_missing
                )
                num_results[c] = res

    # Ajoute colonnes __BIN
    enriched = DF.copy()
    for c, r in cat_results.items():
        s = enriched[c].astype("object").where(enriched[c].notna(), missing_label)
        enriched[c + bin_col_suffix] = s.map(r["mapping"]).astype("Int64")
    for c, r in num_results.items():
        s = to_float_series(enriched[c])
        e = np.array(r["edges_for_cut"], dtype="float64")
        b = pd.cut(s, bins=e, include_lowest=True, duplicates="drop").cat.codes.astype("Int64")
        if include_missing and s.isna().any():
            b = b.where(~s.isna(), -1).astype("Int64")
        enriched[c + bin_col_suffix] = b

    # option min_gini_keep → filtre effectif des colonnes sorties
    keep_vars: Optional[set] = None
    if min_gini_keep is not None:
        rows = []
        for v, info in cat_results.items():
            rows.append((v, info["gini_after"]))
        for v, info in num_results.items():
            rows.append((v, info["gini_after"]))
        summary = pd.DataFrame(rows, columns=["variable", "gini_after"])
        keep_vars = set(summary.loc[summary["gini_after"] >= float(min_gini_keep), "variable"].tolist())

    if keep_vars is None:
        keep_vars = set(list(cat_results.keys()) + list(num_results.keys()))

    bin_cols = [v + bin_col_suffix for v in keep_vars if v + bin_col_suffix in enriched.columns]
    # on ne garde que target + bins sélectionnées
    kept = bin_cols + ([target] if target in enriched.columns else [])
    df_binned = enriched[kept].copy()

    learned = LearnedBins(
        target=target,
        include_missing=include_missing,
        missing_label=missing_label,
        bin_col_suffix=bin_col_suffix,
        cat_results=cat_results,
        num_results=num_results,
    )
    return learned, enriched, df_binned

def transform_with_learned_bins(df, learned: LearnedBins) -> pd.DataFrame:
    DF = deonehot(df, exclude_cols=[learned.target] if learned.target in df.columns else None)
    suffix = learned.bin_col_suffix

    # applique bins catégoriels
    for c, r in learned.cat_results.items():
        if c not in DF.columns:
            continue
        s = DF[c].astype("object").where(DF[c].notna(), learned.missing_label)
        DF[c + suffix] = s.map(r["mapping"]).astype("Int64").fillna(-2).astype("Int64")

    # applique bins numériques
    for c, r in learned.num_results.items():
        if c not in DF.columns:
            continue
        s = to_float_series(DF[c])
        e = np.array(r["edges_for_cut"], dtype="float64")
        b = pd.cut(s, bins=e, include_lowest=True, duplicates="drop").cat.codes.astype("Int64")
        if learned.include_missing and s.isna().any():
            b = b.where(~s.isna(), -1).astype("Int64")
        DF[c + suffix] = b

    # sort un DF modèle: uniquement __BIN (+ cible si présente)
    bin_cols = [c for c in DF.columns if c.endswith(suffix)]
    keep = bin_cols + ([learned.target] if learned.target in DF.columns else [])
    model_df = DF[keep].copy()
    return model_df

# -----------------------------
# Sérialisation (robuste JSON)
# -----------------------------
def _json_default(o):
    """Convertit proprement les objets numpy/pandas pour JSON."""
    import numpy as _np
    import pandas as _pd
    if isinstance(o, (_np.floating, _np.integer)):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, _pd.Interval):
        try:
            return [float(o.left), float(o.right)]
        except Exception:
            return [o.left, o.right]
    return str(o)

def save_bins_json(learned: LearnedBins, path: str):
    d = {
        "target": learned.target,
        "include_missing": bool(learned.include_missing),
        "missing_label": str(learned.missing_label),
        "bin_col_suffix": str(learned.bin_col_suffix),
        "cat_results": learned.cat_results,
        "num_results": learned.num_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, default=_json_default)

def load_bins_json(path: str) -> LearnedBins:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return LearnedBins(
        target=d["target"],
        include_missing=bool(d["include_missing"]),
        missing_label=str(d["missing_label"]),
        bin_col_suffix=str(d["bin_col_suffix"]),
        cat_results=d["cat_results"],
        num_results=d["num_results"],
    )



impute.py :
import numpy as np
import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_float_dtype,
)
from sklearn.base import BaseEstimator, TransformerMixin


class DataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, use_cohort=True, missing_flag=False, ltv_bins=(0, 80, 90, 95, 100, np.inf)):
        self.use_cohort = use_cohort
        self.ltv_bins = ltv_bins
        self.missing_flag = missing_flag

    # ---------- helpers ----------
    @staticmethod
    def _mode(x):
        try:
            return x.mode(dropna=True).iloc[0]
        except Exception:
            return np.nan

    @staticmethod
    def _to_year(series):
        """
        Retourne une Series d'années (Int16) de façon déterministe et rapide.
        Gère:
        - Period (M/Q/…)        -> .dt.year
        - datetime64[ns]        -> .dt.year
        - Chaînes: 'YYYYQn', 'YYYY-MM', 'YYYYMM', 'YYYY...' -> 4 premiers chiffres
        Si rien n'est interprétable, retourne None (désactive les stats 'by year').
        """
        try:
            s = pd.Series(series)
        except Exception:
            return None

        # Period -> year
        if pd.api.types.is_period_dtype(s):
            return s.dt.year.astype("Int16")

        # Datetime-like -> year
        if pd.api.types.is_datetime64_any_dtype(s):
            return pd.to_datetime(s, errors="coerce").dt.year.astype("Int16")

        # Catégorique/objet/chaine -> on extrait les 4 premiers chiffres (YYYY)
        ss = s.astype("string").str.strip()
        year_str = ss.str.extract(r"^\s*(\d{4})")[0]
        year = pd.to_numeric(year_str, errors="coerce").astype("Int16")

        # Si tout est NA, rien d'exploitable -> désactive la logique "by year"
        if year.isna().all():
            return None
        return year

    def _map_fill(self, keys_tuple_series, mapping):
        """keys_tuple_series = Series of tuples; mapping = dict {tuple: value}"""
        if mapping is None:
            return pd.Series(np.nan, index=keys_tuple_series.index)
        return keys_tuple_series.map(mapping)

    # ---------- fit on TRAIN ONLY ----------
    def fit(self, X, y=None):
        df = X.copy()

        # Colonnes de référence (exclure la colonne drop future)
        self.columns_fit_ = [c for c in df.columns if c != 'pre_relief_refi_loan_seq_number']
        self.dtypes_ = df.dtypes.to_dict()

        # Precompute year if present
        vyear = self._to_year(df['vintage']) if 'vintage' in df.columns else None

        # STORAGE
        self.stats_ = {}

        # CREDIT SCORE medians
        if 'credit_score' in df.columns:
            cs = pd.to_numeric(df['credit_score'], errors='coerce').clip(300, 850)
            self.stats_['credit_score_global'] = float(cs.median())
            self.stats_['credit_score_by_lp'] = None
            self.stats_['credit_score_by_year_lp'] = None

            if self.use_cohort and 'loan_purpose' in df.columns:
                med_lp = cs.groupby(df['loan_purpose'], observed=True).median()
                self.stats_['credit_score_by_lp'] = med_lp.to_dict()

                if vyear is not None:
                    med_y_lp = pd.Series(
                        cs.values,
                        index=pd.MultiIndex.from_arrays([vyear, df['loan_purpose']])
                    ).groupby(level=[0, 1], observed=True).median()
                    self.stats_['credit_score_by_year_lp'] = {k: float(v) for k, v in med_y_lp.items()}

        # MI% medians by LTV bins (and year)
        if 'mi_percent' in df.columns and 'original_ltv' in df.columns:
            mi = pd.to_numeric(df['mi_percent'], errors='coerce')
            ltv = pd.to_numeric(df['original_ltv'], errors='coerce').clip(lower=0)
            ltv_bins = pd.cut(ltv, self.ltv_bins, include_lowest=True, right=True)

            self.stats_['mi_by_bin'] = mi.groupby(ltv_bins, observed=True).median().to_dict()
            self.stats_['mi_by_year_bin'] = None
            if self.use_cohort and vyear is not None:
                idx = pd.MultiIndex.from_arrays([vyear, ltv_bins])
                med = pd.Series(mi.values, index=idx).groupby(level=[0, 1], observed=True).median()
                self.stats_['mi_by_year_bin'] = {k: float(v) for k, v in med.items()}

        # DTI medians
        if 'original_dti' in df.columns:
            dti = pd.to_numeric(df['original_dti'], errors='coerce')
            self.stats_['dti_global'] = float(dti.median())
            self.stats_['dti_by_lp'] = None
            self.stats_['dti_by_year_lp'] = None

            if self.use_cohort and 'loan_purpose' in df.columns:
                self.stats_['dti_by_lp'] = dti.groupby(df['loan_purpose'], observed=True).median().to_dict()
                if vyear is not None:
                    med = pd.Series(
                        dti.values,
                        index=pd.MultiIndex.from_arrays([vyear, df['loan_purpose']])
                    ).groupby(level=[0, 1], observed=True).median()
                    self.stats_['dti_by_year_lp'] = {k: float(v) for k, v in med.items()}

        # CLTV medians by year (fallback global)
        if 'original_cltv' in df.columns:
            cltv = pd.to_numeric(df['original_cltv'], errors='coerce')
            self.stats_['cltv_global'] = float(cltv.median())
            self.stats_['cltv_by_year'] = None
            if self.use_cohort and vyear is not None:
                med = pd.Series(cltv.values, index=vyear).groupby(level=0).median()
                self.stats_['cltv_by_year'] = {int(k): float(v) for k, v in med.items() if pd.notna(v)}

        # modes for small ordinal
        for col in ['original_loan_term', 'number_of_borrowers']:
            if col in df.columns:
                self.stats_[f'{col}_mode'] = self._mode(df[col])

        # Fallbacks génériques appris sur le train
        num_cols = df.select_dtypes(include='number').columns.tolist()
        self.stats_['global_num_median'] = {
            c: float(pd.to_numeric(df[c], errors='coerce').median()) for c in num_cols
        }
        nonnum_cols = [c for c in self.columns_fit_ if c not in num_cols]
        self.stats_['global_nonnum_mode'] = {c: self._mode(df[c]) for c in nonnum_cols}

        return self

    # ---------- transform (apply to TRAIN and TEST) ----------
    def transform(self, X):
        df = X.copy()

        # 0) Drop colonne non utilisée
        df.drop(columns=['pre_relief_refi_loan_seq_number'], errors='ignore', inplace=True)

        # 0-bis) Alignement de schéma : ajouter les colonnes vues au fit mais absentes ici
        if hasattr(self, 'columns_fit_'):
            for c in self.columns_fit_:
                if c not in df.columns:
                    df[c] = pd.NA

        # Flags "était manquant" avant toute imputation
        if self.missing_flag:
            cols_impute = df.columns
            missing0 = df[cols_impute].isna().add_prefix('was_missing_').astype('int8')

        # Helpers
        vyear = self._to_year(df['vintage']) if 'vintage' in df.columns else None

        # 1) Catégorielles : Unknown / NotApplicable
        if 'channel' in df.columns and isinstance(df['channel'].dtype, CategoricalDtype):
            df['channel'] = df['channel'].cat.add_categories(['Unknown']).fillna('Unknown')

        if 'property_valuation_method' in df.columns:
            pvm = pd.to_numeric(df['property_valuation_method'].astype('string'), errors='coerce')
            if vyear is not None:
                pvm = pvm.where(vyear >= 2017, 99)  # 99 NotApplicable avant 2017
            pvm = pvm.fillna(9)  # 9 NotAvailable
            df['property_valuation_method'] = pvm.astype('Int16').astype('category')

        if 'special_eligibility_program' in df.columns and isinstance(df['special_eligibility_program'].dtype, CategoricalDtype):
            df['special_eligibility_program'] = df['special_eligibility_program'].cat.add_categories(['Unknown']).fillna('Unknown')
            df['has_special_program'] = df['special_eligibility_program'].isin(['H', 'F', 'R']).astype('int8')

        if 'msa_md' in df.columns:
            df['msa_md'] = df['msa_md'].fillna(0)

        # 2) CREDIT SCORE
        if 'credit_score' in df.columns:
            df['cs_missing'] = df['credit_score'].isna().astype('int8')
            cs = pd.to_numeric(df['credit_score'], errors='coerce').clip(300, 850)

            # cohort fill
            if self.use_cohort and 'loan_purpose' in df.columns:
                if vyear is not None and self.stats_.get('credit_score_by_year_lp'):
                    keys = pd.Series(list(zip(vyear, df['loan_purpose'])), index=df.index)
                    mapped = self._map_fill(keys, self.stats_['credit_score_by_year_lp'])
                    cs = cs.fillna(mapped)
                if self.stats_.get('credit_score_by_lp'):
                    mapped = df['loan_purpose'].map(self.stats_['credit_score_by_lp'])
                    cs = cs.fillna(mapped)

            # global fallback
            cs = cs.fillna(self.stats_.get('credit_score_global', float(np.nan)))
            df['credit_score'] = pd.Series(cs, index=df.index).round().astype('Int16')

        # 3) MI%
        if 'mi_percent' in df.columns:
            df['mi_missing'] = df['mi_percent'].isna().astype('int8')
            mi = pd.to_numeric(df['mi_percent'], errors='coerce').astype('Float32')

            ltv = pd.to_numeric(df['original_ltv'], errors='coerce').clip(lower=0) if 'original_ltv' in df.columns else pd.Series(np.nan, index=df.index)
            # règle métier
            mi = mi.mask(ltv.le(80) & mi.isna(), 0.0)

            # cohort median by LTV bins (and year)
            if 'original_ltv' in df.columns:
                ltv_bins = pd.cut(ltv, self.ltv_bins, include_lowest=True, right=True)

                if self.use_cohort and vyear is not None and self.stats_.get('mi_by_year_bin'):
                    keys = pd.Series(list(zip(vyear, ltv_bins)), index=df.index)
                    mapped = self._map_fill(keys, self.stats_['mi_by_year_bin'])
                    mi = mi.fillna(mapped)

                if self.stats_.get('mi_by_bin'):
                    mapped = ltv_bins.map(self.stats_['mi_by_bin'])
                    mi = mi.fillna(mapped)

            df['mi_percent'] = mi.fillna(0.0).astype('Float32')
            df['has_mi'] = (df['mi_percent'] > 0).astype('int8')

        # 4) DTI
        if 'original_dti' in df.columns:
            df['dti_missing'] = df['original_dti'].isna().astype('int8')
            dti = pd.to_numeric(df['original_dti'], errors='coerce')

            if self.use_cohort and 'loan_purpose' in df.columns:
                if vyear is not None and self.stats_.get('dti_by_year_lp'):
                    keys = pd.Series(list(zip(vyear, df['loan_purpose'])), index=df.index)
                    mapped = self._map_fill(keys, self.stats_['dti_by_year_lp'])
                    dti = dti.fillna(mapped)
                if self.stats_.get('dti_by_lp'):
                    mapped = df['loan_purpose'].map(self.stats_['dti_by_lp'])
                    dti = dti.fillna(mapped)

            dti = dti.fillna(self.stats_.get('dti_global', float(np.nan)))
            df['original_dti'] = pd.Series(dti, index=df.index).round().astype('Int16')

        # 5) CLTV
        if 'original_cltv' in df.columns:
            df['cltv_missing'] = df['original_cltv'].isna().astype('int8')
            cltv = pd.to_numeric(df['original_cltv'], errors='coerce').astype('Float32')

            if 'original_ltv' in df.columns:
                ltv = pd.to_numeric(df['original_ltv'], errors='coerce').astype('Float32')
                cltv = pd.Series(cltv, index=df.index).fillna(ltv)
                cltv = pd.Series(np.where(ltv.notna(), np.maximum(cltv, ltv), cltv), index=df.index).astype('Float32')

            if self.use_cohort and vyear is not None and self.stats_.get('cltv_by_year'):
                mapped = pd.Series(vyear, index=df.index).map(self.stats_['cltv_by_year'])
                cltv = pd.Series(cltv, index=df.index).fillna(mapped)

            cltv = pd.Series(cltv, index=df.index).fillna(self.stats_.get('cltv_global', float(np.nan)))
            df['original_cltv'] = cltv.astype('Float32')

        # 6) Small ordinal -> mode (cast explicite en numérique)
        for col in ['original_loan_term', 'number_of_borrowers']:
            if col in df.columns:
                df[col + '_missing'] = df[col].isna().astype('int8')
                mode_val = self.stats_.get(f'{col}_mode', np.nan)
                ser_num = pd.to_numeric(df[col], errors='coerce')
                mode_num = pd.to_numeric(pd.Series([mode_val]), errors='coerce').iloc[0]
                ser_num = ser_num.fillna(mode_num)
                if ser_num.isna().any():
                    ser_num = ser_num.fillna(method='ffill').fillna(method='bfill')
                df[col] = pd.Series(ser_num.round(), index=df.index).astype('Int16')

        # 7) Filet de sécurité générique
        num_meds = self.stats_.get('global_num_median', {})
        for c, med in num_meds.items():
            if c in df.columns:
                s = pd.to_numeric(df[c], errors='coerce')
                if pd.notna(med):
                    df[c] = s.fillna(med)
                else:
                    df[c] = s

        non_modes = self.stats_.get('global_nonnum_mode', {})
        for c, mode_val in non_modes.items():
            if c not in df.columns:
                continue

            # Datetime
            if is_datetime64_any_dtype(df[c]):
                ser_dt = pd.to_datetime(df[c], errors='coerce')
                if pd.notna(mode_val):
                    df[c] = ser_dt.fillna(mode_val)
                else:
                    df[c] = ser_dt
                continue

            # Catégories
            if isinstance(df[c].dtype, CategoricalDtype):
                cat = df[c]
                cats = cat.cat.categories
                cats_dtype = cats.dtype

                if is_integer_dtype(cats_dtype) or is_float_dtype(cats_dtype):
                    if pd.isna(mode_val):
                        if len(cats) == 0:
                            continue
                        fill_val = cats[0]
                    else:
                        try_conv = pd.to_numeric(pd.Series([mode_val]), errors='coerce').iloc[0]
                        if pd.isna(try_conv):
                            fill_val = cats[0] if len(cats) > 0 else None
                        else:
                            if hasattr(cats_dtype, "type"):
                                fill_val = cats_dtype.type(try_conv)
                            elif len(cats) > 0:
                                fill_val = type(cats[0])(try_conv)
                            else:
                                fill_val = try_conv
                    if fill_val is None:
                        continue
                    if fill_val not in cats:
                        cat = cat.cat.add_categories([fill_val])
                    df[c] = cat.fillna(fill_val)
                else:
                    fill_val = mode_val if pd.notna(mode_val) else 'Unknown'
                    if fill_val not in cats:
                        cat = cat.cat.add_categories([fill_val])
                    df[c] = cat.fillna(fill_val)
                continue

            # Autres non-numériques
            fill_val = mode_val if pd.notna(mode_val) else 'Unknown'
            df[c] = df[c].fillna(fill_val)

        # Added missing flag or drop all missing flags depending on configuration
        if self.missing_flag:
            df = pd.concat([df, missing0], axis=1)
        else:
            drop_cols = [c for c in df.columns if c.endswith('_missing') or c.startswith('was_missing_')]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True, errors='ignore')

        return df















labels.py :
# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, Tuple

COLS_ORIG = [
    "credit_score","first_payment_date","first_time_homebuyer_flag","maturity_date",
    "msa_md","mi_percent","number_of_units","occupancy_status","original_cltv","original_dti",
    "original_upb","original_ltv","original_interest_rate","channel","ppm_flag","amortization_type",
    "property_state","property_type","postal_code","loan_sequence_number","loan_purpose",
    "original_loan_term","number_of_borrowers","seller_name","servicer_name","super_conforming_flag",
    "pre_relief_refi_loan_seq_number","special_eligibility_program","relief_refinance_indicator",
    "property_valuation_method","interest_only_indicator","mi_cancellation_indicator"
]

COLS_PERF = [
    "loan_sequence_number","monthly_reporting_period","current_actual_upb",
    "current_loan_delinquency_status","loan_age","remaining_months_to_legal_maturity",
    "defect_settlement_date","modification_flag","zero_balance_code","zero_balance_effective_date",
    "current_interest_rate","current_non_interest_bearing_upb","ddlpi","mi_recoveries",
    "net_sale_proceeds","non_mi_recoveries","total_expenses","legal_costs",
    "maintenance_and_preservation_costs","taxes_and_insurance","miscellaneous_expenses",
    "actual_loss_calculation","cumulative_modification_cost","step_modification_flag",
    "payment_deferral","estimated_ltv","zero_balance_removal_upb","delinquent_accrued_interest",
    "delinquency_due_to_disaster","borrower_assistance_status_code","current_month_modification_cost",
    "interest_bearing_upb"
]

DTYPES_ORIG = {
    "credit_score":"float32","first_payment_date":"string","maturity_date":"string",
    "msa_md":"float32","mi_percent":"float32","number_of_units":"float32","occupancy_status":"string",
    "original_cltv":"float32","original_dti":"float32","original_upb":"float64","original_ltv":"float32",
    "original_interest_rate":"float32","channel":"string","ppm_flag":"string","amortization_type":"string",
    "property_state":"string","property_type":"string","postal_code":"string","loan_sequence_number":"string",
    "loan_purpose":"string","original_loan_term":"float32","number_of_borrowers":"float32",
    "seller_name":"string","servicer_name":"string","super_conforming_flag":"string",
    "pre_relief_refi_loan_seq_number":"string","special_eligibility_program":"string",
    "relief_refinance_indicator":"string","property_valuation_method":"float32",
    "interest_only_indicator":"string","mi_cancellation_indicator":"string"
}

DTYPES_PERF = {
    "loan_sequence_number":"string","monthly_reporting_period":"string",
    "current_actual_upb":"float64","current_loan_delinquency_status":"string","loan_age":"float32",
    "remaining_months_to_legal_maturity":"float32","defect_settlement_date":"string",
    "modification_flag":"string","zero_balance_code":"string","zero_balance_effective_date":"string",
    "current_interest_rate":"float32","current_non_interest_bearing_upb":"float64","ddlpi":"string",
    "mi_recoveries":"float64","net_sale_proceeds":"string","non_mi_recoveries":"float64",
    "total_expenses":"float64","legal_costs":"float64","maintenance_and_preservation_costs":"float64",
    "taxes_and_insurance":"float64","miscellaneous_expenses":"float64","actual_loss_calculation":"float64",
    "cumulative_modification_cost":"float64","step_modification_flag":"string","payment_deferral":"string",
    "estimated_ltv":"float32","zero_balance_removal_upb":"float64","delinquent_accrued_interest":"float64",
    "delinquency_due_to_disaster":"string","borrower_assistance_status_code":"string",
    "current_month_modification_cost":"float64","interest_bearing_upb":"float64"
}

def _parse_yyyymm(series: pd.Series) -> pd.PeriodIndex:
    s = pd.to_datetime(series.astype("string"), format="%Y%m", errors="coerce")
    return s.dt.to_period("M")

def _month_diff(mpr: pd.Series, fpd: pd.Series) -> pd.Series:
    if not pd.api.types.is_period_dtype(mpr):
        mpr = pd.PeriodIndex(mpr.astype("string"), freq="M").to_series(index=mpr.index)
    if not pd.api.types.is_period_dtype(fpd):
        fpd = pd.PeriodIndex(fpd.astype("string"), freq="M").to_series(index=fpd.index)
    return (mpr.dt.year - fpd.dt.year) * 12 + (mpr.dt.month - fpd.dt.month)

def _normalize_zb_code(z: pd.Series) -> pd.Series:
    z = z.astype("string").str.strip()
    mask_num = z.str.fullmatch(r"\d+")
    z.loc[mask_num] = z.loc[mask_num].str.zfill(2)
    return z

def _make_default_row_flag(
    delinquency_status: pd.Series,
    zero_balance_code: pd.Series,
    delinquency_threshold: int,
    liquidation_codes: Iterable[str],
    include_ra: bool
) -> pd.Series:
    s = delinquency_status.astype("string").str.strip().str.upper()
    is_ra = s.eq("RA") if include_ra else pd.Series(False, index=s.index)
    s_num = pd.to_numeric(s.where(~is_ra), errors="coerce")
    is_90dpd = s_num.ge(delinquency_threshold)
    z = _normalize_zb_code(zero_balance_code)
    is_liq = z.isin(list(liquidation_codes))
    default_row = (is_90dpd | is_ra | is_liq).astype("int8")
    return default_row

def build_default_labels(
    path_orig: str,
    path_perf: str,
    window_months: int = 24,
    delinquency_threshold: int = 3,
    liquidation_codes: Tuple[str, ...] = ("02", "03", "09"),
    include_ra: bool = True,
    require_full_window: bool = False,
) -> pd.DataFrame:
    df_orig = pd.read_csv(
        path_orig, sep="|", header=None, names=COLS_ORIG, dtype=DTYPES_ORIG, engine="c"
    )
    df_perf = pd.read_csv(
        path_perf, sep="|", header=None, names=COLS_PERF, dtype=DTYPES_PERF, engine="c"
    )

    fpd = _parse_yyyymm(df_orig["first_payment_date"])
    df_orig["first_payment_date"] = fpd
    mpr = _parse_yyyymm(df_perf["monthly_reporting_period"])
    df_perf["monthly_reporting_period"] = mpr

    df_perf = df_perf.merge(
        df_orig[["loan_sequence_number", "first_payment_date"]],
        on="loan_sequence_number",
        how="left",
        copy=False,
        validate="m:1"
    )

    df_perf["months_since_orig"] = _month_diff(
        df_perf["monthly_reporting_period"], df_perf["first_payment_date"]
    ).astype("Int32")

    within = df_perf["months_since_orig"].le(window_months)
    dfw = df_perf.loc[within, ["loan_sequence_number",
                               "current_loan_delinquency_status",
                               "zero_balance_code"]].copy()

    dfw["default_row"] = _make_default_row_flag(
        dfw["current_loan_delinquency_status"],
        dfw["zero_balance_code"],
        delinquency_threshold=delinquency_threshold,
        liquidation_codes=liquidation_codes,
        include_ra=include_ra
    )

    loan_level = (
        dfw.groupby("loan_sequence_number", observed=True)["default_row"]
           .max()
           .rename(f"default_{window_months}m")
           .astype("Int8")
           .reset_index()
    )

    if require_full_window:
        has_T = (
            df_perf.loc[df_perf["months_since_orig"].eq(window_months), "loan_sequence_number"]
                  .dropna()
                  .drop_duplicates()
        )
        loan_level = loan_level.merge(
            has_T.to_frame("loan_sequence_number").assign(_ok=1),
            on="loan_sequence_number", how="inner"
        ).drop(columns="_ok")

    df_out = df_orig.merge(loan_level, on="loan_sequence_number", how="left")
    label_col = f"default_{window_months}m"
    df_out[label_col] = df_out[label_col].fillna(0).astype("Int8")
    return df_out


