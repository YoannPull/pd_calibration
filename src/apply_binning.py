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
