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
