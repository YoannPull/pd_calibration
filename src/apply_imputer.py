#!/usr/bin/env python3
#src/apply_imputed.py
"""
Apply a saved DataImputer to one or many datasets.

This script is meant to be used *after* training an imputer on a reference dataset
(e.g., training / in-sample). It loads the persisted imputer (joblib) and applies
it to new data files (single file or batch via glob).

Examples
--------
# 1) Single file (in -> out)
python src/apply_imputer.py \
  --imputer artifacts/imputer/imputer.joblib \
  --data data/processed/labels/window=24m/quarter=2023Q1/data.parquet \
  --out  data/processed/imputed/2023Q1.parquet

# 2) Batch via glob (output directory + suffix)
python src/apply_imputer.py \
  --imputer artifacts/imputer/imputer.joblib \
  --glob "data/processed/labels/window=24m/quarter=*/data.parquet" \
  --out-dir data/processed/imputed/quarters \
  --suffix "" \
  --meta artifacts/imputer/imputer_meta.json \
  --drop-extras

Notes
-----
- Optionally, a target column can be preserved and re-attached via --target.
- --meta and --drop-extras can enforce strict feature alignment with the training
  feature set (same columns and ordering, optionally dropping unknown columns).
- --drop-missing-flags removes missingness indicator columns if you do not want
  them in the final dataset.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd
from joblib import load


def load_any(path: Path) -> pd.DataFrame:
    """
    Load a dataset from disk.

    Supported formats:
    - Parquet (.parquet, .pq)
    - CSV (fallback)
    """
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_any(df: pd.DataFrame, path: Path, force_format: Optional[str] = None) -> None:
    """
    Save a dataset to disk, creating parent directories if needed.

    Parameters
    ----------
    df : pd.DataFrame
        Data to write.
    path : Path
        Output file path. If no extension is provided and force_format is None,
        parquet is used by default and the suffix is set to ".parquet".
    force_format : {"parquet", "csv"}, optional
        If provided, overrides the path extension.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve format from either the explicit argument or the file extension.
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
    """
    Build an output path for batch mode.

    Example: "data.parquet" + suffix="_imputed" -> "data_imputed.parquet"
    """
    stem = inp.stem
    ext = ("." + force_format) if force_format else (inp.suffix or ".parquet")
    return out_dir / f"{stem}{suffix}{ext}"


def parse_args():
    """Define and parse CLI arguments."""
    p = argparse.ArgumentParser(description="Apply a saved DataImputer to dataset(s)")
    p.add_argument("--imputer", required=True, help="Path to imputer.joblib")

    # Inputs
    p.add_argument("--data", action="append", default=[], help="Input file (repeatable)")
    p.add_argument("--glob", default=None, help="Glob pattern for batch (e.g. 'dir/*.parquet')")

    # Outputs
    p.add_argument("--out", default=None, help="Single output path (only if single input)")
    p.add_argument("--out-dir", default=None, help="Output directory for batch mode")
    p.add_argument("--suffix", default="_imputed", help="Suffix added before extension in batch")
    p.add_argument("--format", choices=["parquet", "csv"], default=None, help="Force output format")

    # Options
    p.add_argument("--target", default=None, help="Optional target column to preserve/reattach")
    p.add_argument("--meta", default=None, help="Path to imputer_meta.json to align features")
    p.add_argument(
        "--drop-extras",
        action="store_true",
        help="Drop columns not in meta features (requires --meta)",
    )
    p.add_argument(
        "--drop-missing-flags",
        action="store_true",
        help="Drop missingness indicator columns (e.g., 'was_missing_*' or '*_missing')",
    )
    return p.parse_args()


def align_to_meta(df: pd.DataFrame, meta_path: Path, drop_extras: bool) -> pd.DataFrame:
    """
    Align dataframe columns to a reference feature list stored in a meta JSON.

    The meta file is expected to contain a list under the key "features".
    - If drop_extras=True, columns not present in meta features are removed.
    - Otherwise, meta features are moved to the front (in the meta order),
      and any remaining columns are appended after.

    This is useful to enforce consistency between training and inference data.
    """
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
    """
    Drop columns that encode missingness indicators (if present).

    Heuristic patterns:
    - Columns starting with "was_missing_"
    - Columns ending with "_missing"
    """
    rx = re.compile(r"(?:^was_missing_|_missing$)", re.I)
    to_drop = [c for c in df.columns if rx.search(c)]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def apply_one(
    imputer,
    inp: Path,
    out: Path,
    fmt: Optional[str],
    target: Optional[str],
    meta_path: Optional[Path],
    drop_extras: bool,
    drop_flags: bool,
) -> None:
    """
    Apply the imputer to a single dataset and write the result.

    Workflow:
    1) Load input file
    2) Optionally detach target column
    3) imputer.transform(X)
    4) Optionally reattach target
    5) Optional feature alignment via meta
    6) Optional removal of missingness flags
    7) Save output
    """
    df = load_any(inp)

    # Preserve target if requested (do not feed it to the imputer).
    y = None
    if target and target in df.columns:
        y = df[target].copy()
        df = df.drop(columns=[target])

    # Apply imputation. We assume the imputer returns a pandas DataFrame.
    df_imp = imputer.transform(df)

    # Re-attach target (as last step before optional column alignment).
    if y is not None:
        df_imp[target] = y.values

    # Align columns to the training feature list if a meta file is provided.
    if meta_path and meta_path.exists():
        df_imp = align_to_meta(df_imp, meta_path, drop_extras=drop_extras)

    # Drop missingness indicators if requested.
    if drop_flags:
        df_imp = drop_missing_flags(df_imp)

    save_any(df_imp, out, fmt)
    print(f"✔ Imputation applied: {inp} → {out}  shape={df_imp.shape}")


def main():
    """CLI entry point."""
    args = parse_args()
    imputer = load(args.imputer)

    # Collect input files from explicit --data and optional --glob.
    inputs: List[Path] = []
    for d in args.data:
        inputs.append(Path(d))

    if args.glob:
        # Note: Path().glob uses the current working directory as base.
        for p in sorted(Path().glob(args.glob)):
            if p.is_file():
                inputs.append(p)

    # Remove duplicates and sort for deterministic processing.
    inputs = sorted(set(inputs))
    if not inputs:
        raise SystemExit("No input. Use --data and/or --glob.")

    meta_path = Path(args.meta) if args.meta else None

    # Single-input mode: allow --out, or derive output path.
    if len(inputs) == 1:
        inp = inputs[0]

        if args.out is None and args.out_dir is None:
            # Default: write next to the input file.
            out = derive_out_path(inp, inp.parent, args.suffix, args.format)
        elif args.out is not None:
            # User explicitly provided the output file.
            out = Path(args.out)
        else:
            # User provided an output directory (still fine for one input).
            out = derive_out_path(inp, Path(args.out_dir), args.suffix, args.format)

        apply_one(
            imputer,
            inp,
            out,
            args.format,
            args.target,
            meta_path,
            drop_extras=args.drop_extras,
            drop_flags=args.drop_missing_flags,
        )
        return

    # Batch mode: require --out-dir for multiple inputs.
    if args.out_dir is None:
        raise SystemExit("Multiple inputs detected: please provide --out-dir.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for inp in inputs:
        out = derive_out_path(inp, out_dir, args.suffix, args.format)
        apply_one(
            imputer,
            inp,
            out,
            args.format,
            args.target,
            meta_path,
            drop_extras=args.drop_extras,
            drop_flags=args.drop_missing_flags,
        )

    print(f"✔ Done. Wrote {len(inputs)} file(s) into {out_dir.resolve()}")


if __name__ == "__main__":
    main()
