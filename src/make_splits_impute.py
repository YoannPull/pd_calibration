# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import pandas as pd

# Internal imports
from features.impute import load_quarter_files, coerce_and_impute, READ_KW

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/processed", help="Root folder containing the labels subfolder")
    ap.add_argument("--subdir", default="default_labels", help="Subfolder with default_labels_{T}m_YYYYQ*.csv")
    ap.add_argument("--window", type=int, default=24, help="Label horizon (months) used in filenames")
    ap.add_argument("--out_dir", default="data/processed/default_labels_imputed", help="Output directory")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--imput_cohort", action="store_true",
                    help="Use cohort-aware medians (by vintage year / purpose / LTV bins)")

    # ---- NEW: time-based split options ----
    ap.add_argument("--train_range", type=str, default=None,
                    help="Quarter range for train, e.g. '2018Q1:2021Q4' (open-ended ':2021Q4' or '2018Q1:' allowed)")
    ap.add_argument("--val_range", type=str, default=None,
                    help="Quarter range for validation, e.g. '2022Q1:2022Q4'")
    ap.add_argument("--test_range", type=str, default=None,
                    help="Quarter range for test, e.g. '2023Q1:'")

    ap.add_argument("--train_years", type=str, default=None,
                    help="Years for train if no quarter ranges provided, e.g. '2018-2020,2022'")
    ap.add_argument("--val_years", type=str, default=None,
                    help="Years for validation, e.g. '2021'")
    ap.add_argument("--test_years", type=str, default=None,
                    help="Years for test, e.g. '2022-2023'")

    args = ap.parse_args()

    in_dir = Path(args.in_dir) / args.subdir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fallback if pyarrow isn't installed
    try:
        import pyarrow  # noqa: F401
    except Exception:
        READ_KW.pop("engine", None)
        READ_KW.pop("dtype_backend", None)

    # 1) Load and split quarters with user rules
    df_train, df_val, df_test = load_quarter_files(
        in_dir,
        window_months=args.window,
        train_range=args.train_range,
        val_range=args.val_range,
        test_range=args.test_range,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
    )

    # Basic report
    sizes = {"train": len(df_train), "validation": len(df_val), "test": len(df_test)}
    total = sum(sizes.values()) or 1
    for k, n in sizes.items():
        print(f"{k.capitalize():<12}: {n:>10,} rows  ({n/total:.2%})")

    # 2) Coerce types (schema) and 3) Impute (business rules)
    if len(df_train): df_train = coerce_and_impute(df_train, imput_cohort=args.imput_cohort)
    if len(df_val):   df_val   = coerce_and_impute(df_val,   imput_cohort=args.imput_cohort)
    if len(df_test):  df_test  = coerce_and_impute(df_test,  imput_cohort=args.imput_cohort)

    # 4) Save
    if args.format == "parquet":
        if len(df_train): df_train.to_parquet(out_dir / "train.parquet", index=False)
        if len(df_val):   df_val.to_parquet(out_dir / "validation.parquet", index=False)
        if len(df_test):  df_test.to_parquet(out_dir / "test.parquet", index=False)
    else:
        if len(df_train): df_train.to_csv(out_dir / "train.csv", index=False)
        if len(df_val):   df_val.to_csv(out_dir / "validation.csv", index=False)
        if len(df_test):  df_test.to_csv(out_dir / "test.csv", index=False)

    print(f"Done. Wrote to {out_dir}")

if __name__ == "__main__":
    main()
