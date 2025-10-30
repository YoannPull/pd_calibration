# src/cli.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

# Imports qui reflètent exactement ton arbo actuelle
from features.labels import build_default_labels
from features.impute import coerce_and_impute, load_quarter_files

def save_df(df: pd.DataFrame, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out.with_suffix(".csv"), index=False)

# --------- sous-commandes ---------
def cmd_labels(args):
    df = build_default_labels(
        path_orig=args.orig,
        path_perf=args.perf,
        window_months=args.window,
        delinquency_threshold=args.delinquency_threshold,
        liquidation_codes=tuple(args.liquidation_codes),
        include_ra=not args.no_ra,
        require_full_window=args.require_full_window,
    )
    # option : ajouter vintage depuis first_payment_date
    if args.add_vintage:
        fpd = pd.PeriodIndex(pd.to_datetime(df["first_payment_date"].astype("string"), errors="coerce").to_period("M"))
        qnum = ((fpd.month - 1) // 3 + 1).astype("int")
        df["vintage"] = fpd.year.astype("string") + "Q" + qnum.astype("string")

    out = Path(args.out)
    save_df(df, out)
    print(f"[labels] -> {out}  shape={df.shape}")

def cmd_impute(args):
    inp = Path(args.input)
    out = Path(args.out)

    # lecture simple selon extension
    if inp.suffix.lower() == ".parquet":
        df = pd.read_parquet(inp)
    else:
        df = pd.read_csv(inp)

    df2 = coerce_and_impute(df, imput_cohort=args.cohort)
    save_df(df2, out)
    print(f"[impute] {inp.name} -> {out}  shape={df2.shape}")

def cmd_splits(args):
    # Attend un dossier contenant des fichiers nommés comme:
    # default_labels_{T}m_YYYYQ*.csv
    train, val, test = load_quarter_files(Path(args.dir), window_months=args.window)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train.empty:
        train.to_parquet(out_dir / "train.parquet", index=False)
    if not val.empty:
        val.to_parquet(out_dir / "val.parquet", index=False)
    if not test.empty:
        test.to_parquet(out_dir / "test.parquet", index=False)

    print(f"[splits] -> {out_dir}  "
          f"train={len(train)}  val={len(val)}  test={len(test)}")

# --------- parser principal ---------
def build_parser():
    ap = argparse.ArgumentParser(prog="mlproj", description="CLI projet ML")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("labels", help="Générer les labels à partir des fichiers orig/perf")
    p1.add_argument("--orig", required=True, help="Chemin fichier d'origination")
    p1.add_argument("--perf", required=True, help="Chemin fichier de performance")
    p1.add_argument("--window", type=int, default=24)
    p1.add_argument("--delinquency_threshold", type=int, default=3)
    p1.add_argument("--liquidation_codes", nargs="+", default=["02", "03", "09"])
    p1.add_argument("--no_ra", action="store_true", help="Ne pas considérer RA comme défaut")
    p1.add_argument("--require_full_window", action="store_true")
    p1.add_argument("--add_vintage", action="store_true")
    p1.add_argument("--out", required=True, help="Chemin sortie (.csv ou .parquet)")
    p1.set_defaults(func=cmd_labels)

    p2 = sub.add_parser("impute", help="Coercion de types + imputation business")
    p2.add_argument("--input", required=True, help="Fichier d'entrée (csv/parquet)")
    p2.add_argument("--out", required=True, help="Fichier de sortie (csv/parquet)")
    p2.add_argument("--cohort", action="store_true", help="Imputation par cohortes (vintage/purpose/LTV)")
    p2.set_defaults(func=cmd_impute)

    p3 = sub.add_parser("splits", help="Charger les CSV trimestriels et créer train/val/test")
    p3.add_argument("--dir", required=True, help="Dossier avec default_labels_{T}m_YYYYQ*.csv")
    p3.add_argument("--window", type=int, default=24, help="T de la convention de nommage")
    p3.add_argument("--out_dir", required=True, help="Dossier de sortie des splits")
    p3.set_defaults(func=cmd_splits)

    return ap

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
