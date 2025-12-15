#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-scored", required=True, help="parquet/csv avec au moins grade")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bucket-stats", required=True, help="bucket_stats_recalibrated.json")
    ap.add_argument("--grade-col", default="grade")
    ap.add_argument("--pd-col-out", default="pd_ms", help="nom de la PD master-scale en sortie")
    ap.add_argument("--replace-pd", action="store_true", help="si activé, remplace la colonne 'pd'")
    args = ap.parse_args()

    df = load_any(args.in_scored)

    payload = json.loads(Path(args.bucket_stats).read_text(encoding="utf-8"))
    # mapping grade -> pd (champ 'pd' lissé si isotonic, sinon pd_raw==pd)
    mp = {int(r["bucket"]): float(r["pd"]) for r in payload["train"]}

    if args.grade_col not in df.columns:
        raise ValueError(f"Colonne grade manquante: {args.grade_col}")

    out = df.copy()
    out[args.pd_col_out] = out[args.grade_col].map(mp)

    miss = out[args.pd_col_out].isna().mean()
    if miss > 0:
        raise ValueError(f"{miss:.2%} des lignes ont un grade sans mapping dans bucket-stats.")

    if args.replace_pd:
        out["pd"] = out[args.pd_col_out]

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.suffix in (".parquet", ".pq"):
        out.to_parquet(outp, index=False)
    else:
        out.to_csv(outp, index=False)

    print(f"✔ Master scale appliquée : {outp} (col={args.pd_col_out})")

if __name__ == "__main__":
    main()
