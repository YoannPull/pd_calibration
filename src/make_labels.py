#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère des labels (default_{T}m) à partir de fichiers historisés par quarter.
Sorties (par défaut, via Makefile): data/processed/runs/<RUN>/labels/
  └── window=<T>m/
      ├── quarter=<YYYYQn>/data.parquet
      ├── pooled.parquet
      ├── _manifest.json
      └── _summary.csv
Compatible avec config.yml (sections: data.root, data.quarters[], output.*, labels.*).
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import yaml
import pandas as pd
from features.labels import build_default_labels


def qpaths(root: Path, q: str):
    d = root / f"historical_data_{q}"
    return d / f"historical_data_{q}.txt", d / f"historical_data_time_{q}.txt"


def _infer_vintage(df: pd.DataFrame) -> pd.Series:
    # build_default_labels range "first_payment_date" en Period[M]
    fpd = df["first_payment_date"]
    qnum = ((fpd.dt.month - 1) // 3 + 1).astype("int")
    return fpd.dt.year.astype("string") + "Q" + qnum.astype("string")


def _save_df(df: pd.DataFrame, path: Path, fmt: str = "parquet"):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        out = df.copy()
        # Parquet ne supporte pas Period -> timestamp fin de mois
        for c in out.columns:
            if pd.api.types.is_period_dtype(out[c]):
                out[c] = out[c].dt.to_timestamp("M")
        out.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Build default labels (quartered or single-file) with partitioned outputs")
    p.add_argument("--config", default="config.yml", help="Chemin du YAML (data.*, labels.*, output.*)")
    p.add_argument("--outdir", default=None, help="Dossier de sortie (ex: data/processed/runs/<RUN>/labels)")
    p.add_argument("--format", choices=["parquet", "csv"], default=None, help="Format de sortie (override config)")
    p.add_argument("--pooled", action="store_true", help="Sauvegarder aussi le pooled (override config.output.make_pooled)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    root = Path(cfg["data"]["root"])
    quarters = list(cfg["data"].get("quarters", []))
    out_dir = Path(args.outdir or cfg.get("output", {}).get("dir", "data/processed/default_labels"))
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = (args.format or cfg.get("output", {}).get("format", "parquet")).lower()

    # Paramètres de labels
    label_cfg = cfg["labels"]
    window_months = int(label_cfg["window_months"])
    delinquency_threshold = int(label_cfg.get("delinquency_threshold", 3))
    liquidation_codes = tuple(label_cfg.get("liquidation_codes", ["02", "03", "09"]))
    include_ra = bool(label_cfg.get("include_ra", True))
    require_full_window = bool(label_cfg.get("require_full_window", False))

    want_pooled = bool(args.pooled or cfg.get("output", {}).get("make_pooled", False))

    # Partition racine pour cet horizon
    window_dir = out_dir / f"window={window_months}m"
    window_dir.mkdir(parents=True, exist_ok=True)

    lbl_col = f"default_{window_months}m"
    summary_rows = []
    produced_quarters = []

    def _summ(df: pd.DataFrame, tag: str):
        summary_rows.append({
            "tag": tag,
            "n_rows": int(len(df)),
            "n_unique_loans": int(df["loan_sequence_number"].nunique()),
            "default_rate": float(df[lbl_col].mean()) if lbl_col in df.columns and len(df) > 0 else None,
        })

    if quarters:
        # ---- Mode multi-quarters
        pooled = []
        for q in quarters:
            orig, perf = qpaths(root, q)
            if not orig.exists() or not perf.exists():
                print(f"[WARN] Skipping {q}: files not found -> {orig} / {perf}")
                continue

            df = build_default_labels(
                path_orig=str(orig),
                path_perf=str(perf),
                window_months=window_months,
                delinquency_threshold=delinquency_threshold,
                liquidation_codes=liquidation_codes,
                include_ra=include_ra,
                require_full_window=require_full_window,
            )
            df["vintage"] = _infer_vintage(df)

            q_dir = window_dir / f"quarter={q}"
            out_path = q_dir / f"data.{fmt}"
            _save_df(df, out_path, fmt)
            print(f"✔ Saved {q} -> {out_path}  shape={df.shape}")

            produced_quarters.append(q)
            _summ(df, q)
            if want_pooled:
                pooled.append(df)

        if want_pooled and pooled:
            df_pooled = pd.concat(pooled, ignore_index=True)
            pooled_path = window_dir / f"pooled.{fmt}"
            _save_df(df_pooled, pooled_path, fmt)
            print(f"✔ Saved pooled -> {pooled_path}  shape={df_pooled.shape}")
            _summ(df_pooled, "pooled")

    else:
        # ---- Mode single-file (compat historique)
        orig = Path(cfg["data"]["orig"])
        perf = Path(cfg["data"]["perf"])
        if not orig.exists() or not perf.exists():
            raise SystemExit(f"[ERR] Input files not found -> {orig} / {perf}")

        df = build_default_labels(
            path_orig=str(orig),
            path_perf=str(perf),
            window_months=window_months,
            delinquency_threshold=delinquency_threshold,
            liquidation_codes=liquidation_codes,
            include_ra=include_ra,
            require_full_window=require_full_window,
        )
        df["vintage"] = _infer_vintage(df)

        out_path = window_dir / f"single.{fmt}"
        _save_df(df, out_path, fmt)
        print(f"✔ Saved -> {out_path}  shape={df.shape}")
        _summ(df, "single")

    # ---- Écrit un manifeste (provenance + snapshot de config)
    manifest = {
        "run_dir": str(out_dir.resolve()),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "window_months": window_months,
        "format": fmt,
        "quarters_requested": quarters,
        "quarters_produced": produced_quarters,
        "pooled": want_pooled,
        "config_snapshot": cfg,
    }
    (window_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    # ---- Résumé agrégé
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(window_dir / "_summary.csv", index=False)


if __name__ == "__main__":
    main()
