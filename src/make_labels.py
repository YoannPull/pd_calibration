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
