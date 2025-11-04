#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build default labels (default_{T}m) per quarter, in parallel, with EXPLICIT splits only.

Sorties:
  data/processed/default_labels/
    └── window=<T>m/
        ├── quarter=<YYYYQn>/data.parquet
        ├── pooled.parquet              # concat des quarters design_explicit
        ├── oos.parquet                 # concat des quarters oos_explicit
        ├── _summary.csv
        ├── _manifest.json
        └── _splits.json                # copie fidèle des splits explicites (avec liste de validations)

Config (obligatoire) : config.yml
  data.root, data.quarters[]                    # liste globale de tous les quarters à produire
  labels.*                                      # règles de labeling
  output.dir, output.format
  splits:
    mode: explicit
    explicit:
      design_quarters: [ ... ]                  # pooled
      oos_quarters:    [ ... ]                  # hold-out final
      default_val_quarters: ["YYYYQn", ...]     # (optionnel) liste; doit être dans design_quarters
      # rétrocompat: default_val_quarter: "YYYYQn" (string) -> converti en liste d'un élément
"""

from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple

import yaml
import pandas as pd

from features.labels import build_default_labels


# ------------------------- Helpers -------------------------
def qpaths(root: Path, q: str) -> Tuple[Path, Path]:
    d = root / f"historical_data_{q}"
    return d / f"historical_data_{q}.txt", d / f"historical_data_time_{q}.txt"


def _infer_vintage(df: pd.DataFrame) -> pd.Series:
    fpd = df["first_payment_date"]
    qnum = ((fpd.dt.month - 1) // 3 + 1).astype("int")
    return fpd.dt.year.astype("string") + "Q" + qnum.astype("string")


def _to_parquet_safe(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_period_dtype(out[c]):
            out[c] = out[c].dt.to_timestamp("M")
    out.to_parquet(path, index=False)


def _to_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _save_df(df: pd.DataFrame, path: Path, fmt: str):
    if fmt == "parquet":
        _to_parquet_safe(df, path)
    else:
        _to_csv(df, path)


def _concat_parquet(outputs: List[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow.parquet as pq
        writer = None
        schema = None
        for p in outputs:
            if not p.exists():
                continue
            table = pq.read_table(p)
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(str(out_path), schema)
            else:
                if table.schema != schema:
                    table = table.cast(schema)
            writer.write_table(table)
        if writer is not None:
            writer.close()
        else:
            pd.DataFrame().to_parquet(out_path, index=False)
    except Exception:
        dfs = [pd.read_parquet(p) for p in outputs if p.exists()]
        if dfs:
            pd.concat(dfs, ignore_index=True).to_parquet(out_path, index=False)
        else:
            pd.DataFrame().to_parquet(out_path, index=False)


def _concat_csv(outputs: List[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote = False
    with open(out_path, "w", newline="") as fout:
        for p in outputs:
            if not p.exists():
                continue
            with open(p, "r") as fin:
                if wrote:
                    next(fin, None)
                for line in fin:
                    fout.write(line)
            wrote = True


def concat_outputs(outputs: List[Path], out_path: Path, fmt: str):
    if fmt == "parquet":
        _concat_parquet(outputs, out_path)
    else:
        _concat_csv(outputs, out_path)


# ------------------------- Worker -------------------------
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
) -> Dict[str, Any]:
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
        df["vintage"] = _infer_vintage(df)

        q_dir = Path(window_dir) / f"quarter={q}"
        out_path = q_dir / f"data.{fmt}"
        _save_df(df, out_path, fmt)

        lbl_col = f"default_{window_months}m"
        return {
            "quarter": q,
            "ok": True,
            "path": str(out_path),
            "n_rows": int(len(df)),
            "n_unique_loans": int(df["loan_sequence_number"].nunique()),
            "default_rate": float(df[lbl_col].mean()) if lbl_col in df.columns and len(df) > 0 else None,
        }
    except Exception as e:
        return {"quarter": q, "ok": False, "error": repr(e)}


# ------------------------- CLI -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Build labels per quarter with explicit splits (supports multiple validation quarters).")
    p.add_argument("--config", default="config.yml", help="YAML with data.*, labels.*, output.*, splits.explicit.*")
    p.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    # ---- Mandatory sections
    if cfg.get("splits", {}).get("mode", "") != "explicit":
        raise SystemExit("[ERR] config.splits.mode must be 'explicit' (no legacy/boundary mode).")
    explicit = cfg["splits"].get("explicit") or {}
    design_quarters = list(explicit.get("design_quarters", []))
    oos_quarters = list(explicit.get("oos_quarters", []))

    # Accept single string OR list for default validation
    dvq_single = explicit.get("default_val_quarter")
    dvq_list = explicit.get("default_val_quarters")
    if dvq_list is not None and not isinstance(dvq_list, list):
        raise SystemExit("[ERR] splits.explicit.default_val_quarters must be a list if provided.")
    if dvq_list is None and dvq_single:
        default_val_quarters: List[str] = [dvq_single]
    else:
        default_val_quarters = list(dvq_list or [])

    # ---- Data / output / labels
    root = Path(cfg["data"]["root"])
    all_quarters: List[str] = list(cfg["data"].get("quarters", []))
    if not all_quarters:
        raise SystemExit("[ERR] data.quarters must list ALL quarters to build (explicit mode).")

    out_dir = Path(cfg.get("output", {}).get("dir", "data/processed/default_labels"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = (cfg.get("output", {}).get("format", "parquet")).lower()
    if fmt not in ("parquet", "csv"):
        raise SystemExit("[ERR] output.format must be 'parquet' or 'csv'.")

    label_cfg = cfg["labels"]
    window_months = int(label_cfg["window_months"])
    delinquency_threshold = int(label_cfg.get("delinquency_threshold", 3))
    liquidation_codes = tuple(label_cfg.get("liquidation_codes", ["02", "03", "09"]))
    include_ra = bool(label_cfg.get("include_ra", True))
    require_full_window = bool(label_cfg.get("require_full_window", False))

    window_dir = out_dir / f"window={window_months}m"
    window_dir.mkdir(parents=True, exist_ok=True)

    # ---- Validate explicit splits against all_quarters
    missing_d = [q for q in design_quarters if q not in all_quarters]
    missing_o = [q for q in oos_quarters if q not in all_quarters]
    if missing_d or missing_o:
        raise SystemExit(f"[ERR] splits.explicit contains quarters not in data.quarters. "
                         f"missing_in_all (design={missing_d}, oos={missing_o})")

    inter = set(design_quarters).intersection(oos_quarters)
    if inter:
        raise SystemExit(f"[ERR] A quarter cannot be in both design and oos: {sorted(inter)}")

    # default_val_quarters ⊆ design_quarters
    invalid_val = [q for q in default_val_quarters if q not in design_quarters]
    if invalid_val:
        raise SystemExit(f"[ERR] default_val_quarters must be subset of design_quarters. Offenders: {invalid_val}")

    # ---- Produce all quarters requested (build once)
    workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
    print(f"→ Building {len(all_quarters)} quarters with {workers} workers...")
    from functools import partial
    worker = partial(
        quarter_worker,
        root=str(root),
        window_dir=str(window_dir),
        fmt=fmt,
        window_months=window_months,
        delinquency_threshold=delinquency_threshold,
        liquidation_codes=liquidation_codes,
        include_ra=include_ra,
        require_full_window=require_full_window,
    )

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    produced_quarters: List[str] = []

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, q) for q in all_quarters]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            if r.get("ok"):
                produced_quarters.append(r["quarter"])
                print(f"✔ {r['quarter']} saved ({r['n_rows']} rows)")
            else:
                print(f"[WARN] {r['quarter']} failed: {r.get('error')}")

    # ---- Summary
    lbl_col = f"default_{window_months}m"
    summary_rows = [{
        k: r.get(k) for k in ("quarter","n_rows","n_unique_loans","default_rate","ok","path","error")
    } for r in sorted(results, key=lambda x: x["quarter"])]
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(window_dir / "_summary.csv", index=False)

    # ---- Build pooled/oos from EXPLICIT lists (using only produced/ok)
    ok_map: Dict[str, Path] = {r["quarter"]: Path(r["path"]) for r in results if r.get("ok") and r.get("path")}

    # keep order as in data.quarters
    design_ordered = [q for q in all_quarters if q in set(design_quarters) and q in ok_map]
    oos_ordered    = [q for q in all_quarters if q in set(oos_quarters) and q in ok_map]

    if design_ordered:
        concat_outputs([ok_map[q] for q in design_ordered], window_dir / f"pooled.{fmt}", fmt)
        print(f"✔ pooled -> {window_dir / f'pooled.{fmt}'}  (quarters: {design_ordered[:3]}{'...' if len(design_ordered)>3 else ''})")
    else:
        print("[WARN] No design_quarters produced; pooled not created.")

    if oos_ordered:
        concat_outputs([ok_map[q] for q in oos_ordered], window_dir / f"oos.{fmt}", fmt)
        print(f"✔ oos -> {window_dir / f'oos.{fmt}'}  (quarters: {oos_ordered[:3]}{'...' if len(oos_ordered)>3 else ''})")
    else:
        print("[INFO] No oos_quarters (empty).")

    # ---- Persist splits definition (for downstream scripts)
    splits_payload = {
        "mode": "explicit",
        "pooled_quarters": design_ordered,
        "oos_quarters": oos_ordered,
        "validation_quarters": default_val_quarters,  # toujours une LISTE
    }
    (window_dir / "_splits.json").write_text(json.dumps(splits_payload, indent=2), encoding="utf-8")

    # ---- Manifest
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_dir": str(out_dir.resolve()),
        "window_months": window_months,
        "format": fmt,
        "quarters_requested": all_quarters,
        "quarters_produced": produced_quarters,
        "elapsed_sec": round(time.time() - t0, 2),
        "splits_file": str((window_dir / "_splits.json").resolve()),
        "label_column": lbl_col,
    }
    (window_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print("✓ Done.")


if __name__ == "__main__":
    main()
