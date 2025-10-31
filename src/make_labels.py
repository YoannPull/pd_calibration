#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build default labels (default_{T}m) per quarter, in parallel.
Outputs (default, via Makefile): data/processed/default_labels/
  └── window=<T>m/
      ├── quarter=<YYYYQn>/data.parquet
      ├── pooled.parquet        # concat of quarters <= pooled_until
      ├── oos.parquet           # concat of quarters  > pooled_until
      ├── _manifest.json
      └── _summary.csv
Config: config.yml (sections: data.root, data.quarters[], output.*, labels.*).
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import yaml
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

from features.labels import build_default_labels


# ------------------------- Helpers -------------------------
def qpaths(root: Path, q: str) -> Tuple[Path, Path]:
    d = root / f"historical_data_{q}"
    return d / f"historical_data_{q}.txt", d / f"historical_data_time_{q}.txt"


def _infer_vintage(df: pd.DataFrame) -> pd.Series:
    # In build_default_labels, first_payment_date is Period[M]
    fpd = df["first_payment_date"]
    qnum = ((fpd.dt.month - 1) // 3 + 1).astype("int")
    return fpd.dt.year.astype("string") + "Q" + qnum.astype("string")


def _to_parquet_safe(df: pd.DataFrame, path: Path):
    """
    Parquet doesn't support Period dtype → convert Period[M] to end-of-month timestamps.
    """
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


def _concat_parquet(outputs: List[Path], out_path: Path):
    """
    Concatenate many parquet files into a single parquet, streaming if possible.
    Falls back to pandas concat if pyarrow streaming isn't available.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow as pa
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
                # Optional: enforce the same schema (cast if needed)
                if table.schema != schema:
                    table = table.cast(schema)
            writer.write_table(table)
        if writer is not None:
            writer.close()
        else:
            # nothing wrote → create empty file with empty df
            pd.DataFrame().to_parquet(out_path, index=False)
    except Exception:
        # Fallback: pandas concat (may be memory heavy)
        dfs = []
        for p in outputs:
            if p.exists():
                dfs.append(pd.read_parquet(p))
        if dfs:
            pd.concat(dfs, ignore_index=True).to_parquet(out_path, index=False)
        else:
            pd.DataFrame().to_parquet(out_path, index=False)


def _concat_csv(outputs: List[Path], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with open(out_path, "w", newline="") as fout:
        for p in outputs:
            if not p.exists():
                continue
            with open(p, "r") as fin:
                if wrote_header:
                    next(fin, None)  # skip header
                for line in fin:
                    fout.write(line)
            wrote_header = True


def concat_outputs(outputs: List[Path], out_path: Path, fmt: str):
    if fmt == "parquet":
        _concat_parquet(outputs, out_path)
    else:
        _concat_csv(outputs, out_path)


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
    """
    Worker process: builds labels for a quarter, writes file, returns summary.
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
        df["vintage"] = _infer_vintage(df)

        q_dir = window_dir_p / f"quarter={q}"
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
    p = argparse.ArgumentParser(description="Build default labels per quarter (parallel), plus pooled/oos splits.")
    p.add_argument("--config", default="config.yml", help="YAML with data.*, labels.*, output.*")
    p.add_argument("--outdir", default=None, help="Output base dir (default from config.output.dir)")
    p.add_argument("--format", choices=["parquet", "csv"], default=None, help="Output format override")
    p.add_argument("--pooled", action="store_true", help="Also build pooled (override output.make_pooled)")
    p.add_argument("--pooled-until", default=None, help="Quarter (YYYYQn) last included in pooled; others go to oos")
    p.add_argument("--workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))

    root = Path(cfg["data"]["root"])
    quarters: List[str] = list(cfg["data"].get("quarters", []))
    out_dir = Path(args.outdir or cfg.get("output", {}).get("dir", "data/processed/default_labels"))
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = (args.format or cfg.get("output", {}).get("format", "parquet")).lower()
    want_pooled = bool(args.pooled or cfg.get("output", {}).get("make_pooled", False))
    pooled_until = args.pooled_until or cfg.get("output", {}).get("pooled_until", None)

    # Labels config
    label_cfg = cfg["labels"]
    window_months = int(label_cfg["window_months"])
    delinquency_threshold = int(label_cfg.get("delinquency_threshold", 3))
    liquidation_codes = tuple(label_cfg.get("liquidation_codes", ["02", "03", "09"]))
    include_ra = bool(label_cfg.get("include_ra", True))
    require_full_window = bool(label_cfg.get("require_full_window", False))

    window_dir = out_dir / f"window={window_months}m"
    window_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results: List[Dict[str, Any]] = []
    produced_quarters: List[str] = []

    # ---------- Parallel per quarter ----------
    if quarters:
        workers = args.workers or max(1, (os.cpu_count() or 2) - 1)
        print(f"→ Building {len(quarters)} quarters with {workers} workers...")

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(
                    quarter_worker,
                    q,
                    str(root),
                    str(window_dir),
                    fmt,
                    window_months,
                    delinquency_threshold,
                    liquidation_codes,
                    include_ra,
                    require_full_window,
                )
                for q in quarters
            ]
            for fut in as_completed(futures):
                r = fut.result()
                results.append(r)
                if r.get("ok"):
                    produced_quarters.append(r["quarter"])
                    print(f"✔ {r['quarter']} saved ({r['n_rows']} rows)")
                else:
                    print(f"[WARN] {r['quarter']} failed: {r.get('error')}")

    else:
        # Single-file mode (compat)
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
        results.append({
            "quarter": "single",
            "ok": True,
            "path": str(out_path),
            "n_rows": int(len(df)),
            "n_unique_loans": int(df["loan_sequence_number"].nunique()),
            "default_rate": float(df[f"default_{window_months}m"].mean()),
        })
        produced_quarters.append("single")

    # ---------- Summary ----------
    lbl_col = f"default_{window_months}m"
    summary_rows = [ {k: r.get(k) for k in ("quarter","n_rows","n_unique_loans","default_rate","ok","path","error")} for r in sorted(results, key=lambda x: x["quarter"]) ]
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(window_dir / "_summary.csv", index=False)

    # ---------- Build pooled / oos ----------
    if want_pooled and quarters:
        # Filter OK quarters with output paths
        ok_map: Dict[str, Path] = {
            r["quarter"]: Path(r["path"])
            for r in results
            if r.get("ok") and r.get("path")
        }

        # Sort quarters as in config order
        quarters_ordered = [q for q in quarters if q in ok_map]

        if pooled_until is None:
            print("[WARN] No 'pooled_until' provided (config.output.pooled_until or --pooled-until).")
            print("       All quarters will be considered pooled; oos will be empty.")
            pooled_list = quarters_ordered
            oos_list: List[str] = []
        else:
            if pooled_until not in quarters_ordered:
                print(f"[WARN] pooled_until={pooled_until} not in produced quarters; using all as pooled.")
                pooled_list = quarters_ordered
                oos_list = []
            else:
                idx = quarters_ordered.index(pooled_until)
                pooled_list = quarters_ordered[: idx + 1]
                oos_list = quarters_ordered[idx + 1 :]

        pooled_paths = [ok_map[q] for q in pooled_list]
        oos_paths = [ok_map[q] for q in oos_list]

        if pooled_paths:
            concat_outputs(pooled_paths, window_dir / f"pooled.{fmt}", fmt)
            print(f"✔ pooled -> {window_dir / f'pooled.{fmt}'}  (quarters: {pooled_list[:3]}{'...' if len(pooled_list)>3 else ''})")
        else:
            print("[WARN] No pooled quarters found; pooled file not created.")

        if oos_paths:
            concat_outputs(oos_paths, window_dir / f"oos.{fmt}", fmt)
            print(f"✔ oos -> {window_dir / f'oos.{fmt}'}  (quarters: {oos_list[:3]}{'...' if len(oos_list)>3 else ''})")
        else:
            print("[INFO] No oos quarters (empty).")

    # ---------- Manifest ----------
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_dir": str(out_dir.resolve()),
        "window_months": window_months,
        "format": fmt,
        "quarters_requested": quarters,
        "quarters_produced": produced_quarters,
        "pooled": want_pooled,
        "pooled_until": pooled_until,
        "config_snapshot": cfg,
        "elapsed_sec": round(time.time() - t0, 2),
    }
    (window_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
