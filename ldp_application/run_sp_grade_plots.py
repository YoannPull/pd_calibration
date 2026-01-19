#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click launcher for S&P grade plots.

Expected layout:
  ldp_application/
    run_sp_grade_plots.py
    scripts/sp_grade_plots.py
    outputs/sp_grade_is_oos/sp_grade_tables_YYYY_YYYY.csv
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _detect_runner(prefer_poetry: bool) -> list[str]:
    root = _project_root()
    has_pyproject = (root / "pyproject.toml").exists()
    has_poetry = shutil.which("poetry") is not None
    if prefer_poetry and has_pyproject and has_poetry:
        return ["poetry", "run", "python"]
    return [sys.executable]


def _find_latest_tables_csv(outdir: Path) -> Path | None:
    cands = sorted(outdir.glob("sp_grade_tables_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def main() -> int:
    root = _project_root()
    script_path = (root / "scripts" / "sp_grade_plots.py").resolve()

    default_out = root / "outputs" / "sp_grade_is_oos"
    default_tables = _find_latest_tables_csv(default_out)

    parser = argparse.ArgumentParser(description="Run S&P grade plots in one command.")
    parser.add_argument("--prefer-poetry", action="store_true")
    parser.add_argument("--no-poetry", action="store_true")

    parser.add_argument("--tables-csv", type=str, default=str(default_tables) if default_tables else "")
    parser.add_argument("--outdir", type=str, default=str(default_out / "plots_timeseries"))
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--p-max", type=float, default=None)
    parser.add_argument("--min-total-n", type=int, default=1)
    parser.add_argument("--ttc-label", type=str, default=None)

    args = parser.parse_args()

    if not script_path.exists():
        print(f"ERROR: cannot find script: {script_path}", file=sys.stderr)
        return 2

    if not args.tables_csv:
        print(f"ERROR: no tables csv found in {default_out}. Run tables first.", file=sys.stderr)
        return 2

    tables_csv = Path(args.tables_csv).expanduser().resolve()
    if not tables_csv.exists():
        print(f"ERROR: tables CSV not found: {tables_csv}", file=sys.stderr)
        return 2

    prefer_poetry = bool(args.prefer_poetry) and (not args.no_poetry)
    runner = _detect_runner(prefer_poetry=prefer_poetry)

    cmd = runner + [str(script_path)]
    cmd += ["--tables-csv", str(tables_csv)]
    cmd += ["--outdir", str(Path(args.outdir).expanduser())]
    cmd += ["--confidence-level", str(args.confidence_level)]
    cmd += ["--min-total-n", str(args.min_total_n)]
    if args.p_max is not None:
        cmd += ["--p-max", str(args.p_max)]
    if args.ttc_label:
        cmd += ["--ttc-label", str(args.ttc_label)]

    print("Running command:\n  " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: plots run failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
