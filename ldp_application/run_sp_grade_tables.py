
#!/usr/bin/env python3
from __future__ import annotations
# -*- coding: utf-8 -*-
"""
One-click launcher for S&P grade tables (snapshot input).

Expected layout:
  ldp_application/
    run_sp_grade_tables.py
    scripts/sp_grade_tables.py
    data/processed/sp_corporate_monthly.csv  (or .xlsx)
"""

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


def main() -> int:
    root = _project_root()
    script_path = (root / "scripts" / "sp_grade_tables.py").resolve()

    parser = argparse.ArgumentParser(description="Run S&P grade tables in one command (snapshot input).")
    parser.add_argument("--prefer-poetry", action="store_true")
    parser.add_argument("--no-poetry", action="store_true")

    # pass-through args
    parser.add_argument(
        "--file",
        type=str,
        default=str(root / "data" / "processed" / "sp_corporate_monthly.csv"),
        help="Snapshot file (.csv or .xlsx) in corporate.xlsx-like schema",
    )
    parser.add_argument("--outdir", type=str, default=str(root / "outputs" / "sp_grade_is_oos"))
    parser.add_argument("--agency", type=str, default="Standard & Poor's Ratings Services")
    parser.add_argument("--horizon-months", type=int, default=12)
    parser.add_argument("--default-rating", type=str, default="D")
    parser.add_argument("--confidence-level", type=float, default=0.95)

    parser.add_argument("--is-start-year", type=int, default=2010)
    parser.add_argument("--is-end-year", type=int, default=2018)
    parser.add_argument("--oos-start-year", type=int, default=2019)
    parser.add_argument("--oos-end-year", type=int, default=2020)

    parser.add_argument("--grade-whitelist", type=str, nargs="*", default=None)

    parser.add_argument("--ttc-source", type=str, default="is", choices=["is", "sp2012"])
    parser.add_argument(
        "--ttc-estimator",
        type=str,
        default="pooled",
        choices=["pooled", "jeffreys_mean", "pooled_if_nonzero_else_jeffreys"],
    )
    parser.add_argument("--drop-grades-without-ttc", action="store_true")
    parser.add_argument("--keep-grades-without-ttc", action="store_true")

    # year handling in snapshot
    parser.add_argument("--prefer-year-column", action="store_true")
    parser.add_argument("--no-prefer-year-column", action="store_true")

    args = parser.parse_args()

    if not script_path.exists():
        print(f"ERROR: cannot find script: {script_path}", file=sys.stderr)
        return 2

    prefer_poetry = bool(args.prefer_poetry) and (not args.no_poetry)
    runner = _detect_runner(prefer_poetry=prefer_poetry)

    cmd = runner + [str(script_path)]
    cmd += ["--file", str(Path(args.file).expanduser())]
    cmd += ["--outdir", str(Path(args.outdir).expanduser())]
    cmd += ["--agency", args.agency]
    cmd += ["--horizon-months", str(args.horizon_months)]
    cmd += ["--default-rating", args.default_rating]
    cmd += ["--confidence-level", str(args.confidence_level)]
    cmd += ["--is-start-year", str(args.is_start_year)]
    cmd += ["--is-end-year", str(args.is_end_year)]
    cmd += ["--oos-start-year", str(args.oos_start_year)]
    cmd += ["--oos-end-year", str(args.oos_end_year)]
    cmd += ["--ttc-source", args.ttc_source]
    cmd += ["--ttc-estimator", args.ttc_estimator]

    if args.grade_whitelist:
        cmd += ["--grade-whitelist"] + list(args.grade_whitelist)
    if args.drop_grades_without_ttc:
        cmd += ["--drop-grades-without-ttc"]
    if args.keep_grades_without_ttc:
        cmd += ["--keep-grades-without-ttc"]

    if args.prefer_year_column and (not args.no_prefer_year_column):
        cmd += ["--prefer-year-column"]
    if args.no_prefer_year_column:
        cmd += ["--no-prefer-year-column"]

    print("Running command:\n  " + " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: tables run failed with exit code {e.returncode}", file=sys.stderr)
        return e.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
