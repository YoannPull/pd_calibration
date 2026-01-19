#!/usr/bin/env python3
# src/run_oos_backtest.py
# -*- coding: utf-8 -*-

"""
CLI wrapper for the paper-ready OOS backtest.

Example:
    python -m run_oos_backtest \
      --oos data/processed/scored/oos_scored.parquet \
      --bucket-stats artifacts/model_from_binned/bucket_stats.json \
      --out outputs/oos_backtest \
      --pdk-target pd_ttc \
      --tl-main-mode decision
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Optional: help local execution when project isn't installed as a package
def find_project_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    markers = {"pyproject.toml", "Makefile", ".git"}
    for p in [start] + list(start.parents):
        if any((p / m).exists() for m in markers):
            return p
    return start

ROOT = find_project_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.oos_backtest import BacktestConfig, run_oos_backtest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # I/O
    p.add_argument("--oos", dest="oos_path", required=True, help="Path to loan-level OOS scored parquet.")
    p.add_argument("--bucket-stats", dest="bucket_stats_path", required=True, help="Path to bucket_stats.json.")
    p.add_argument("--bucket-section", default="train", help="Section to read in bucket_stats.json (default: train).")
    p.add_argument("--out", dest="out_dir", default="outputs/oos_backtest", help="Output directory.")
    p.add_argument("--save-pdf", action="store_true", help="Also export figures as PDF.")
    p.add_argument("--no-pdf", action="store_true", help="Disable PDF export (overrides --save-pdf).")

    # Columns
    p.add_argument("--vintage-col", default="vintage")
    p.add_argument("--grade-col", default="grade")
    p.add_argument("--default-col", default="default_12m")
    p.add_argument("--pd-loan-col", default="pd")

    # Stats
    p.add_argument("--conf-level", type=float, default=0.95)
    p.add_argument("--pdk-target", choices=["pd_ttc", "pd_hat"], default="pd_ttc")

    # PD evolution
    p.add_argument("--pd-evolution-max-grades", type=int, default=None)
    p.add_argument("--pd-plot-logy", action="store_true")
    p.add_argument("--pd-plot-linear", action="store_true")
    p.add_argument("--pd-y-units", choices=["bps", "percent"], default="bps")

    # Traffic light
    p.add_argument("--tl-main-mode", choices=["pval", "decision"], default="decision")
    p.add_argument("--tl-pval-amber", type=float, default=0.10)
    p.add_argument("--tl-prob-red", type=float, default=0.95)
    p.add_argument("--tl-prob-amber", type=float, default=0.90)
    p.add_argument("--tl-es-red", type=float, default=0.0005)
    p.add_argument("--tl-es-amber", type=float, default=0.00025)
    p.add_argument("--focus-year", type=int, default=None)

    # Beta posterior
    p.add_argument("--beta-grade", type=int, default=5)
    p.add_argument("--beta-x-points", type=int, default=1500)
    p.add_argument("--beta-quantile-span", type=float, default=0.999)
    p.add_argument("--beta-fill-alpha", type=float, default=0.10)
    p.add_argument("--beta-line-alpha", type=float, default=0.25)
    p.add_argument("--beta-year-linewidth", type=float, default=2.4)
    p.add_argument("--beta-quarter-linewidth", type=float, default=1.0)

    # Paper table
    p.add_argument("--paper-snapshot", default="2023Q4")
    p.add_argument("--paper-units", choices=["bps", "percent"], default="bps")
    p.add_argument("--paper-alpha", type=float, default=0.05)
    p.add_argument("--paper-include-counts", action="store_true")
    p.add_argument("--paper-no-counts", action="store_true")
    p.add_argument("--paper-caption", default="Jeffreys posterior diagnostics and ECB p-values (snapshot: 2023Q4)")
    p.add_argument("--paper-label", default="tab:detailed_backtest_2023Q4")
    p.add_argument("--paper-stem", default="paper_table_jeffreys_snapshot_2023Q4")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    save_pdf = bool(args.save_pdf)
    if args.no_pdf:
        save_pdf = False

    pd_plot_logy = True
    if args.pd_plot_linear:
        pd_plot_logy = False
    if args.pd_plot_logy:
        pd_plot_logy = True

    paper_include_counts = True
    if args.paper_no_counts:
        paper_include_counts = False
    if args.paper_include_counts:
        paper_include_counts = True

    cfg = BacktestConfig(
        oos_path=Path(args.oos_path),
        bucket_stats_path=Path(args.bucket_stats_path),
        bucket_section=args.bucket_section,
        out_dir=Path(args.out_dir),
        save_pdf=save_pdf,
        vintage_col=args.vintage_col,
        grade_col=args.grade_col,
        default_col=args.default_col,
        pd_loan_col=args.pd_loan_col,
        pdk_target=args.pdk_target,
        conf_level=args.conf_level,
        pd_evolution_max_grades=args.pd_evolution_max_grades,
        pd_plot_logy=pd_plot_logy,
        pd_y_units=args.pd_y_units,
        tl_main_mode=args.tl_main_mode,
        tl_pval_amber=args.tl_pval_amber,
        tl_prob_red=args.tl_prob_red,
        tl_prob_amber=args.tl_prob_amber,
        tl_es_red=args.tl_es_red,
        tl_es_amber=args.tl_es_amber,
        focus_year=args.focus_year,
        beta_grade=args.beta_grade,
        beta_x_points=args.beta_x_points,
        beta_quantile_span=args.beta_quantile_span,
        beta_fill_alpha=args.beta_fill_alpha,
        beta_line_alpha=args.beta_line_alpha,
        beta_year_linewidth=args.beta_year_linewidth,
        beta_quarter_linewidth=args.beta_quarter_linewidth,
        paper_table_snapshot=args.paper_snapshot,
        paper_table_units=args.paper_units,
        paper_table_alpha=args.paper_alpha,
        paper_table_include_counts=paper_include_counts,
        paper_table_caption=args.paper_caption,
        paper_table_label=args.paper_label,
        paper_table_stem=args.paper_stem,
    )

    run_oos_backtest(cfg)


if __name__ == "__main__":
    main()

