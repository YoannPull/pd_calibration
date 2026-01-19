#!/usr/bin/env python3
# src/apply_binning.py
"""
Apply *pre-learned* binning rules to a raw dataset.

Typical use case:
- You learn bins on an in-sample (IS) dataset (e.g., during model development),
- then you apply the exact same bins to an out-of-sample (OOS) dataset to ensure
  consistent feature preprocessing.

Inputs
------
- A raw CSV dataset (semicolon-separated).
- A JSON file containing the learned bin definitions.

Output
------
- A binned CSV dataset (semicolon-separated).
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from config.config import load_config
from features.binning import load_bins_json, transform_with_learned_bins

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Apply previously learned bins to a raw dataset "
            "(e.g., to produce a binned out-of-sample dataset)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional arguments: mandatory inputs.
    parser.add_argument("data", type=str, help="Path to the input CSV file to bin")
    parser.add_argument("bins", type=str, help="Path to the JSON file containing learned bins")

    # Optional arguments: configuration, outputs, and logging.
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_binned.csv",
        help="Path to the output (binned) CSV file",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    return parser.parse_args(argv)


def configure_logging(log_level: str) -> None:
    """
    Configure Python logging based on a user-provided level.

    Notes
    -----
    We convert the string (e.g., "INFO") into the corresponding numeric level.
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )


def main(argv=None):
    """Main CLI entry point."""
    args = parse_args(argv)
    configure_logging(args.log_level)

    logger.info("Starting apply_binning with args: %s", args)

    # 1) Load project configuration (e.g., column names, conventions).
    config = load_config(args.config)
    id_column = config["ID_CLIENT"]

    # 2) Load raw input data.
    #    - CSV is expected to be semicolon-separated.
    #    - Force the identifier column to string to preserve leading zeros, etc.
    df = pd.read_csv(args.data, sep=";", dtype={id_column: str})
    logger.info("Loaded data from %s with shape=%s", args.data, df.shape)

    # 3) Load the learned bins (produced previously on the training/IS sample).
    learned = load_bins_json(args.bins)
    logger.info("Loaded learned bins from %s", args.bins)

    # 4) Apply the learned bins to the dataset (feature transformation).
    df_binned = transform_with_learned_bins(df, learned)
    logger.info("Binning completed, output shape=%s", df_binned.shape)

    # 5) Optional: move the identifier column to the first position for readability.
    if id_column in df_binned.columns:
        cols = [id_column] + [c for c in df_binned.columns if c != id_column]
        df_binned = df_binned[cols]

    # 6) Save the transformed dataset.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_binned.to_csv(output_path, sep=";", index=False)

    logger.info("Saved binned data to %s", output_path)


if __name__ == "__main__":
    main()
