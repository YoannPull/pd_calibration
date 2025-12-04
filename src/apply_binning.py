#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path

import pandas as pd

from config.config import load_config
from features.binning import load_bins_json, transform_with_learned_bins

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Appliquer des bins déjà appris sur un jeu de données brut "
            "(par exemple pour produire un jeu binned OOS)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("data", type=str, help="Chemin vers le fichier CSV de données à binner")
    parser.add_argument("bins", type=str, help="Chemin vers le fichier JSON contenant les bins appris")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Chemin du fichier de configuration YAML",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data_binned.csv",
        help="Chemin du fichier de sortie (CSV binned)",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        help="Niveau de log (DEBUG, INFO, WARNING, ERROR)",
    )

    return parser.parse_args(argv)


def configure_logging(log_level: str) -> None:
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Niveau de log invalide : {log_level}")
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    )


def main(argv=None):
    args = parse_args(argv)
    configure_logging(args.log_level)

    logger.info("Démarrage d'apply_binning avec les arguments : %s", args)

    # 1) Chargement config
    config = load_config(args.config)
    id_column = config["ID_CLIENT"]

    # 2) Chargement des données
    df = pd.read_csv(args.data, sep=";", dtype={id_column: str})
    logger.info("Données chargées depuis %s avec shape=%s", args.data, df.shape)

    # 3) Chargement des bins appris
    learned = load_bins_json(args.bins)
    logger.info("Bins appris chargés depuis %s", args.bins)

    # 4) Application des bins
    df_binned = transform_with_learned_bins(df, learned)
    logger.info("Binning terminé, shape=%s", df_binned.shape)

    # 5) On met l'ID en première colonne si présent
    if id_column in df_binned.columns:
        cols = [id_column] + [c for c in df_binned.columns if c != id_column]
        df_binned = df_binned[cols]

    # 6) Sauvegarde
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_binned.to_csv(output_path, sep=";", index=False)

    logger.info("Données binned sauvegardées dans %s", output_path)


if __name__ == "__main__":
    main()