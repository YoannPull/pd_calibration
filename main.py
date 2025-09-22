# main.py
import yaml
from pathlib import Path
import pandas as pd
from src.features.labels import build_default_labels

def qpaths(root: Path, q: str):
    d = root / f"historical_data_{q}"
    return d / f"historical_data_{q}.txt", d / f"historical_data_time_{q}.txt"

def main():
    cfg = yaml.safe_load(open("config.yml"))
    root = Path(cfg["data"]["root"])

    # quarters à traiter (liste) ; sinon on prend les chemins explicites orig/perf
    quarters = cfg["data"].get("quarters", [])
    out_dir = Path(cfg["output"].get("dir", "data/processed"))
    out_dir.mkdir(parents=True, exist_ok=True)

    label_cfg = cfg["labels"]
    pooled = []  # pour concaténer si on veut un CSV poolé

    if quarters:
        for q in quarters:
            orig, perf = qpaths(root, q)
            if not orig.exists() or not perf.exists():
                print(f" Skipping {q}: files not found")
                continue

            df = build_default_labels(
                path_orig=str(orig),
                path_perf=str(perf),
                window_months=label_cfg["window_months"],
                delinquency_threshold=label_cfg.get("delinquency_threshold", 3),
                liquidation_codes=tuple(label_cfg.get("liquidation_codes", ["02","03","09"])),
                include_ra=label_cfg.get("include_ra", True),
                require_full_window=label_cfg.get("require_full_window", False),
            )

            # Ajoute une colonne vintage (quarter) utile pour les splits
            fpd = df["first_payment_date"]  # period[M]
            qnum = ((fpd.dt.month - 1) // 3 + 1).astype("int")
            df["vintage"] = fpd.dt.year.astype("string") + "Q" + qnum.astype("string")

            out_path = out_dir / f"default_labels_{label_cfg['window_months']}m_{q}.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved {q} -> {out_path}  shape={df.shape}")

            if cfg["output"].get("make_pooled", False):
                pooled.append(df)

        if pooled:
            df_pooled = pd.concat(pooled, ignore_index=True)
            pooled_path = out_dir / f"default_labels_{label_cfg['window_months']}m_pooled.csv"
            df_pooled.to_csv(pooled_path, index=False)
            print(f"Saved pooled -> {pooled_path}  shape={df_pooled.shape}")

    else:
        # Mode single-file (compat avec ton ancien config)
        orig = Path(cfg["data"]["orig"])
        perf = Path(cfg["data"]["perf"])
        df = build_default_labels(
            path_orig=str(orig),
            path_perf=str(perf),
            window_months=label_cfg["window_months"],
            delinquency_threshold=label_cfg.get("delinquency_threshold", 3),
            liquidation_codes=tuple(label_cfg.get("liquidation_codes", ["02","03","09"])),
            include_ra=label_cfg.get("include_ra", True),
            require_full_window=label_cfg.get("require_full_window", False),
        )
        fpd = df["first_payment_date"]
        qnum = ((fpd.dt.month - 1) // 3 + 1).astype("int")
        df["vintage"] = fpd.dt.year.astype("string") + "Q" + qnum.astype("string")

        out_path = Path(cfg["output"]["path"]).with_suffix(".csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f" Saved -> {out_path}  shape={df.shape}")

if __name__ == "__main__":
    main()
