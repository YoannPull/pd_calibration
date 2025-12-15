#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recalibration "type (1)" = update de la table grade -> PD
- mean     : PD brute par grade
- isotonic : PD monotone lissée par grade (IsotonicRegression)

Supporte:
- agrégation pooled     : sum(bad)/sum(count) par grade
- agrégation time_mean  : moyenne arithmétique des DR_{grade, vintage} (par grade),
                          utile si tu veux "moyenner dans le temps"

Optionnel:
- filtre fenêtre glissante: --window-years 5 (nécessite time_col)
- ou bornes: --vintage-start / --vintage-end

Sortie:
- un JSON du style bucket_stats "train" (liste de records par grade)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression


def load_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def parse_vintage_to_period(s: pd.Series, freq: str = "Q") -> pd.PeriodIndex:
    """
    Convertit une colonne vintage en PeriodIndex.
    Supporte:
      - '2015Q1', '2015-Q1'
      - dates parseables ('2015-03-31', etc.) -> conversion en trimestre
      - Period déjà présent
    """
    if isinstance(s.dtype, pd.PeriodDtype):
        return s.astype("period[Q]").array

    s_str = s.astype(str)

    # Cas type 'YYYYQn' ou 'YYYY-Qn'
    m = s_str.str.match(r"^\d{4}[- ]?Q[1-4]$")
    if m.all():
        cleaned = s_str.str.replace(" ", "", regex=False).str.replace("-", "", regex=False)
        # '2015Q1' ok pour PeriodIndex(freq='Q')
        return pd.PeriodIndex(cleaned, freq="Q")

    # Sinon on tente datetime
    dt = pd.to_datetime(s_str, errors="coerce", utc=False)
    if dt.notna().mean() < 0.95:
        bad = s_str[dt.isna()].head(5).tolist()
        raise ValueError(
            f"Impossible de parser vintage en dates/quarters. Exemples non parseables: {bad}"
        )

    if freq.upper().startswith("Q"):
        return dt.dt.to_period("Q")
    if freq.upper().startswith("M"):
        return dt.dt.to_period("M")
    raise ValueError(f"freq non supportée: {freq}")


def ensure_grade(
    df: pd.DataFrame,
    grade_col: str,
    score_col: str | None,
    buckets_json: str | None,
    n_buckets: int = 10,
) -> pd.DataFrame:
    if grade_col in df.columns:
        return df

    if score_col is None or buckets_json is None:
        raise ValueError(
            f"Colonne '{grade_col}' absente. Fournis --score-col et --buckets pour reconstruire les grades."
        )

    edges = json.loads(Path(buckets_json).read_text(encoding="utf-8"))["edges"]
    scores = df[score_col].to_numpy()

    raw_grade = np.digitize(scores, np.array(edges)[1:], right=True) + 1
    grade = n_buckets + 1 - raw_grade  # même convention que ton train_model.py

    out = df.copy()
    out[grade_col] = grade.astype(int)
    return out


def compute_pd_table(
    df: pd.DataFrame,
    target_col: str,
    grade_col: str,
    method: str,
    aggregation: str,
    time_col: str | None,
    time_freq: str,
    smooth: float,
    score_col: str | None,
) -> dict:
    # Base stats par obs
    d = df.copy()
    d[target_col] = d[target_col].astype(int)
    d[grade_col] = d[grade_col].astype(int)

    # Option score ranges
    has_score = score_col is not None and score_col in d.columns

    if aggregation == "pooled":
        g = d.groupby(grade_col, as_index=False).agg(
            count=(target_col, "size"),
            bad=(target_col, "sum"),
        )
        g["pd_raw"] = (g["bad"] + smooth) / (g["count"] + 2 * smooth)

        if has_score:
            score_stats = d.groupby(grade_col).agg(
                min_score=(score_col, "min"),
                max_score=(score_col, "max"),
            ).reset_index()
            g = g.merge(score_stats, on=grade_col, how="left")
        else:
            g["min_score"] = np.nan
            g["max_score"] = np.nan

    elif aggregation == "time_mean":
        if time_col is None:
            raise ValueError("aggregation=time_mean nécessite --time-col.")

        d["_period"] = parse_vintage_to_period(d[time_col], freq=time_freq)

        gt = d.groupby([grade_col, "_period"], as_index=False).agg(
            count=(target_col, "size"),
            bad=(target_col, "sum"),
        )
        gt["dr"] = (gt["bad"] + smooth) / (gt["count"] + 2 * smooth)

        # moyenne arithmétique des DR dans le temps
        g = gt.groupby(grade_col, as_index=False).agg(
            n_periods=("_period", "nunique"),
            count=("count", "sum"),   # volume total (utile pour pondérer isotonic)
            bad=("bad", "sum"),
            pd_raw=("dr", "mean"),
        )

        if has_score:
            score_stats = d.groupby(grade_col).agg(
                min_score=(score_col, "min"),
                max_score=(score_col, "max"),
            ).reset_index()
            g = g.merge(score_stats, on=grade_col, how="left")
        else:
            g["min_score"] = np.nan
            g["max_score"] = np.nan
    else:
        raise ValueError(f"aggregation non supportée: {aggregation}")

    g = g.sort_values(grade_col).reset_index(drop=True)

    # Isotonic smoothing sur les PD de grade (pondérée par les volumes)
    if method == "mean":
        g["pd"] = g["pd_raw"]
    elif method == "isotonic":
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
        x = g[grade_col].to_numpy(dtype=float)
        y = g["pd_raw"].to_numpy(dtype=float)
        w = g["count"].to_numpy(dtype=float)  # poids = volumes par grade
        g["pd"] = iso.fit_transform(x, y, sample_weight=w)
    else:
        raise ValueError(f"method non supportée: {method}")

    # Monotonicité check (PD doit croître avec grade)
    mono = bool(np.all(np.diff(g["pd"].to_numpy()) >= -1e-15))

    # Payload compatible "bucket_stats.json" (style train_model.py)
    recs = []
    for _, r in g.iterrows():
        recs.append({
            "bucket": int(r[grade_col]),
            "count": int(r["count"]),
            "bad": int(r["bad"]),
            "min_score": None if pd.isna(r["min_score"]) else float(r["min_score"]),
            "max_score": None if pd.isna(r["max_score"]) else float(r["max_score"]),
            "pd_raw": float(r["pd_raw"]),
            "pd": float(r["pd"]),
        })

    payload = {
        "method": method,
        "aggregation": aggregation,
        "target": target_col,
        "grade_col": grade_col,
        "time_col": time_col,
        "time_freq": time_freq,
        "smooth": smooth,
        "n_obs": int(len(d)),
        "monotone_pd": mono,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "train": recs,
    }
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scored", action="append", required=True,
                   help="Chemin vers un fichier *_scored (parquet/csv). Peut être répété.")
    p.add_argument("--target", required=True)
    p.add_argument("--grade-col", default="grade")
    p.add_argument("--time-col", default=None)
    p.add_argument("--time-freq", default="Q", choices=["Q", "M"])
    p.add_argument("--method", default="mean", choices=["mean", "isotonic"])
    p.add_argument("--aggregation", default="pooled", choices=["pooled", "time_mean"])
    p.add_argument("--smooth", type=float, default=0.0, help="Pseudo-comptes (0.5 = Jeffreys-like).")
    p.add_argument("--out-json", required=True)

    # Fenêtre glissante / filtres temps
    p.add_argument("--window-years", type=int, default=None,
                   help="Garde les X dernières années (nécessite --time-col).")
    p.add_argument("--vintage-start", default=None, help="Ex: 2015Q1 (ou date).")
    p.add_argument("--vintage-end", default=None, help="Ex: 2024Q4 (ou date).")

    # Si grade pas présent
    p.add_argument("--score-col", default="score_ttc")
    p.add_argument("--buckets", default=None, help="risk_buckets.json pour reconstruire les grades si besoin.")
    p.add_argument("--n-buckets", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()

    dfs = [load_any(p) for p in args.scored]
    df = pd.concat(dfs, axis=0, ignore_index=True)

    df = ensure_grade(
        df,
        grade_col=args.grade_col,
        score_col=args.score_col,
        buckets_json=args.buckets,
        n_buckets=args.n_buckets,
    )

    # Filtrage fenêtre
    if args.time_col is not None and (args.window_years or args.vintage_start or args.vintage_end):
        per = parse_vintage_to_period(df[args.time_col], freq=args.time_freq)
        df = df.copy()
        df["_period"] = per

        if args.vintage_start:
            p0 = parse_vintage_to_period(pd.Series([args.vintage_start]), freq=args.time_freq)[0]
            df = df[df["_period"] >= p0]

        if args.vintage_end:
            p1 = parse_vintage_to_period(pd.Series([args.vintage_end]), freq=args.time_freq)[0]
            df = df[df["_period"] <= p1]

        if args.window_years:
            # on garde les derniers X*freq periods
            maxp = df["_period"].max()
            if args.time_freq == "Q":
                keep = args.window_years * 4
            else:
                keep = args.window_years * 12
            minp = (maxp - keep + 1)
            df = df[df["_period"] >= minp]

        df = df.drop(columns=["_period"], errors="ignore")

    payload = compute_pd_table(
        df=df,
        target_col=args.target,
        grade_col=args.grade_col,
        method=args.method,
        aggregation=args.aggregation,
        time_col=args.time_col,
        time_freq=args.time_freq,
        smooth=args.smooth,
        score_col=args.score_col,
    )

    outp = Path(args.out_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"✔ Recalibration écrite dans: {outp}")
    print(f"  - method={payload['method']} aggregation={payload['aggregation']} monotone={payload['monotone_pd']}")


if __name__ == "__main__":
    main()
