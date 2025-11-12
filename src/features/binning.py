# src/features/binning.py
# -*- coding: utf-8 -*-

from __future__ import annotations
import json, math, warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Parallélisation
try:
    from joblib import Parallel, delayed
except Exception:  # fallback si joblib indisponible
    Parallel = None
    def delayed(f): return f  # type: ignore

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*is_period_dtype is deprecated.*")

# =============================
# Réglages "pipeline.py"-like
# =============================
DENYLIST_STRICT_DEFAULT = [
    "first_payment_date",       # proxy temporel
    "maturity_date",            # proxy temporel redondant
    "vintage",                  # proxy temporel
    "mi_cancellation_indicator" # post-événement → fuite
]

EXCLUDE_IDS_DEFAULT: Tuple[str, ...] = (
    "loan_sequence_number", "postal_code", "seller_name", "servicer_name", "msa_md"
)

# -----------------------------
# Utilitaires généraux
# -----------------------------
def gini_trapz(df_cum, y_col="bad_client_share_cumsum", x_col="good_client_share_cumsum", signed=False):
    df = df_cum[[x_col, y_col]].astype(float).copy().sort_values(x_col)
    df[x_col] = df[x_col].clip(0, 1)
    df[y_col] = df[y_col].clip(0, 1)
    if df[x_col].iloc[0] > 0 or df[y_col].iloc[0] > 0:
        df = pd.concat([pd.DataFrame({x_col: [0.0], y_col: [0.0]}), df], ignore_index=True)
    if df[x_col].iloc[-1] < 1 - 1e-12 or df[y_col].iloc[-1] < 1 - 1e-12:
        df = pd.concat([df, pd.DataFrame({x_col: [1.0], y_col: [1.0]})], ignore_index=True)
    area = np.trapezoid(df[y_col].to_numpy(), df[x_col].to_numpy()) if hasattr(np, "trapezoid") else np.trapz(df[y_col].to_numpy(), df[x_col].to_numpy())
    g = 1 - 2 * area
    return g if signed else abs(g)

def _is_period_dtype(dt):  # compat
    try:
        return pd.api.types.is_period_dtype(dt)
    except Exception:
        return False

def to_float_series(s: pd.Series) -> pd.Series:
    if _is_period_dtype(s.dtype):
        ts = s.dt.to_timestamp(how="start")
        days = (ts.astype("int64") // 86_400_000_000_000)
        return days.astype("float64")
    if pd.api.types.is_datetime64_any_dtype(s):
        s_dt = s
        try:
            if getattr(s_dt.dt, "tz", None) is not None:
                s_dt = s_dt.dt.tz_convert(None)
        except Exception:
            pass
        s_dt = s_dt.astype("datetime64[ns]")
        days = (s_dt.astype("int64") // 86_400_000_000_000)
        return days.astype("float64")
    return pd.to_numeric(s, errors="coerce").astype("float64")

# -----------------------------
# Helpers pré-traitements
# -----------------------------
def drop_missing_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    mask = cols.str.startswith("was_missing_") | cols.str.endswith("_missing")
    to_drop = cols[mask].tolist()
    if to_drop:
        return df.drop(columns=to_drop, errors="ignore")
    return df

def apply_denylist(df: pd.DataFrame, denylist: List[str]) -> pd.DataFrame:
    if not denylist:
        return df
    return df.drop(columns=[c for c in denylist if c in df.columns], errors="ignore")

# -----------------------------
# Dé-one-hot + denylist
# -----------------------------
def detect_onehot_groups(df, exclude_cols=None, exclusivity_thr=0.95):
    exclude = set(exclude_cols or [])
    groups = {}
    for c in df.columns:
        if c in exclude or "_" not in c:
            continue
        base, lab = c.rsplit("_", 1)
        s = df[c]
        is_ohe = (pd.api.types.is_bool_dtype(s) or (pd.api.types.is_numeric_dtype(s) and s.dropna().isin([0, 1]).all()))
        if is_ohe:
            groups.setdefault(base, []).append((c, lab))
    clean = {}
    for base, items in groups.items():
        cols = [c for c, _ in items]
        vals = df[cols].apply(pd.to_numeric, errors="coerce")
        row_sum = vals.fillna(0).astype("Int64").sum(axis=1)
        excl_rate = float(((row_sum <= 1) | row_sum.isna()).mean())
        if excl_rate >= exclusivity_thr:
            clean[base] = items
    return clean

def deonehot(df, exclude_cols=None, ambiguous_label=None):
    groups = detect_onehot_groups(df, exclude_cols=exclude_cols)
    out = df.copy()
    for base, items in groups.items():
        cols = [c for c, _ in items]
        labels = [lab for _, lab in items]
        gvals = df[cols].apply(pd.to_numeric, errors="coerce")
        row_sum = gvals.fillna(0).astype("Int64").sum(axis=1)
        ser = pd.Series(pd.NA, index=df.index, dtype="object")
        for c, lab in zip(cols, labels):
            ser[df[c] == 1] = (pd.NA if lab == "<NA>" else lab)
        amb = row_sum > 1
        if amb.any():
            ser[amb] = ambiguous_label if ambiguous_label is not None else pd.NA
        out[base] = ser.astype("category")
        out.drop(columns=cols, inplace=True, errors="ignore")
    return out

# -----------------------------
# Catégorielles : max |Gini| par fusion
# -----------------------------
def _cat_stats(df, col, target_col, include_missing=True, missing_label="__MISSING__"):
    y = df[target_col].astype(int)
    s = df[col]
    if include_missing:
        s = s.astype("object").where(s.notna(), missing_label)
    tmp = pd.DataFrame({col: s, target_col: y})
    agg = tmp.groupby(col, dropna=not include_missing)[target_col].agg(["sum", "count"])
    agg.rename(columns={"sum": "n_bad", "count": "n_total"}, inplace=True)
    agg["n_good"] = agg["n_total"] - agg["n_bad"]
    n_bad = int(y.sum())
    n_good = int(len(y) - y.sum())
    denom_bad = n_bad if n_bad > 0 else 1
    denom_good = n_good if n_good > 0 else 1
    agg["bad_rate"] = agg["n_bad"] / agg["n_total"].where(agg["n_total"] > 0, 1)
    agg["bad_share"] = agg["n_bad"] / denom_bad
    agg["good_share"] = agg["n_good"] / denom_good
    return agg.reset_index().rename(columns={col: "modality"})

def _groups_df_from_bins(stats_df, bins):
    rows = []
    for i, mods in enumerate(bins):
        sub = stats_df[stats_df["modality"].isin(mods)]
        n_bad = int(sub["n_bad"].sum())
        n_good = int(sub["n_good"].sum())
        n_tot = int(sub["n_total"].sum())
        br = n_bad / n_tot if n_tot > 0 else 0.0
        rows.append(
            {
                "bin_id": i,
                "modalities": tuple(mods),
                "n_total": n_tot,
                "n_bad": n_bad,
                "n_good": n_good,
                "bad_rate": br,
                "bad_share": sub["bad_share"].sum(),
                "good_share": sub["good_share"].sum(),
            }
        )
    gdf = pd.DataFrame(rows).sort_values("bad_rate", ascending=True, kind="mergesort").reset_index(drop=True)
    gdf["bad_cum"] = gdf["bad_share"].cumsum()
    gdf["good_cum"] = gdf["good_share"].cumsum()
    return gdf

def _gini_from_bins(stats_df, bins):
    gdf = _groups_df_from_bins(stats_df, bins)
    df_cum = gdf.rename(
        columns={"good_cum": "good_client_share_cumsum", "bad_cum": "bad_client_share_cumsum"}
    )[["good_client_share_cumsum", "bad_client_share_cumsum"]]
    return gini_trapz(df_cum)

def maximize_gini_categorical(
    df,
    col,
    target_col,
    include_missing=True,
    missing_label="__MISSING__",
    max_bins=6,
    min_bin_size=200,
    min_bin_frac=None,
    ordered=False,
    explicit_order=None,
):
    stats_df = _cat_stats(df, col, target_col, include_missing, missing_label)
    if ordered and explicit_order is not None:
        order = [m for m in explicit_order if m in set(stats_df["modality"])]
        order += [m for m in stats_df["modality"] if m not in set(order)]
    else:
        order = list(stats_df.sort_values("bad_rate")["modality"])
    groups = [[m] for m in order]
    if len(groups) <= 1:
        mapping = {m: 0 for m in order}
        g = _gini_from_bins(stats_df, groups)
        return {
            "mapping": mapping,
            "gini_before": float(g),
            "gini_after": float(g),
            "bins": [tuple(grp) for grp in groups],
        }

    n_total = int(stats_df["n_total"].sum())
    need = 0
    if min_bin_frac is not None:
        need = max(need, math.ceil(float(min_bin_frac) * max(n_total, 1)))
    if min_bin_size is not None:
        need = max(need, int(min_bin_size))

    def ok(gs):
        if max_bins is not None and len(gs) > max_bins:
            return False
        if need:
            for mods in gs:
                if int(stats_df[stats_df["modality"].isin(mods)]["n_total"].sum()) < need:
                    return False
        return True

    def reorder(gs):
        if ordered:
            return gs

        def br(mods):
            sub = stats_df[stats_df["modality"].isin(mods)]
            nb, nt = sub["n_bad"].sum(), sub["n_total"].sum()
            return (nb / nt) if nt > 0 else 0.0

        return sorted(gs, key=br)

    while not ok(groups):
        best_g, best_i = -np.inf, None
        for i in range(len(groups) - 1):
            merged = groups[:i] + [groups[i] + groups[i + 1]] + groups[i + 2 :]
            merged = reorder(merged)
            g_try = _gini_from_bins(stats_df, merged)
            if g_try > best_g:
                best_g, best_i = g_try, i
        if best_i is None:
            best_i = 0
        groups = groups[:best_i] + [groups[best_i] + groups[best_i + 1]] + groups[best_i + 2 :]
        groups = reorder(groups)

    g_before = _gini_from_bins(stats_df, [[m] for m in order])
    g_after = _gini_from_bins(stats_df, groups)
    bins = [tuple(mods) for mods in groups]
    mapping = {m: i for i, mods in enumerate(bins) for m in mods}
    return {"mapping": mapping, "gini_before": float(g_before), "gini_after": float(g_after), "bins": bins}

# -----------------------------
# Numériques : seuils quantiles (candidats) pour max |Gini|
# -----------------------------
def _safe_edges_for_cut(edges, s_float):
    e = np.array(edges, dtype="float64")
    for i in range(1, len(e)):
        if not (e[i] > e[i - 1]):
            e[i] = np.nextafter(e[i - 1], np.inf)
    arr = s_float.to_numpy()
    if arr.size == 0:
        return e
    s_min = float(np.nanmin(arr)) if np.isfinite(arr).any() else -1.0
    s_max = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    if len(e) >= 2:
        e[0] = min(e[1] - 1e-6 * (abs(e[1]) + 1.0), s_min - 1e-6 * (abs(e[1]) + 1.0))
        e[-1] = max(e[-2] + 1e-6 * (abs(e[-2]) + 1.0), s_max + 1e-6 * (abs(e[-2]) + 1.0))
    return e

def _gini_from_numeric_bins(y_int, x_float, edges, include_missing=True):
    y = y_int.astype(int).to_numpy()
    x = x_float.to_numpy()
    K = len(edges) - 1
    idx = np.digitize(x, edges[1:-1], right=True)
    n_bad = int(y.sum())
    n_good = int(len(y) - y.sum())
    denom_bad = n_bad if n_bad > 0 else 1
    denom_good = n_good if n_good > 0 else 1
    rows = []
    for k in range(K):
        m = (idx == k) & ~np.isnan(x)
        nk = int(m.sum())
        nb = int(y[m].sum())
        ng = nk - nb
        br = nb / nk if nk > 0 else 0.0
        rows.append({"bin": k, "n_total": nk, "n_bad": nb, "n_good": ng, "bad_rate": br})
    if include_missing and np.isnan(x).any():
        m = np.isnan(x)
        nk = int(m.sum())
        nb = int(y[m].sum())
        ng = nk - nb
        br = nb / nk if nk > 0 else 0.0
        rows.append({"bin": K, "n_total": nk, "n_bad": nb, "n_good": ng, "bad_rate": br})
    gdf = pd.DataFrame(rows)
    if gdf.empty:
        return 0.0, gdf
    gdf["bad_share"] = gdf["n_bad"] / denom_bad
    gdf["good_share"] = gdf["n_good"] / denom_good
    gdf = gdf.sort_values("bad_rate").reset_index(drop=True)
    gdf["bad_cum"] = gdf["bad_share"].cumsum()
    gdf["good_cum"] = gdf["good_share"].cumsum()
    df_cum = gdf.rename(
        columns={"good_cum": "good_client_share_cumsum", "bad_cum": "bad_client_share_cumsum"}
    )[["good_client_share_cumsum", "bad_client_share_cumsum"]]
    return gini_trapz(df_cum), gdf

def maximize_gini_numeric(
    df,
    col,
    target_col,
    max_bins=6,
    min_bin_size=200,
    min_bin_frac=None,
    n_quantiles=50,
    q_low=0.02,
    q_high=0.98,
    include_missing=True,
    min_gain=1e-5,
):
    s = to_float_series(df[col])
    y = df[target_col].astype(int)
    # Cas dégénéré
    if s.dropna().nunique() < 2:
        s_f = s[np.isfinite(s)]
        if s_f.empty:
            e_cut = np.array([-1.0, 1.0], dtype="float64")
        else:
            lo, hi = float(np.nanmin(s_f)), float(np.nanmax(s_f))
            eps = 1e-6 * (abs(lo) + abs(hi) + 1.0)
            e_cut = np.array([lo - eps, hi + eps], dtype="float64")
        g0, _ = _gini_from_numeric_bins(y, s, [-np.inf, np.inf], include_missing)
        return {"edges": [-np.inf, np.inf], "edges_for_cut": e_cut, "gini_before": float(g0), "gini_after": float(g0)}

    qs = np.linspace(q_low, q_high, n_quantiles)
    cand_vals = np.unique(s.quantile(qs).dropna().values)
    edges = [-np.inf, np.inf]

    n = len(s)
    need = 0
    if min_bin_frac is not None:
        need = max(need, math.ceil(float(min_bin_frac) * max(n, 1)))
    if min_bin_size is not None:
        need = max(need, int(min_bin_size))

    def edges_ok(e):
        arr = s.to_numpy()
        idx = np.digitize(arr, e[1:-1], right=True)
        for k in range(len(e) - 1):
            if int(((idx == k) & ~np.isnan(arr)).sum()) < need:
                return False
        return True

    # baseline correcte
    g0, _ = _gini_from_numeric_bins(y, s, edges, include_missing)
    best_g = g0

    improved = True
    while improved and (len(edges) - 1) < max_bins:
        improved = False
        best_gain = min_gain
        best_t = None
        g_best = best_g
        for t in cand_vals:
            if t in edges:
                continue
            new_e = sorted([*edges, t])
            if any(np.isclose(new_e[i], new_e[i + 1]) for i in range(len(new_e) - 1)):
                continue
            if not edges_ok(new_e):
                continue
            g_try, _ = _gini_from_numeric_bins(y, s, new_e, include_missing)
            gain = g_try - best_g
            if gain > best_gain:
                best_gain, best_t, g_best = gain, t, g_try
        if best_t is not None:
            edges = sorted([*edges, best_t])
            best_g = g_best
            improved = True

    g_after, _ = _gini_from_numeric_bins(y, s, edges, include_missing)
    e = sorted(edges)
    e_cut = _safe_edges_for_cut(e, s)
    return {"edges": e, "edges_for_cut": e_cut, "gini_before": float(g0), "gini_after": float(g_after)}

# -----------------------------
# Pipeline binning complet (train) + transform (val/test)
# -----------------------------
@dataclass
class LearnedBins:
    target: str
    include_missing: bool
    missing_label: str
    bin_col_suffix: str
    cat_results: Dict[str, dict]
    num_results: Dict[str, dict]

# --- workers parallélisés
def _compute_cat_result(df_small, col, target_col, include_missing, missing_label,
                        max_bins_categ, min_bin_size_categ, min_bin_frac_categ):
    res = maximize_gini_categorical(
        df_small, col, target_col,
        include_missing=include_missing, missing_label=missing_label,
        max_bins=max_bins_categ, min_bin_size=min_bin_size_categ, min_bin_frac=min_bin_frac_categ
    )
    return col, res

def _compute_num_result(df_small, col, target_col, max_bins_num, min_bin_size_num, min_bin_frac_num,
                        n_quantiles_num, include_missing):
    res = maximize_gini_numeric(
        df_small, col, target_col,
        max_bins=max_bins_num, min_bin_size=min_bin_size_num, min_bin_frac=min_bin_frac_num,
        n_quantiles=n_quantiles_num, include_missing=include_missing
    )
    return col, res

def run_binning_maxgini_on_df(
    df: pd.DataFrame,
    target_col: str,
    include_missing: bool = True,
    missing_label: str = "__MISSING__",
    max_bins_categ: int = 6,
    min_bin_size_categ: int = 200,
    min_bin_frac_categ: Optional[float] = None,
    max_bins_num: int = 6,
    min_bin_size_num: int = 200,
    min_bin_frac_num: Optional[float] = None,
    n_quantiles_num: int = 50,
    bin_col_suffix: str = "__BIN",
    exclude_ids: Tuple[str, ...] = EXCLUDE_IDS_DEFAULT,
    min_gini_keep: Optional[float] = None,
    # nouveautés
    denylist_strict: Optional[List[str]] = None,
    drop_missing_flags: bool = False,
    n_jobs_categ: int = 1,
    n_jobs_num: int = 1,
):
    # pré-traitements "pipeline.py"-like
    DF = df.copy()
    if drop_missing_flags:
        DF = drop_missing_flag_columns(DF)
    if denylist_strict:
        DF = apply_denylist(DF, denylist_strict)

    DF = deonehot(DF, exclude_cols=[target_col])
    target = target_col

    # détecte les catégorielles raisonnables (en appliquant exclude_ids aux objets aussi)
    exclude_ids_set = set(exclude_ids or [])
    cat_cols = []
    for c in DF.columns:
        if c == target or c in exclude_ids_set:
            continue
        s = DF[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s):
            cat_cols.append(c)
        elif (pd.api.types.is_object_dtype(s) or str(s.dtype).startswith("string")) and s.nunique(dropna=True) <= 50:
            cat_cols.append(c)
        elif (pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 8 and not any(k in c.lower() for k in ["id", "sequence"])):
            cat_cols.append(c)

    # numériques (mêmes exclusions — IDs, binaires, petits entiers, clés commerciales)
    num_cols = []
    for c in DF.columns:
        if c in (cat_cols + [target]) or c in exclude_ids_set:
            continue
        s = DF[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if s.dropna().isin([0, 1]).all():
                continue
            if pd.api.types.is_integer_dtype(s) and s.dropna().nunique() <= 8:
                continue
            if any(k in c.lower() for k in ["id", "sequence", "postal", "zip", "msa", "code", "seller", "servicer"]):
                continue
            num_cols.append(c)
        elif _is_period_dtype(s.dtype) or pd.api.types.is_datetime64_any_dtype(s):
            num_cols.append(c)

    # --- binning catégoriel (parallélisé)
    cat_results = {}
    if cat_cols:
        if n_jobs_categ != 1 and Parallel is not None:
            tasks = (
                delayed(_compute_cat_result)(
                    DF[[c, target]].copy(), c, target, include_missing, missing_label,
                    max_bins_categ, min_bin_size_categ, min_bin_frac_categ
                )
                for c in cat_cols
            )
            out = Parallel(n_jobs=n_jobs_categ, backend="loky", verbose=0)(list(tasks))
            for c, res in out:
                cat_results[c] = res
        else:
            for c in cat_cols:
                res = maximize_gini_categorical(
                    DF[[c, target]].copy(), c, target,
                    include_missing=include_missing, missing_label=missing_label,
                    max_bins=max_bins_categ, min_bin_size=min_bin_size_categ, min_bin_frac=min_bin_frac_categ
                )
                cat_results[c] = res

    # --- binning numérique (parallélisé)
    num_results = {}
    if num_cols:
        if n_jobs_num != 1 and Parallel is not None:
            tasks = (
                delayed(_compute_num_result)(
                    DF[[c, target]].copy(), c, target,
                    max_bins_num, min_bin_size_num, min_bin_frac_num, n_quantiles_num, include_missing
                )
                for c in num_cols
            )
            out = Parallel(n_jobs=n_jobs_num, backend="loky", verbose=0)(list(tasks))
            for c, res in out:
                num_results[c] = res
        else:
            for c in num_cols:
                res = maximize_gini_numeric(
                    DF[[c, target]].copy(), c, target,
                    max_bins=max_bins_num, min_bin_size=min_bin_size_num, min_bin_frac=min_bin_frac_num,
                    n_quantiles=n_quantiles_num, include_missing=include_missing
                )
                num_results[c] = res

    # Ajoute colonnes __BIN
    enriched = DF.copy()
    for c, r in cat_results.items():
        s = enriched[c].astype("object").where(enriched[c].notna(), missing_label)
        enriched[c + bin_col_suffix] = s.map(r["mapping"]).astype("Int64")
    for c, r in num_results.items():
        s = to_float_series(enriched[c])
        e = np.array(r["edges_for_cut"], dtype="float64")
        b = pd.cut(s, bins=e, include_lowest=True, duplicates="drop").cat.codes.astype("Int64")
        if include_missing and s.isna().any():
            b = b.where(~s.isna(), -1).astype("Int64")
        enriched[c + bin_col_suffix] = b

    # option min_gini_keep → filtre effectif des colonnes sorties
    keep_vars: Optional[set] = None
    if min_gini_keep is not None:
        rows = []
        for v, info in cat_results.items():
            rows.append((v, info["gini_after"]))
        for v, info in num_results.items():
            rows.append((v, info["gini_after"]))
        summary = pd.DataFrame(rows, columns=["variable", "gini_after"])
        keep_vars = set(summary.loc[summary["gini_after"] >= float(min_gini_keep), "variable"].tolist())

    if keep_vars is None:
        keep_vars = set(list(cat_results.keys()) + list(num_results.keys()))

    bin_cols = [v + bin_col_suffix for v in keep_vars if v + bin_col_suffix in enriched.columns]
    # on ne garde que target + bins sélectionnées
    kept = bin_cols + ([target] if target in enriched.columns else [])
    df_binned = enriched[kept].copy()

    learned = LearnedBins(
        target=target,
        include_missing=include_missing,
        missing_label=missing_label,
        bin_col_suffix=bin_col_suffix,
        cat_results=cat_results,
        num_results=num_results,
    )
    return learned, enriched, df_binned

def transform_with_learned_bins(df, learned: LearnedBins) -> pd.DataFrame:
    DF = deonehot(df, exclude_cols=[learned.target] if learned.target in df.columns else None)
    suffix = learned.bin_col_suffix

    # applique bins catégoriels
    for c, r in learned.cat_results.items():
        if c not in DF.columns:
            continue
        s = DF[c].astype("object").where(DF[c].notna(), learned.missing_label)
        DF[c + suffix] = s.map(r["mapping"]).astype("Int64").fillna(-2).astype("Int64")

    # applique bins numériques
    for c, r in learned.num_results.items():
        if c not in DF.columns:
            continue
        s = to_float_series(DF[c])
        e = np.array(r["edges_for_cut"], dtype="float64")
        b = pd.cut(s, bins=e, include_lowest=True, duplicates="drop").cat.codes.astype("Int64")
        if learned.include_missing and s.isna().any():
            b = b.where(~s.isna(), -1).astype("Int64")
        DF[c + suffix] = b

    # sort un DF modèle: uniquement __BIN (+ cible si présente)
    bin_cols = [c for c in DF.columns if c.endswith(suffix)]
    keep = bin_cols + ([learned.target] if learned.target in DF.columns else [])
    model_df = DF[keep].copy()
    return model_df

# -----------------------------
# Sérialisation (robuste JSON)
# -----------------------------
def _json_default(o):
    """Convertit proprement les objets numpy/pandas pour JSON."""
    import numpy as _np
    import pandas as _pd
    if isinstance(o, (_np.floating, _np.integer)):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, _pd.Interval):
        try:
            return [float(o.left), float(o.right)]
        except Exception:
            return [o.left, o.right]
    return str(o)

def save_bins_json(learned: LearnedBins, path: str):
    d = {
        "target": learned.target,
        "include_missing": bool(learned.include_missing),
        "missing_label": str(learned.missing_label),
        "bin_col_suffix": str(learned.bin_col_suffix),
        "cat_results": learned.cat_results,
        "num_results": learned.num_results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2, default=_json_default)

def load_bins_json(path: str) -> LearnedBins:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return LearnedBins(
        target=d["target"],
        include_missing=bool(d["include_missing"]),
        missing_label=str(d["missing_label"]),
        bin_col_suffix=str(d["bin_col_suffix"]),
        cat_results=d["cat_results"],
        num_results=d["num_results"],
    )
