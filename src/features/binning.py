# src/features/binning.py
# -*- coding: utf-8 -*-

"""
Credit-risk-oriented binning (discretization) utilities.

This module implements a pragmatic, production-friendly binning pipeline:
- Numeric binning via greedy split search that maximizes Gini (proxy for separation).
- Optional monotonicity post-processing (merge adjacent bins until bad-rate is monotone).
- Categorical binning via bad-rate ordering + adjacent merging to reach max_bins.
- Robust handling of missing values and out-of-sample (OOS) values using infinite edges.
- JSON serialization that is resilient to NumPy scalar / array types (keys + values).

Main entry points
-----------------
- run_binning_maxgini_on_df: learn bins on a dataset and return (learned, enriched_df, final_df)
- transform_with_learned_bins: apply learned bins to a new dataset (Val/Test/OOS)
- save_bins_json / load_bins_json: persist and reload LearnedBins definitions

Conventions
-----------
- For binned columns, we create `<original_col><bin_col_suffix>` (default: "__BIN").
- Missing values are assigned bin code -1 (Int64) when include_missing=True.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional parallelization (kept for future scaling; not used by default here).
try:
    from joblib import Parallel, delayed  # noqa: F401
except ImportError:
    Parallel = None

    def delayed(f):  # type: ignore
        return f


# Silence a known pandas deprecation warning in older code paths.
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*is_period_dtype is deprecated.*",
)

# =============================================================================
# DEFAULTS / CONFIG
# =============================================================================

# Example "strict denylist" of columns that are often problematic / leakage-prone in credit datasets.
DENYLIST_STRICT_DEFAULT = [
    "first_payment_date",
    "maturity_date",
    "vintage",
    "mi_cancellation_indicator",
]

# Columns that are typically identifiers / high-cardinality keys to exclude from binning.
EXCLUDE_IDS_DEFAULT: Tuple[str, ...] = (
    "loan_sequence_number",
    "postal_code",
    "seller_name",
    "servicer_name",
    "msa_md",
)

# =============================================================================
# NUMERIC HELPERS
# =============================================================================


def gini_trapz(
    df_cum: pd.DataFrame,
    y_col: str = "bad_client_share_cumsum",
    x_col: str = "good_client_share_cumsum",
    signed: bool = False,
) -> float:
    """
    Compute a Gini index from cumulative "good" and "bad" shares using trapezoidal AUC.

    Notes
    -----
    - This function expects a cumulative curve (x: cumulative good share, y: cumulative bad share).
    - We enforce endpoints (0,0) and (1,1) for numerical stability.
    - Returned value is |Gini| by default (signed=False).
    """
    df = df_cum[[x_col, y_col]].astype(float).copy().sort_values(x_col)

    # Clip to [0, 1] to reduce numerical issues.
    df[x_col] = df[x_col].clip(0, 1)
    df[y_col] = df[y_col].clip(0, 1)

    # Ensure endpoints exist.
    if df[x_col].iloc[0] > 0 or df[y_col].iloc[0] > 0:
        df = pd.concat(
            [pd.DataFrame({x_col: [0.0], y_col: [0.0]}), df],
            ignore_index=True,
        )
    if df[x_col].iloc[-1] < 1 - 1e-12 or df[y_col].iloc[-1] < 1 - 1e-12:
        df = pd.concat(
            [df, pd.DataFrame({x_col: [1.0], y_col: [1.0]})],
            ignore_index=True,
        )

    x_vals = df[x_col].to_numpy()
    y_vals = df[y_col].to_numpy()

    # NumPy compatibility: trapezoid is newer, trapz exists historically.
    try:
        area = np.trapezoid(y_vals, x_vals)
    except AttributeError:  # pragma: no cover
        area = np.trapz(y_vals, x_vals)

    g = 1 - 2 * area
    return g if signed else abs(g)


def to_float_series(s: pd.Series) -> pd.Series:
    """
    Convert a pandas Series to float64 in a way that supports dates/periods.

    Conversion rules
    ----------------
    - object/category: attempt numeric coercion (non-numeric -> NaN)
    - period: convert to timestamp then to approximate "days" (int64 ns -> days)
    - datetime: convert to "days" (int64 ns -> days), removing timezone if present
    - otherwise: numeric coercion to float64

    Rationale
    ---------
    The binning search operates on numeric arrays. For dates, we only need an
    order-preserving numeric representation, so "days since epoch" is sufficient.
    """
    if pd.api.types.is_object_dtype(s) or str(s.dtype) == "category":
        return pd.to_numeric(s, errors="coerce").astype("float64")

    if pd.api.types.is_period_dtype(s):
        ts = s.dt.to_timestamp(how="start")
        days = ts.astype("int64") // 86_400_000_000_000
        return days.astype("float64")

    if pd.api.types.is_datetime64_any_dtype(s):
        s_dt = s
        # Remove timezone if any (best-effort).
        try:
            if getattr(s_dt.dt, "tz", None) is not None:
                s_dt = s_dt.dt.tz_convert(None)
        except Exception:
            pass
        s_dt = s_dt.astype("datetime64[ns]")
        days = s_dt.astype("int64") // 86_400_000_000_000
        return days.astype("float64")

    return pd.to_numeric(s, errors="coerce").astype("float64")


# =============================================================================
# PREPROCESSING (denylist, one-hot recombination)
# =============================================================================


def drop_missing_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop helper columns that encode missingness (e.g., was_missing_x, x_missing).

    Useful when you don't want the binning routine to "learn" on engineered flags
    that may duplicate information or create artifacts in downstream modeling.
    """
    cols = df.columns
    mask = cols.str.startswith("was_missing_") | cols.str.endswith("_missing")
    return df.drop(columns=cols[mask], errors="ignore")


def apply_denylist(df: pd.DataFrame, denylist: List[str]) -> pd.DataFrame:
    """
    Drop columns present in a denylist.

    This is commonly used to remove known leakage columns or unstable fields.
    """
    if not denylist:
        return df
    return df.drop(columns=[c for c in denylist if c in df.columns], errors="ignore")


def deonehot(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
    ambiguous_label: Optional[str] = None,  # kept for potential future behavior
) -> pd.DataFrame:
    """
    Recombine one-hot encoded columns into a single categorical feature.

    Example
    -------
    color_red, color_blue -> color in {"red","blue", NA}

    Rules
    -----
    - We only merge groups that look mutually exclusive (row sum <= 1).
    - We treat boolean columns or numeric {0,1} columns as one-hot candidates.
    - Columns listed in exclude_cols are never considered for merging.
    """
    exclude = set(exclude_cols or [])
    groups: Dict[str, List[Tuple[str, str]]] = {}

    for c in df.columns:
        if c in exclude or "_" not in c:
            continue
        base, lab = c.rsplit("_", 1)
        s = df[c]
        is_onehot_like = pd.api.types.is_bool_dtype(s) or (
            pd.api.types.is_numeric_dtype(s) and s.dropna().isin([0, 1]).all()
        )
        if is_onehot_like:
            groups.setdefault(base, []).append((c, lab))

    out = df.copy()
    for base, items in groups.items():
        cols = [c for c, _ in items]
        labels = [lab for _, lab in items]

        gvals = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        # Only merge if mutually exclusive (or mostly so).
        if gvals.sum(axis=1).max() <= 1:
            ser = pd.Series(pd.NA, index=df.index, dtype="object")
            for c, lab in zip(cols, labels):
                ser[df[c] == 1] = lab
            out[base] = ser.astype("category")
            out.drop(columns=cols, inplace=True, errors="ignore")

    return out


# =============================================================================
# MONOTONICITY (numeric post-processing)
# =============================================================================


def _check_monotonicity(stats_df: pd.DataFrame) -> bool:
    """
    Return True if bad_rate is monotone (non-decreasing OR non-increasing).

    We ignore NaN bad_rate values (should be rare; typically corresponds to empty bins).
    """
    if len(stats_df) < 2:
        return True

    rates = stats_df["bad_rate"].values
    valid_rates = rates[~np.isnan(rates)]
    if len(valid_rates) < 2:
        return True

    diffs = np.diff(valid_rates)
    is_increasing = np.all(diffs >= -1e-9)  # float tolerance
    is_decreasing = np.all(diffs <= 1e-9)
    return is_increasing or is_decreasing


def force_monotonicity_numeric(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    edges: List[float],
    include_missing: bool,
) -> List[float]:
    """
    Enforce monotonic bad-rate across numeric bins by merging adjacent bins.

    Strategy
    --------
    - Compute per-bin bad_rate under current edges.
    - If bad_rate is not monotone, identify the first "violating" change and
      remove the corresponding interior edge (merge bins).
    - Stop when monotone or when no safe merge can be found.

    Notes
    -----
    - Missing is handled as a separate bin at the end by _gini_from_numeric_bins;
      monotonicity is checked only on numeric bins (excluding missing).
    """
    current_edges = sorted(list(set(edges)))

    max_iter = len(current_edges) + 5  # guard against infinite loops
    for _ in range(max_iter):
        _, gdf = _gini_from_numeric_bins(
            df[target_col],
            to_float_series(df[col]),
            current_edges,
            include_missing,
        )

        # Only numeric bins: bins are 0..K-1, missing is bin K.
        gdf_num = gdf[gdf["bin"] < (len(current_edges) - 1)].copy()

        if _check_monotonicity(gdf_num):
            return current_edges

        rates = gdf_num["bad_rate"].values
        if len(rates) < 2:
            return current_edges

        # Determine the overall direction (up or down) from endpoints.
        trend_up = rates[-1] > rates[0]

        best_merge_idx = -1
        diffs = np.diff(rates)

        # Find the first violation given the global trend.
        for i, d in enumerate(diffs):
            if trend_up and d < 0:
                best_merge_idx = i + 1
                break
            if (not trend_up) and d > 0:
                best_merge_idx = i + 1
                break

        # Fallback: merge at the first zigzag if trend-based selection fails.
        if best_merge_idx == -1:
            for i, d in enumerate(diffs):
                if (trend_up and d < 0) or ((not trend_up) and d > 0):
                    best_merge_idx = i + 1
                    break

        # Merge by removing the offending interior edge.
        if 0 < best_merge_idx < len(current_edges) - 1:
            current_edges.pop(best_merge_idx)
        else:
            break

    return current_edges


# =============================================================================
# CORE NUMERIC BINNING (Gini maximization)
# =============================================================================


def _safe_edges_for_cut(edges: List[float], s_float: pd.Series) -> np.ndarray:
    """
    Make edges robust for pd.cut and OOS values.

    We enforce:
    - first edge = -inf
    - last edge  = +inf
    - unique, sorted edges

    If edges are degenerate, we fall back to [-inf, +inf] (single bin).
    """
    e = sorted(list(set(edges)))
    if len(e) < 2:
        return np.array([-np.inf, np.inf])

    e[0] = -np.inf
    e[-1] = np.inf
    return np.array(e, dtype="float64")


def _gini_from_numeric_bins(
    y: pd.Series,
    x: pd.Series,
    edges: List[float],
    include_missing: bool = True,
) -> Tuple[float, pd.DataFrame]:
    """
    Compute Gini for a given set of numeric bin edges, and return per-bin stats.

    Returns
    -------
    gini : float
        Absolute Gini computed from cumulative shares.
    gdf  : pd.DataFrame
        Per-bin counts and bad_rate. Missing values are optionally appended as
        a special bin with index K = (number_of_bins).
    """
    y_arr = np.array(y, dtype=int)
    x_arr = np.array(x, dtype=float)

    # Digitize using interior edges only (edges include -inf/+inf).
    idx = np.digitize(x_arr, edges[1:-1], right=True)

    K = len(edges) - 1  # number of numeric bins
    rows: List[dict] = []

    for k in range(K):
        m = (idx == k) & ~np.isnan(x_arr)
        nk = int(m.sum())
        nb = int(y_arr[m].sum()) if nk > 0 else 0
        rows.append(
            {
                "bin": k,
                "n_total": nk,
                "n_bad": nb,
                "n_good": nk - nb,
                "bad_rate": (nb / nk) if nk > 0 else 0.0,
            }
        )

    # Missing values as a dedicated bin at the end (optional).
    if include_missing:
        m_nan = np.isnan(x_arr)
        nk = int(m_nan.sum())
        if nk > 0:
            nb = int(y_arr[m_nan].sum())
            rows.append(
                {
                    "bin": K,  # special bin id
                    "n_total": nk,
                    "n_bad": nb,
                    "n_good": nk - nb,
                    "bad_rate": nb / nk,
                }
            )

    gdf = pd.DataFrame(rows)
    if gdf.empty or gdf["n_total"].sum() == 0:
        return 0.0, gdf

    n_bad_total = gdf["n_bad"].sum()
    n_good_total = gdf["n_good"].sum()
    if n_bad_total == 0 or n_good_total == 0:
        # Degenerate target distribution => no separation signal.
        return 0.0, gdf

    # Sort bins by bad_rate to build cumulative shares (ROC-like curve).
    gdf_sorted = gdf.sort_values("bad_rate").copy()
    gdf_sorted["bad_share"] = gdf_sorted["n_bad"] / n_bad_total
    gdf_sorted["good_share"] = gdf_sorted["n_good"] / n_good_total
    gdf_sorted["bad_cum"] = gdf_sorted["bad_share"].cumsum()
    gdf_sorted["good_cum"] = gdf_sorted["good_share"].cumsum()

    g = gini_trapz(gdf_sorted, y_col="bad_cum", x_col="good_cum")
    return float(g), gdf


def maximize_gini_numeric(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    max_bins: int = 6,
    min_bin_size: int = 200,
    min_bin_frac: Optional[float] = None,
    n_quantiles: int = 50,
    include_missing: bool = True,
    force_mono: bool = True,
) -> Dict[str, object]:
    """
    Learn numeric bin edges by greedily adding splits that increase Gini.

    Algorithm (greedy)
    ------------------
    1) Build candidate split points from quantiles of the feature.
    2) Start with a single bin [-inf, +inf].
    3) Iteratively add the split that yields the best Gini improvement, subject to
       min_bin_size constraints, until max_bins is reached or no improvement.

    Post-processing
    ---------------
    If force_mono=True, we merge adjacent bins until bad_rate is monotone.
    """
    s = to_float_series(df[col])
    y = df[target_col]

    # Candidate split points from quantiles (deduplicated + lightly thinned).
    qs = np.linspace(0, 1, n_quantiles + 1)
    cand_vals = np.unique(s.quantile(qs).dropna().values)

    if len(cand_vals) > 0:
        min_diff = (cand_vals.max() - cand_vals.min()) * 0.001
        cand_vals = cand_vals[np.concatenate(([True], np.diff(cand_vals) > min_diff))]

    # Initial edges: 1 bin
    edges: List[float] = [-np.inf, np.inf]
    best_g, _ = _gini_from_numeric_bins(y, s, edges, include_missing)

    n_total = len(s)
    min_size = min_bin_size
    if min_bin_frac is not None:
        min_size = max(min_size, int(n_total * min_bin_frac))

    improved = True
    while improved and (len(edges) - 1) < max_bins:
        improved = False
        best_t: Optional[float] = None
        current_best_g = best_g

        for t in cand_vals:
            if t <= edges[0] or t >= edges[-1] or t in edges:
                continue

            trial_edges = sorted(edges + [float(t)])
            g_try, gdf_try = _gini_from_numeric_bins(y, s, trial_edges, include_missing)

            # Enforce minimum bin size on numeric bins only (exclude missing bin).
            min_n = gdf_try.loc[gdf_try["bin"] < (len(trial_edges) - 1), "n_total"].min()
            if min_n < min_size:
                continue

            if g_try > current_best_g + 1e-5:
                current_best_g = g_try
                best_t = float(t)
                improved = True

        if improved and best_t is not None:
            edges = sorted(edges + [best_t])
            best_g = current_best_g

    # Optional monotonicity enforcement (merge edges).
    if force_mono and (len(edges) - 1) > 1:
        edges = force_monotonicity_numeric(df, col, target_col, edges, include_missing)
        best_g, _ = _gini_from_numeric_bins(y, s, edges, include_missing)

    e_cut = _safe_edges_for_cut(edges, s)

    return {
        "edges": edges,
        "edges_for_cut": e_cut.tolist(),
        "gini_final": float(best_g),
    }


# =============================================================================
# CATEGORICAL BINNING
# =============================================================================


def _cat_stats(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    include_missing: bool = True,
    missing_label: str = "__MISSING__",
) -> pd.DataFrame:
    """
    Compute per-modality counts and bad_rate for a categorical feature.
    """
    y = df[target_col].astype(int)
    s = df[col]

    if include_missing:
        # Cast to object to safely fill NAs; infer_objects reduces warnings in newer pandas.
        s = s.astype("object").fillna(missing_label).infer_objects(copy=False)

    tmp = pd.DataFrame({"mod": s, "tgt": y})
    agg = tmp.groupby("mod", observed=True)["tgt"].agg(["sum", "count"])
    agg.columns = ["n_bad", "n_total"]
    agg["n_good"] = agg["n_total"] - agg["n_bad"]
    agg["bad_rate"] = agg["n_bad"] / agg["n_total"]

    return agg.reset_index().rename(columns={"mod": "modality"})


def maximize_gini_categorical(
    df: pd.DataFrame,
    col: str,
    target_col: str,
    include_missing: bool = True,
    missing_label: str = "__MISSING__",
    max_bins: int = 6,
    min_bin_size: int = 200,
    min_bin_frac: Optional[float] = None,
) -> Dict[str, object]:
    """
    Learn categorical bins by ordering modalities by bad_rate and merging adjacent ones.

    Approach
    --------
    1) Compute bad_rate per modality and sort modalities by bad_rate.
    2) Start with one bin per modality.
    3) While number of bins > max_bins, merge the pair of adjacent bins whose
       bad_rate difference is smallest (heuristic).

    Output
    ------
    mapping: dict modality -> integer bin id
    bins:    list of modality lists (bin composition)
    """
    stats = _cat_stats(df, col, target_col, include_missing, missing_label)
    stats = stats.sort_values("bad_rate").reset_index(drop=True)

    # Start with each modality as its own bin.
    bins: List[List[object]] = [[m] for m in stats["modality"]]

    # Optional min size logic could be added here if you want hard constraints for categorical bins.
    # (Kept simple for now: merge until max_bins is reached.)
    _ = min_bin_size, min_bin_frac  # explicitly unused but kept in signature

    while len(bins) > max_bins:
        best_i = -1
        min_diff = float("inf")

        # Compute bad_rate for current bins.
        current_rates: List[float] = []
        for b in bins:
            sub = stats[stats["modality"].isin(b)]
            nb = sub["n_bad"].sum()
            nt = sub["n_total"].sum()
            current_rates.append(nb / nt if nt > 0 else 0.0)

        # Merge the closest adjacent rates (in the ordered list).
        for i in range(len(current_rates) - 1):
            d = abs(current_rates[i] - current_rates[i + 1])
            if d < min_diff:
                min_diff = d
                best_i = i

        if best_i != -1:
            bins[best_i] = bins[best_i] + bins[best_i + 1]
            bins.pop(best_i + 1)
        else:
            break

    mapping: Dict[object, int] = {}
    for i, mod_list in enumerate(bins):
        for m in mod_list:
            # JSON safety: convert NumPy scalars to native Python types.
            key = m.item() if hasattr(m, "item") else m
            mapping[key] = i

    return {
        "mapping": mapping,
        "bins": bins,
        "gini_final": 0.0,  # placeholder (not computed for categorical in this implementation)
    }


# =============================================================================
# MAIN PIPELINE + WRAPPERS
# =============================================================================


@dataclass
class LearnedBins:
    """
    Container for all learned binning artifacts.

    cat_results and num_results store per-feature learned parameters:
    - categorical: {"mapping": {modality -> bin_id}, "bins": [...], ...}
    - numeric:     {"edges_for_cut": [...], "edges": [...], "gini_final": ...}
    """
    target: str
    include_missing: bool
    missing_label: str
    bin_col_suffix: str
    cat_results: Dict[str, dict]
    num_results: Dict[str, dict]


def run_binning_maxgini_on_df(
    df: pd.DataFrame,
    target_col: str,
    include_missing: bool = True,
    missing_label: str = "__MISSING__",
    max_bins_categ: int = 6,
    min_bin_size_categ: int = 200,
    max_bins_num: int = 6,
    min_bin_size_num: int = 200,
    n_quantiles_num: int = 50,
    bin_col_suffix: str = "__BIN",
    exclude_ids: Tuple[str, ...] = EXCLUDE_IDS_DEFAULT,
    min_gini_keep: Optional[float] = None,
    denylist_strict: Optional[List[str]] = None,
    drop_missing_flags: bool = False,
    n_jobs_categ: int = 1,
    n_jobs_num: int = 1,
    **kwargs,
) -> Tuple[LearnedBins, pd.DataFrame, pd.DataFrame]:
    """
    Learn bins on a DataFrame and return:
    - learned: LearnedBins object (serializable)
    - enriched: original DF with added binned columns
    - final_df: only bin columns + target (handy for model training)

    Column handling
    --------------
    - Optionally drop missing-flag columns.
    - Optionally apply a strict denylist.
    - Recombine one-hot groups into single categorical features (deonehot).
    - Split features into:
        * numeric: numeric dtype AND nunique > 10
        * categorical: everything else
    """
    _ = kwargs, n_jobs_categ, n_jobs_num  # currently unused hooks

    DF = df.copy()

    if drop_missing_flags:
        DF = drop_missing_flag_columns(DF)

    if denylist_strict:
        DF = apply_denylist(DF, denylist_strict)

    # Recombine one-hot into categorical to avoid binning each dummy separately.
    DF = deonehot(DF, exclude_cols=[target_col])

    exclude_set = set(list(exclude_ids) + [target_col])
    cat_cols: List[str] = []
    num_cols: List[str] = []

    for c in DF.columns:
        if c in exclude_set:
            continue
        s = DF[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # Learn categorical bins.
    cat_results: Dict[str, dict] = {}
    for c in cat_cols:
        cat_results[c] = maximize_gini_categorical(
            DF[[c, target_col]],
            c,
            target_col,
            include_missing=include_missing,
            missing_label=missing_label,
            max_bins=max_bins_categ,
            min_bin_size=min_bin_size_categ,
        )

    # Learn numeric bins.
    num_results: Dict[str, dict] = {}
    for c in num_cols:
        num_results[c] = maximize_gini_numeric(
            DF[[c, target_col]],
            c,
            target_col,
            max_bins=max_bins_num,
            min_bin_size=min_bin_size_num,
            min_bin_frac=None,
            n_quantiles=n_quantiles_num,
            include_missing=include_missing,
            force_mono=True,
        )

    # Build enriched dataset with binned columns.
    enriched = DF.copy()

    for c, res in cat_results.items():
        s = (
            enriched[c]
            .astype("object")
            .fillna(missing_label)
            .infer_objects(copy=False)
        )
        enriched[c + bin_col_suffix] = (
            s.map(res["mapping"]).fillna(-1).astype("Int64")
        )

    for c, res in num_results.items():
        s = to_float_series(enriched[c])
        e = res["edges_for_cut"]
        b = pd.cut(s, bins=e, labels=False, include_lowest=True).astype("Int64")
        if include_missing and s.isna().any():
            b = b.fillna(-1)
        enriched[c + bin_col_suffix] = b

    # Keep only bin columns (+ target) for a compact modeling table.
    bin_cols = [c for c in enriched.columns if c.endswith(bin_col_suffix)]
    final_df = enriched[bin_cols + [target_col]].copy()

    learned = LearnedBins(
        target=target_col,
        include_missing=include_missing,
        missing_label=missing_label,
        bin_col_suffix=bin_col_suffix,
        cat_results=cat_results,
        num_results=num_results,
    )

    # Optional filtering hook (not enforced here):
    # - if you want to drop features with low separation, compute gini_final per feature
    #   and filter based on min_gini_keep. (Currently only numeric stores gini_final.)
    _ = min_gini_keep

    return learned, enriched, final_df


def transform_with_learned_bins(df: pd.DataFrame, learned: LearnedBins) -> pd.DataFrame:
    """
    Apply previously learned bins to a new dataset (Val/Test/OOS).

    Important
    ---------
    - We do not "re-learn" anything here: we reuse the stored mappings/edges.
    - OOS robustness: numeric edges are stored with [-inf, ..., +inf], so new
      extreme values still fall into a defined bin.
    - Missing values are mapped to -1 when include_missing=True.
    """
    DF = df.copy()
    suffix = learned.bin_col_suffix

    # Categorical: map modality -> bin id.
    for c, res in learned.cat_results.items():
        if c in DF.columns:
            s = (
                DF[c]
                .astype("object")
                .fillna(learned.missing_label)
                .infer_objects(copy=False)
            )
            DF[c + suffix] = s.map(res["mapping"]).fillna(-1).astype("Int64")

    # Numeric: cut using stored edges.
    for c, res in learned.num_results.items():
        if c in DF.columns:
            s = to_float_series(DF[c])
            e = res["edges_for_cut"]
            b = pd.cut(s, bins=e, labels=False, include_lowest=True).astype("Int64")
            if learned.include_missing:
                b = b.fillna(-1)
            DF[c + suffix] = b

    # Return a compact view: all bin columns (+ target if present).
    cols = [c for c in DF.columns if c.endswith(suffix)]
    if learned.target in DF.columns:
        cols.append(learned.target)

    return DF[cols]


# =============================================================================
# JSON I/O (NumPy-safe)
# =============================================================================


def save_bins_json(learned: LearnedBins, path: str) -> None:
    """
    Save LearnedBins to JSON, handling NumPy keys/values safely.

    Why we need this
    ----------------
    - NumPy scalars are not always JSON-serializable.
    - Dict keys can also be NumPy scalars (e.g., category values), which must be
      converted to native Python types.
    """

    def default(o):
        """Convert non-serializable objects (NumPy) to JSON-friendly representations."""
        if hasattr(o, "item"):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    def clean_keys(obj):
        """Recursively convert dict keys that may be NumPy scalars to Python types."""
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                k_clean = k.item() if hasattr(k, "item") else k
                new_dict[k_clean] = clean_keys(v)
            return new_dict
        if isinstance(obj, list):
            return [clean_keys(i) for i in obj]
        if isinstance(obj, tuple):
            return tuple(clean_keys(i) for i in obj)
        return obj

    data = {
        "target": learned.target,
        "include_missing": learned.include_missing,
        "missing_label": learned.missing_label,
        "bin_col_suffix": learned.bin_col_suffix,
        "cat_results": learned.cat_results,
        "num_results": learned.num_results,
    }

    clean_data = clean_keys(data)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, default=default)


def load_bins_json(path: str) -> LearnedBins:
    """
    Load a LearnedBins object from JSON.

    Note
    ----
    JSON has no tuple type; any tuple-like structures will come back as lists.
    In this implementation, that's fine because we only require list semantics
    for edges and bin compositions.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return LearnedBins(**data)
