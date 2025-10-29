# -*- coding: utf-8 -*-
# ============================================================
# Binning max |Gini| (cat + num), WOE + sélection, logit + isotonic,
# diagnostics (KS, déciles, calibration, PSI, importance, ablation, prior-shift)
# + Reporting trimestriel (métriques & classes de risque)
# ============================================================

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*is_period_dtype is deprecated.*")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")

# ================================
# Réglages généraux
# ================================
# Sous-échantillonnage pour accélérer (stratifié sur la cible si dispo)
USE_SUBSET         = False
SUBSET_FRAC_TRAIN  = None      # ex: 0.25  (prioritaire sur MAX si non-None)
SUBSET_FRAC_VAL    = None      # ex: 0.25
SUBSET_MAX_TRAIN   = 150_000   # nb max lignes train (utilisé si FRAC est None)
SUBSET_MAX_VAL     = 50_000    # nb max lignes val   (utilisé si FRAC est None)
SUBSET_RANDOM_STATE= 42
SUBSET_MIN_PER_CL  = 100       # garde au moins N obs/classe si possible
TARGET_NAME        = "default_24m"

# Ablation
ABLATION_MAX_STEPS = 10        # 0 pour désactiver l’ablation
ABLATION_MAX_AUC_LOSS = 0.02   # perte AUC tolérée par étape

# PSI proxy temporel (drop automatique si instable)
PSI_CUTOFF_PROXY   = 0.25
conditional_proxies = ["original_interest_rate"]  # variables "sensibles" au drift macro

# ================================
# Imports
# ================================
import re
import json
import math
import logging
from itertools import zip_longest
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Parallélisation
try:
    from joblib import Parallel, delayed
except Exception:  # fallback si joblib indisponible
    Parallel = None
    def delayed(f): return f

# Sklearn (modèle + metrics)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss, roc_curve

# ================================
# Logging
# ================================
logger = logging.getLogger("binning_pipeline")
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[%(levelname)s] %(message)s")
    _h.setFormatter(_fmt)
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

def set_verbosity(verbose: int = 0):
    """0: WARN, 5: INFO, 10+: DEBUG."""
    if verbose >= 10:
        logger.setLevel(logging.DEBUG)
    elif verbose >= 5:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARN)

# ================================
# Options I/O (facultatif)
# ================================
no_flag_missing = True  # supprimer les *_missing auto-générées ?

# ================================
# Chargement des données (adapter les chemins)
# ================================
df_train_imp = pd.read_parquet("data/processed/merged/imputed/train.parquet")
df_val_imp   = pd.read_parquet("data/processed/merged/imputed/validation.parquet")

# --- (NOUVEAU) identifie la colonne trimestre et en garde une copie pour le reporting ---
QUARTER_COL_CANDIDATES = ["vintage", "as_of_quarter", "quarter", "origination_quarter"]
QUARTER_COL = next((c for c in QUARTER_COL_CANDIDATES if c in df_train_imp.columns), None)
if QUARTER_COL is None:
    raise RuntimeError("Aucune colonne de trimestre trouvée (essayé: vintage/as_of_quarter/quarter/origination_quarter).")
df_train_quarter = df_train_imp[QUARTER_COL].copy()
df_val_quarter   = df_val_imp[QUARTER_COL].copy()

# --- 0) Sous-échantillonnage (facultatif, très utile pour itérer vite) ---
def stratified_sample(df, y_col, frac=None, max_rows=None, random_state=42, min_per_class=50):
    """Sous-échantillonne df (stratifié sur y_col si dispo).
       frac prioritaire sur max_rows si non-None.
    """
    if frac is None and (max_rows is None or len(df) <= max_rows):
        return df.copy()

    if frac is not None:
        if y_col in df.columns:
            return (df
                    .groupby(df[y_col], group_keys=False, dropna=False)
                    .apply(lambda g: g.sample(frac=frac, random_state=random_state))
                    .reset_index(drop=True))
        else:
            return df.sample(frac=frac, random_state=random_state).reset_index(drop=True)

    # max_rows défini
    n = len(df)
    if n <= max_rows:
        return df.copy()

    if y_col in df.columns:
        # répartition proportionnelle + min par classe si possible
        counts = df[y_col].value_counts(dropna=False)
        props = counts / counts.sum()
        target_counts = (props * max_rows).astype(int)
        # garantie min
        for cls in counts.index:
            target_counts.loc[cls] = max(target_counts.loc[cls], min_per_class if counts.loc[cls] >= min_per_class else counts.loc[cls])
        # ajuste au total demandé
        delta = max_rows - int(target_counts.sum())
        if delta > 0:
            # ajoute delta aux classes les plus fréquentes
            order = counts.sort_values(ascending=False).index
            for cls in order:
                if delta == 0: break
                can_add = counts.loc[cls] - target_counts.loc[cls]
                add = min(delta, max(can_add, 0))
                target_counts.loc[cls] += add
                delta -= add
        elif delta < 0:
            # retire -delta depuis les plus fréquentes
            order = counts.sort_values(ascending=False).index
            for cls in order:
                if delta == 0: break
                can_remove = target_counts.loc[cls] - min_per_class
                rm = min(-delta, max(can_remove, 0))
                target_counts.loc[cls] -= rm
                delta += rm

        parts = []
        for cls, n_take in target_counts.items():
            g = df[df[y_col] == cls]
            if n_take > 0 and len(g) > 0:
                parts.append(g.sample(n=min(int(n_take), len(g)), random_state=random_state))
        return pd.concat(parts, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    else:
        return df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

if USE_SUBSET:
    df_train_imp = stratified_sample(
        df_train_imp, y_col=TARGET_NAME,
        frac=SUBSET_FRAC_TRAIN, max_rows=SUBSET_MAX_TRAIN,
        random_state=SUBSET_RANDOM_STATE, min_per_class=SUBSET_MIN_PER_CL
    )
    df_val_imp = stratified_sample(
        df_val_imp, y_col=TARGET_NAME,
        frac=SUBSET_FRAC_VAL, max_rows=SUBSET_MAX_VAL,
        random_state=SUBSET_RANDOM_STATE, min_per_class=SUBSET_MIN_PER_CL
    )
    print(f"[SUBSET] train -> {df_train_imp.shape}, val -> {df_val_imp.shape}")

# --- 1) Drop des colonnes *_missing si demandé ---
if no_flag_missing:
    rx = re.compile(r"_missing$", re.I)
    missing_cols = sorted(set(
        [c for c in df_train_imp.columns if rx.search(c)] +
        [c for c in df_val_imp.columns if rx.search(c)]
    ))
    if missing_cols:
        logger.info("Suppression des colonnes *_missing : %s", missing_cols)
        df_train_imp.drop(columns=missing_cols, inplace=True, errors="ignore")
        df_val_imp.drop(columns=missing_cols, inplace=True, errors="ignore")

# --- 2) Denylist FORTE (toujours retirée) ---
denylist_strict = [
    "first_payment_date",       # proxy temporel
    "maturity_date",            # proxy temporel redondant
    "vintage",                  # proxy temporel
    "mi_cancellation_indicator" # post-événement → fuite
]
df_train_imp.drop(columns=denylist_strict, inplace=True, errors="ignore")
df_val_imp.drop(columns=denylist_strict, inplace=True, errors="ignore")

print("train shape:", df_train_imp.shape)
print("val   shape:", df_val_imp.shape)

# ============================================
# Utils Gini (X=Good, Y=Bad)
# ============================================
def gini_trapz(df_cum,
               y_col="bad_client_share_cumsum",
               x_col="good_client_share_cumsum",
               signed=False):
    df = df_cum[[x_col, y_col]].astype(float).copy().sort_values(x_col)
    df[x_col] = df[x_col].clip(0, 1)
    df[y_col] = df[y_col].clip(0, 1)
    if df[x_col].iloc[0] > 0 or df[y_col].iloc[0] > 0:
        df = pd.concat([pd.DataFrame({x_col: [0.0], y_col: [0.0]}), df], ignore_index=True)
    if df[x_col].iloc[-1] < 1 - 1e-12 or df[y_col].iloc[-1] < 1 - 1e-12:
        df = pd.concat([df, pd.DataFrame({x_col: [1.0], y_col: [1.0]})], ignore_index=True)
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    area = np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x)
    g = 1 - 2 * area
    return g if signed else abs(g)

# ======================================================
# Dé-one-hot (protège la cible) + contrôle d'exclusivité
# ======================================================
def detect_onehot_groups(df, allow_singleton=True, exclude_cols=None,
                         exclusivity_check=True, exclusivity_thr=0.95):
    exclude = set(exclude_cols or [])
    groups = {}
    for col in df.columns:
        if col in exclude or "_" not in col:
            continue
        base, label = col.rsplit("_", 1)
        s = df[col]
        is_ohe = (pd.api.types.is_bool_dtype(s) or
                  (pd.api.types.is_numeric_dtype(s) and s.dropna().isin([0, 1]).all()))
        if is_ohe:
            groups.setdefault(base, []).append((col, label))

    clean = {}
    for base, items in groups.items():
        if len(items) >= 2 or allow_singleton:
            if exclusivity_check:
                cols_sorted = [c for c, _ in items]
                vals = df[cols_sorted].apply(pd.to_numeric, errors="coerce")
                row_sum = vals.fillna(0).astype("Int64").sum(axis=1)
                excl_rate = float(((row_sum <= 1) | row_sum.isna()).mean())
                if excl_rate < exclusivity_thr:
                    logger.warning("[WARN] OHE: groupe '%s' exclus (exclusivité %.1f%% < %.1f%%).",
                                   base, 100*excl_rate, 100*exclusivity_thr)
                    continue
            clean[base] = items
    return clean

def deonehot_categoricals(df, allow_singleton=False, exclude_cols=None,
                          ambiguous_label=None,
                          exclusivity_check=True, exclusivity_thr=0.95):
    groups = detect_onehot_groups(
        df, allow_singleton=allow_singleton, exclude_cols=exclude_cols,
        exclusivity_check=exclusivity_check, exclusivity_thr=exclusivity_thr
    )
    out = df.copy()

    def label_sort_key(lab):
        return (1, "") if lab == "<NA>" else (0, str(lab))

    for base, items in groups.items():
        items_sorted = sorted(items, key=lambda x: label_sort_key(x[1]))
        cols_sorted = [c for c, _ in items_sorted]
        labels = [lab for _, lab in items_sorted]

        group_vals = df[cols_sorted].apply(pd.to_numeric, errors="coerce")
        row_sum = group_vals.fillna(0).astype("Int64").sum(axis=1)
        n_amb = int((row_sum > 1).sum())
        if n_amb > 0:
            rate = n_amb / max(len(df), 1)
            logger.warning("[WARN] deonehot: groupe '%s' ambigu sur %d lignes (%.2f%%). NaN affectés.",
                           base, n_amb, 100*rate)

        if len(items_sorted) == 1 and allow_singleton:
            col, lab = items_sorted[0]
            ser = pd.Series("__OTHER__", index=df.index, dtype="object")
            mask = (df[col] == 1)
            ser[mask] = (pd.NA if lab == "<NA>" else lab)
            if n_amb > 0:
                amb_mask = row_sum > 1
                ser[amb_mask] = ambiguous_label if ambiguous_label is not None else pd.NA
            out[base] = ser.astype("category")
            out.drop(columns=[col], inplace=True, errors="ignore")
            continue

        if len(items_sorted) >= 2:
            ser = pd.Series(pd.NA, index=df.index, dtype="object")
            for c, lab in zip(cols_sorted, labels):
                mask = (df[c] == 1)
                ser[mask] = (pd.NA if lab == "<NA>" else lab)
            if n_amb > 0:
                amb_mask = row_sum > 1
                ser[amb_mask] = ambiguous_label if ambiguous_label is not None else pd.NA
            out[base] = ser.astype("category")
            out.drop(columns=cols_sorted, inplace=True, errors="ignore")

    return out

# ======================================
# Cible binaire (auto/forcée)
# ======================================
def infer_binary_target(df, prefer_name_patterns=('default', 'delinq', 'bad', 'target', 'label')):
    candidates = []
    for col in df.columns:
        s = df[col]
        is_bool = pd.api.types.is_bool_dtype(s)
        is_binary_int = (pd.api.types.is_integer_dtype(s) or pd.api.types.is_numeric_dtype(s)) and s.dropna().isin([0, 1]).all()
        is_binary_cat = isinstance(s.dtype, pd.CategoricalDtype) and s.dropna().nunique() == 2
        if is_bool or is_binary_int or is_binary_cat:
            score = 0.0
            name_lower = col.lower()
            for p in prefer_name_patterns:
                if p in name_lower:
                    score += 10.0
            try:
                score += float(1 - min(max(float(s.astype("Int64").mean(skipna=True)), 1e-6), 1 - 1e-6))
            except Exception:
                pass
            candidates.append((score, col))
    if not candidates:
        raise ValueError("Aucune colonne binaire éligible trouvée pour servir de cible.")
    candidates.sort(reverse=True)
    target = candidates[0][1]
    rate = float(pd.to_numeric(df[target], errors="coerce").fillna(0).mean())
    logger.info("[INFO] Cible inférée: '%s' (taux d'événements ≈ %.3f)", target, rate)
    return target

# ===================================================
# Colonnes catégorielles brutes
# ===================================================
def find_categorical_columns(df, target_col=None, max_levels_object=50, exclude_ids=None):
    exclude_ids = set(exclude_ids or [])
    cat_cols = []
    for col in df.columns:
        if col == target_col or col in exclude_ids:
            continue
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(s):
            cat_cols.append(col)
        elif pd.api.types.is_object_dtype(s) or str(s.dtype).startswith("string"):
            if s.nunique(dropna=True) <= max_levels_object:
                cat_cols.append(col)
        elif pd.api.types.is_integer_dtype(s) and s.nunique(dropna=True) <= 8:
            if not any(k in col.lower() for k in ['id', 'sequence', 'loan_sequence']):
                cat_cols.append(col)
    return cat_cols

def extract_ordinal_info(df, cat_cols):
    ordinal_cols, explicit_orders = [], {}
    for col in cat_cols:
        s = df[col]
        if isinstance(s.dtype, pd.CategoricalDtype) and getattr(s.dtype, "ordered", False):
            ordinal_cols.append(col)
            explicit_orders[col] = list(s.dtype.categories)
    return ordinal_cols, explicit_orders

# =========================================================
# Binning catégoriel (fusion pour max |Gini|)
# =========================================================
def _cat_stats(df, col, target_col="target",
               include_missing=True, missing_label="__MISSING__"):
    target = df[target_col].astype(int)
    ser = df[col]
    if include_missing:
        ser = ser.astype("object").where(ser.notna(), missing_label)
    tmp = pd.DataFrame({col: ser, target_col: target})
    agg = tmp.groupby(col, dropna=not include_missing)[target_col].agg(["sum", "count"])
    agg.rename(columns={"sum": "n_bad", "count": "n_total"}, inplace=True)
    agg["n_good"] = agg["n_total"] - agg["n_bad"]
    n_total = len(df)
    n_bad = int(target.sum())
    n_good = n_total - n_bad
    denom_bad = n_bad if n_bad > 0 else 1
    denom_good = n_good if n_good > 0 else 1
    agg["bad_rate"] = agg["n_bad"] / agg["n_total"].where(agg["n_total"] > 0, 1)
    agg["bad_share"] = agg["n_bad"] / denom_bad
    agg["good_share"] = agg["n_good"] / denom_good
    return agg.reset_index().rename(columns={col: "modality"})

def _groups_df_from_bins(stats_df, bins, order_key="bad_rate", ascending=True):
    rows = []
    for i, mods in enumerate(bins):
        sub = stats_df[stats_df["modality"].isin(mods)]
        n_bad = int(sub["n_bad"].sum())
        n_good = int(sub["n_good"].sum())
        n_tot = int(sub["n_total"].sum())
        br = n_bad / n_tot if n_tot > 0 else 0.0
        rows.append({"bin_id": i, "modalities": tuple(mods),
                     "n_total": n_tot, "n_bad": n_bad, "n_good": n_good,
                     "bad_rate": br,
                     "bad_share": sub["bad_share"].sum(),
                     "good_share": sub["good_share"].sum()})
    gdf = pd.DataFrame(rows).sort_values(order_key, ascending=ascending, kind="mergesort").reset_index(drop=True)
    gdf["bad_cum"] = gdf["bad_share"].cumsum()
    gdf["good_cum"] = gdf["good_share"].cumsum()
    return gdf

def _gini_from_bins(stats_df, bins, order_key="bad_rate", ascending=True):
    gdf = _groups_df_from_bins(stats_df, bins, order_key, ascending)
    df_cum = gdf.rename(columns={"good_cum": "good_client_share_cumsum",
                                 "bad_cum": "bad_client_share_cumsum"})[["good_client_share_cumsum", "bad_client_share_cumsum"]]
    return gini_trapz(df_cum, y_col="bad_client_share_cumsum",
                      x_col="good_client_share_cumsum", signed=False)

def _initial_order(stats_df, ordered=False, explicit_order=None, nominal_order_key="bad_rate"):
    if ordered:
        order = list(explicit_order) if explicit_order is not None else list(stats_df["modality"])
        order = [m for m in order if m in set(stats_df["modality"])] + \
                [m for m in stats_df["modality"] if m not in set(order)]
    else:
        order = list(stats_df.sort_values(nominal_order_key)["modality"])
    return order

def _reorder_after_merge(groups, stats_df, ordered, nominal_order_key="bad_rate"):
    if ordered:
        return groups

    def grp_bad_rate(mods):
        sub = stats_df[stats_df["modality"].isin(mods)]
        nb, nt = sub["n_bad"].sum(), sub["n_total"].sum()
        return (nb / nt) if nt > 0 else 0.0

    return sorted(groups, key=lambda mods: grp_bad_rate(mods))

def maximize_gini_via_merging(
    df, col, target_col,
    include_missing=True, missing_label="__MISSING__",
    ordered=False, explicit_order=None,
    max_bins=6, min_bin_size=200, min_bin_frac=None,
    order_key_for_curve="bad_rate", nominal_order_key="bad_rate"
):
    stats_df = _cat_stats(df, col, target_col, include_missing, missing_label)
    order = _initial_order(stats_df, ordered, explicit_order, nominal_order_key)
    groups = [[m] for m in order]
    if len(groups) <= 1:
        mapping = {m: 0 for m in order}
        gdf_final = _groups_df_from_bins(stats_df, groups, order_key_for_curve, True)
        gini_single = _gini_from_bins(stats_df, groups, order_key_for_curve, True)
        return {"mapping": mapping, "gini_before": float(gini_single), "gini_after": float(gini_single),
                "bins_table": gdf_final, "bins": [tuple(grp) for grp in groups]}

    n_total = int(stats_df["n_total"].sum())
    min_needed = 0
    if min_bin_frac is not None:
        min_needed = max(min_needed, math.ceil(float(min_bin_frac) * max(n_total, 1)))
    if min_bin_size is not None:
        min_needed = max(min_needed, int(min_bin_size))

    def constraints_ok(groups_):
        if max_bins is not None and len(groups_) > max_bins:
            return False
        if min_needed and min_needed > 0:
            for mods in groups_:
                if int(stats_df[stats_df["modality"].isin(mods)]["n_total"].sum()) < min_needed:
                    return False
        return True

    while not constraints_ok(groups):
        best_g, best_i = -np.inf, None
        for i in range(len(groups) - 1):
            merged = groups[:i] + [groups[i] + groups[i + 1]] + groups[i + 2:]
            merged = _reorder_after_merge(merged, stats_df, ordered, nominal_order_key)
            g_try = _gini_from_bins(stats_df, merged, order_key_for_curve, True)
            if g_try > best_g:
                best_g, best_i = g_try, i
        if best_i is None:
            best_i = 0
        groups = groups[:best_i] + [groups[best_i] + groups[best_i + 1]] + groups[best_i + 2:]
        groups = _reorder_after_merge(groups, stats_df, ordered, nominal_order_key)

    gini_before = _gini_from_bins(stats_df, [[m] for m in order], order_key_for_curve, True)
    gini_after  = _gini_from_bins(stats_df, groups, order_key_for_curve, True)
    final_bins  = [tuple(mods) for mods in groups]
    mapping     = {m: b for b, mods in enumerate(final_bins) for m in mods}
    gdf_final   = _groups_df_from_bins(stats_df, groups, order_key_for_curve, True)
    return {"mapping": mapping, "gini_before": float(gini_before), "gini_after": float(gini_after),
            "bins_table": gdf_final, "bins": final_bins}

# ==========================================================
# Binning numérique (quantiles glouton, max |Gini|)
#            + conversions dates -> jours + edges sûres
# ==========================================================
def _is_period_dtype(dt):
    try:
        return pd.api.types.is_period_dtype(dt)
    except Exception:
        return False

def _to_float_series(s):
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
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("float64")
    return pd.to_numeric(s, errors="coerce").astype("float64")

def _safe_edges_for_cut(edges, s_float):
    e = np.array(edges, dtype="float64")
    for i in range(1, len(e)):
        if not (e[i] > e[i - 1]):
            e[i] = np.nextafter(e[i - 1], np.inf)
    s_vals = s_float.to_numpy()
    try:
        s_min = float(np.nanmin(s_vals))
        s_max = float(np.nanmax(s_vals))
    except ValueError:
        s_min, s_max = -1.0, 1.0
    rel_eps_lo = 1e-6 * (abs(e[1]) + 1.0) if len(e) > 1 else 1e-6
    rel_eps_hi = 1e-6 * (abs(e[-2]) + 1.0) if len(e) > 1 else 1e-6
    if len(e) >= 2:
        e[0]  = min(e[1] - rel_eps_lo, s_min - rel_eps_lo)
        e[-1] = max(e[-2] + rel_eps_hi, s_max + rel_eps_hi)
    return e

def _gini_from_numeric_bins(y_int, x_float, edges, include_missing=True):
    y = y_int.astype(int).to_numpy()
    x = x_float.to_numpy()
    bins_idx = np.digitize(x, edges[1:-1], right=True)
    K = len(edges) - 1
    n_total = len(y)
    n_bad = int(y.sum())
    n_good = n_total - n_bad
    denom_bad = n_bad if n_bad > 0 else 1
    denom_good = n_good if n_good > 0 else 1
    rows = []
    for k in range(K):
        mask = (bins_idx == k) & ~np.isnan(x)
        nk = int(mask.sum())
        nb = int(y[mask].sum())
        ng = nk - nb
        br = nb / nk if nk > 0 else 0.0
        rows.append({"bin": k, "n_total": nk, "n_bad": nb, "n_good": ng, "bad_rate": br})
    if include_missing and np.isnan(x).any():
        mask = np.isnan(x)
        nk = int(mask.sum())
        nb = int(y[mask].sum())
        ng = nk - nb
        br = nb / nk if nk > 0 else 0.0
        rows.append({"bin": K, "n_total": nk, "n_bad": nb, "n_good": ng, "bad_rate": br})

    gdf = pd.DataFrame(rows)
    if gdf.empty:
        return 0.0, gdf
    gdf["bad_share"]  = gdf["n_bad"]  / denom_bad
    gdf["good_share"] = gdf["n_good"] / denom_good
    gdf = gdf.sort_values("bad_rate").reset_index(drop=True)
    gdf["bad_cum"]  = gdf["bad_share"].cumsum()
    gdf["good_cum"] = gdf["good_share"].cumsum()
    df_cum = gdf.rename(columns={"good_cum": "good_client_share_cumsum",
                                 "bad_cum": "bad_client_share_cumsum"})[["good_client_share_cumsum", "bad_client_share_cumsum"]]
    g = gini_trapz(df_cum, y_col="bad_client_share_cumsum",
                   x_col="good_client_share_cumsum", signed=False)
    return g, gdf

def optimize_numeric_binning_by_quantiles(
    df, col, target_col,
    max_bins=6, min_bin_size=200, min_bin_frac=None,
    n_quantiles=50, q_low=0.02, q_high=0.98,
    include_missing=True, min_gain=1e-5
):
    s = _to_float_series(df[col])
    y = df[target_col].astype(int)
    nunique = s.dropna().nunique()

    if nunique < 2:
        s_f = s[np.isfinite(s)]
        if s_f.empty:
            e_cut = np.array([-1.0, 1.0], dtype="float64")
        else:
            lo, hi = float(np.nanmin(s_f)), float(np.nanmax(s_f))
            eps = 1e-6 * (abs(lo) + abs(hi) + 1.0)
            e_cut = np.array([lo - eps, hi + eps], dtype="float64")
        g0, _ = _gini_from_numeric_bins(y, s, [-np.inf, np.inf], include_missing)
        return {"edges": [-np.inf, np.inf], "edges_for_cut": e_cut, "labels": [f"({e_cut[0]}, {e_cut[1]}]"],
                "gini_before": float(g0), "gini_after": float(g0), "bins_table": pd.DataFrame()}

    qs = np.linspace(q_low, q_high, n_quantiles)
    cand_vals = s.quantile(qs).dropna().unique()
    cand_vals = np.unique(cand_vals)
    edges = [-np.inf, np.inf]

    n = len(s)
    min_needed = 0
    if min_bin_frac is not None:
        min_needed = max(min_needed, math.ceil(float(min_bin_frac) * max(n, 1)))
    if min_bin_size is not None:
        min_needed = max(min_needed, int(min_bin_size))

    def edges_ok(e):
        arr = s.to_numpy()
        bins_idx = np.digitize(arr, e[1:-1], right=True)
        for k in range(len(e) - 1):
            if int(((bins_idx == k) & ~np.isnan(arr)).sum()) < min_needed:
                return False
        return True

    gini0, _ = _gini_from_numeric_bins(y, s, edges, include_missing)
    best_gini = gini0
    improved = True
    while improved and (len(edges) - 1) < max_bins:
        improved = False
        best_gain = min_gain
        best_t = None
        g_best = best_gini
        for t in cand_vals:
            if t in edges:
                continue
            new_edges = sorted([*edges, t])
            if any(np.isclose(new_edges[i], new_edges[i + 1]) for i in range(len(new_edges) - 1)):
                continue
            if not edges_ok(new_edges):
                continue
            g_try, _ = _gini_from_numeric_bins(y, s, new_edges, include_missing)
            gain = g_try - best_gini
            if gain > best_gain:
                best_gain, best_t, g_best = gain, t, g_try
        if best_t is not None:
            edges = sorted([*edges, best_t])
            best_gini = g_best
            improved = True

    gini_after, bins_table = _gini_from_numeric_bins(y, s, edges, include_missing)
    e = sorted(edges)
    e_cut = _safe_edges_for_cut(e, s)
    labels = [f"({e[i]}, {e[i + 1]}]" for i in range(len(e) - 1)]
    return {"edges": e, "edges_for_cut": e_cut, "labels": labels,
            "gini_before": float(gini0), "gini_after": float(gini_after),
            "bins_table": bins_table}

# ==========================================================
# Parallélisation — helpers
# ==========================================================
def _compute_cat_bin_result(df_small, col, target_col,
                            include_missing, missing_label,
                            is_ord, explicit_order,
                            max_bins, min_bin_size, min_bin_frac,
                            order_key_for_curve, nominal_order_key):
    res = maximize_gini_via_merging(
        df=df_small, col=col, target_col=target_col,
        include_missing=include_missing, missing_label=missing_label,
        ordered=is_ord, explicit_order=explicit_order,
        max_bins=max_bins, min_bin_size=min_bin_size, min_bin_frac=min_bin_frac,
        order_key_for_curve=order_key_for_curve, nominal_order_key=nominal_order_key
    )
    return col, res

def _compute_num_bin_result(df_small, col, target_col,
                            max_bins, min_bin_size, min_bin_frac, n_quantiles,
                            include_missing):
    res = optimize_numeric_binning_by_quantiles(
        df=df_small, col=col, target_col=target_col,
        max_bins=max_bins, min_bin_size=min_bin_size, min_bin_frac=min_bin_frac,
        n_quantiles=n_quantiles, include_missing=include_missing
    )
    return col, res

# ==========================================================
# Catégorielles (parallélisées)
# ==========================================================
def auto_bin_all_categoricals(
    df, cat_columns, target_col,
    include_missing=True, missing_label="__MISSING__",
    ordinal_cols=None, explicit_orders=None,
    max_bins=6, min_bin_size=200, min_bin_frac=None,
    order_key_for_curve="bad_rate", nominal_order_key="bad_rate",
    add_binned_columns=True, bin_col_suffix="__BIN",
    n_jobs=1, verbose=0
):
    set_verbosity(verbose)
    ordinal_cols = set(ordinal_cols or [])
    explicit_orders = explicit_orders or {}
    df_out = df.copy()
    results, summary_rows = {}, []

    if n_jobs != 1 and Parallel is not None and len(cat_columns) > 0:
        tasks = (
            delayed(_compute_cat_bin_result)(
                df_out[[col, target_col]].copy(),
                col, target_col,
                include_missing, missing_label,
                (col in ordinal_cols), explicit_orders.get(col),
                max_bins, min_bin_size, min_bin_frac,
                order_key_for_curve, nominal_order_key
            )
            for col in cat_columns
        )
        out = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(list(tasks))
        for col, res in out:
            results[col] = res
    else:
        for col in cat_columns:
            is_ord = col in ordinal_cols
            res = maximize_gini_via_merging(
                df=df_out, col=col, target_col=target_col,
                include_missing=include_missing, missing_label=missing_label,
                ordered=is_ord, explicit_order=explicit_orders.get(col),
                max_bins=max_bins, min_bin_size=min_bin_size, min_bin_frac=min_bin_frac,
                order_key_for_curve=order_key_for_curve, nominal_order_key=nominal_order_key
            )
            results[col] = res

    for col, res in results.items():
        summary_rows.append({
            "variable": col, "type": "categorical",
            "n_bins_final": len(res["bins"]),
            "gini_before": res["gini_before"],
            "gini_after": res["gini_after"],
            "gini_gain": res["gini_after"] - res["gini_before"]
        })
        if add_binned_columns and col in df_out.columns:
            ser = df_out[col].astype("object")
            if include_missing:
                ser = ser.where(ser.notna(), missing_label)
            df_out[col + bin_col_suffix] = ser.map(res["mapping"]).astype("Int64")

    summary = (pd.DataFrame(summary_rows)
               .sort_values("gini_after", ascending=False)
               .reset_index(drop=True))
    return {"results": results, "summary": summary, "df": df_out}

# ==========================================================
# Numériques (parallélisées)
# ==========================================================
def auto_bin_all_numerics(
    df, target_col,
    max_bins=6, min_bin_size=200, min_bin_frac=None,
    n_quantiles=50, include_missing=True,
    add_binned_columns=True, bin_col_suffix="__BIN",
    exclude_ids=None,
    n_jobs=1, verbose=0
):
    set_verbosity(verbose)
    exclude_ids = set(exclude_ids or [])
    df_out = df.copy()
    results, summary_rows = {}, []

    numeric_cols = []
    for col in df.columns:
        if col == target_col or col in exclude_ids:
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            if s.dropna().isin([0, 1]).all():
                continue
            if pd.api.types.is_integer_dtype(s) and s.dropna().nunique() <= 8:
                continue
            if any(k in col.lower() for k in ['id', 'sequence', 'postal', 'zip', 'msa', 'code', 'seller', 'servicer']):
                continue
            numeric_cols.append(col)
        elif _is_period_dtype(s.dtype) or pd.api.types.is_datetime64_any_dtype(s):
            numeric_cols.append(col)

    if n_jobs != 1 and Parallel is not None and len(numeric_cols) > 0:
        tasks = (
            delayed(_compute_num_bin_result)(
                df_out[[col, target_col]].copy(),
                col, target_col,
                max_bins, min_bin_size, min_bin_frac, n_quantiles,
                include_missing
            )
            for col in numeric_cols
        )
        out = Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(list(tasks))
        for col, res in out:
            results[col] = res
    else:
        for col in numeric_cols:
            res = optimize_numeric_binning_by_quantiles(
                df=df_out, col=col, target_col=target_col,
                max_bins=max_bins, min_bin_size=min_bin_size, min_bin_frac=min_bin_frac,
                n_quantiles=n_quantiles, include_missing=include_missing
            )
            results[col] = res

    for col, res in results.items():
        summary_rows.append({
            "variable": col, "type": "numeric",
            "n_bins_final": len(res["edges"]) - 1,
            "gini_before": res["gini_before"],
            "gini_after": res["gini_after"],
            "gini_gain": res["gini_after"] - res["gini_before"]
        })
        if add_binned_columns and col in df_out.columns:
            s = _to_float_series(df_out[col])
            b = pd.cut(s, bins=res["edges_for_cut"], include_lowest=True, duplicates="drop")
            b = b.cat.codes.astype("Int64")
            if include_missing and s.isna().any():
                b = b.where(~s.isna(), -1).astype("Int64")
            df_out[col + bin_col_suffix] = b

    summary = (pd.DataFrame(summary_rows)
               .sort_values("gini_after", ascending=False)
               .reset_index(drop=True))
    return {"results": results, "summary": summary, "df": df_out}

# ============================================
# Assemblage final + One-Hot des BIN (avec -1/-2)
# ============================================
def _ensure_sentinel_categories(base_df: pd.DataFrame, bin_cols):
    base = base_df.copy()
    for c in bin_cols:
        if c in base.columns:
            s = base[c].astype("Int64")
            cats = sorted(set([int(v) for v in s.dropna().unique()]).union({-1, -2}))
            base[c] = pd.Categorical(s, categories=cats)
    return base

def build_final_datasets(out_cat, out_num, drop_original=True, bin_col_suffix="__BIN",
                         keep_vars=None):
    df_enrichi = out_num["df"].copy()
    for c in out_cat["df"].columns:
        if c.endswith(bin_col_suffix) and c not in df_enrichi.columns:
            df_enrichi[c] = out_cat["df"][c]

    cat_all = list(out_cat["results"].keys())
    num_all = list(out_num["results"].keys())

    if keep_vars is not None:
        keep_vars = set(keep_vars)
        cat_keep = [c for c in cat_all if c in keep_vars]
        num_keep = [c for c in num_all if c in keep_vars]
    else:
        cat_keep, num_keep = cat_all, num_all

    all_bin_cols  = [c for c in df_enrichi.columns if c.endswith(bin_col_suffix)]
    keep_bin_cols = [c + bin_col_suffix for c in (cat_keep + num_keep)]
    drop_bin_cols = [c for c in all_bin_cols if c not in keep_bin_cols]

    base_drop = cat_all + num_all + drop_bin_cols
    if drop_original:
        df_binned = (df_enrichi
                     .drop(columns=base_drop, errors="ignore")
                     .rename(columns={c: c.replace(bin_col_suffix, "") for c in keep_bin_cols}))
    else:
        df_binned = df_enrichi.drop(columns=drop_bin_cols, errors="ignore").copy()

    base = df_enrichi.drop(columns=base_drop, errors="ignore")
    base = _ensure_sentinel_categories(base, keep_bin_cols)

    df_ohe = pd.get_dummies(
        base,
        columns=keep_bin_cols,
        prefix={c: c.replace(bin_col_suffix, "") for c in keep_bin_cols},
        dummy_na=False,
        dtype=np.uint8
    )
    df_ohe = df_ohe.reindex(sorted(df_ohe.columns), axis=1)
    return df_enrichi, df_binned, df_ohe

# ============================================
# Sérialisation binnings (JSON)
# ============================================
def bins_to_dict(res, include_missing=True, missing_label="__MISSING__", bin_col_suffix="__BIN"):
    return {
        "target": res["target"],
        "include_missing": include_missing,
        "missing_label": missing_label,
        "bin_col_suffix": bin_col_suffix,
        "cat_results": {
            var: {
                "mapping": {str(k): int(v) for k, v in info["mapping"].items()},
                "gini_before": info["gini_before"],
                "gini_after": info["gini_after"],
                "bins": [list(b) for b in info["bins"]],
            }
            for var, info in res["cat_results"].items()
        },
        "num_results": {
            var: {
                "edges": list(info["edges"]),
                "edges_for_cut": list(np.asarray(info["edges_for_cut"], dtype="float64")),
                "gini_before": info["gini_before"],
                "gini_after": info["gini_after"],
                "labels": list(info.get("labels", [])),
            }
            for var, info in res["num_results"].items()
        },
    }

def bins_from_dict(d):
    return d

def save_bins_json(res, path, **meta):
    d = bins_to_dict(res, **meta)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def load_bins_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ============================================
# LANCEUR complet (protège la cible, parallélisé)
# ============================================
def run_full_pipeline_on_onehot_df(
    df_onehot,
    target_col=None,
    max_bins_categ=6, min_bin_size_categ=200, min_bin_frac_categ=None,
    max_bins_num=6,   min_bin_size_num=200, min_bin_frac_num=200, n_quantiles_num=50,
    include_missing=True, missing_label="__MISSING__", max_levels_object=50,
    bin_col_suffix="__BIN",
    exclude_ids=("loan_sequence_number", "postal_code", "seller_name", "servicer_name", "msa_md"),
    n_jobs_categ=-1, n_jobs_num=-1, verbose=0,
    min_gini_keep=None
):
    set_verbosity(verbose)

    DF = deonehot_categoricals(
        df_onehot,
        allow_singleton=False,
        exclude_cols=[target_col] if target_col else None,
        exclusivity_check=True, exclusivity_thr=0.95
    )

    if target_col is not None and target_col not in DF.columns:
        logger.info("[INFO] Colonne cible '%s' introuvable après préparation. Inférence automatique...", target_col)
        TARGET = infer_binary_target(DF)
    else:
        TARGET = target_col if target_col is not None else infer_binary_target(DF)

    cat_cols = find_categorical_columns(DF, target_col=TARGET, max_levels_object=max_levels_object,
                                        exclude_ids=exclude_ids)
    ordinal_cols, explicit_orders = extract_ordinal_info(DF, cat_cols)
    out_cat = auto_bin_all_categoricals(
        df=DF, cat_columns=cat_cols, target_col=TARGET,
        include_missing=include_missing, missing_label=missing_label,
        ordinal_cols=ordinal_cols, explicit_orders=explicit_orders,
        max_bins=max_bins_categ, min_bin_size=min_bin_size_categ, min_bin_frac=min_bin_frac_categ,
        order_key_for_curve="bad_rate", nominal_order_key="bad_rate",
        add_binned_columns=True, bin_col_suffix=bin_col_suffix,
        n_jobs=n_jobs_categ, verbose=verbose
    )

    out_num = auto_bin_all_numerics(
        df=out_cat["df"], target_col=TARGET,
        max_bins=max_bins_num, min_bin_size=min_bin_size_num, min_bin_frac=min_bin_frac_num,
        n_quantiles=n_quantiles_num, include_missing=include_missing,
        add_binned_columns=True, bin_col_suffix=bin_col_suffix,
        exclude_ids=exclude_ids,
        n_jobs=n_jobs_num, verbose=verbose
    )

    summary = (pd.concat([out_cat["summary"], out_num["summary"]], ignore_index=True)
               .sort_values(["type", "gini_after"], ascending=[True, False])
               .reset_index(drop=True))
    keep_vars = None
    if min_gini_keep is not None:
        keep_vars = summary.loc[summary["gini_after"] >= float(min_gini_keep), "variable"].tolist()
        if verbose:
            nb_drop = (summary["gini_after"] < float(min_gini_keep)).sum()
            logger.info("[INFO] min_gini_keep=%s -> exclusion de %d variables.", min_gini_keep, int(nb_drop))

    df_enrichi, df_binned, df_ohe = build_final_datasets(
        out_cat, out_num,
        drop_original=True,
        bin_col_suffix=bin_col_suffix,
        keep_vars=keep_vars
    )

    return {
        "target": TARGET,
        "summary": summary,
        "df_enrichi": df_enrichi,
        "df_binned": df_binned,
        "df_ohe": df_ohe,
        "cat_results": out_cat["results"],
        "num_results": out_num["results"]
    }

# ============================================
# Transformer val/test avec bins appris (protège la cible)
# ============================================
def transform_with_learned_bins(df_raw_onehot, res, bin_col_suffix="__BIN",
                                include_missing=True,
                                exclude_ids=("loan_sequence_number", "postal_code", "seller_name", "servicer_name", "msa_md")):
    DF = deonehot_categoricals(
        df_raw_onehot,
        allow_singleton=False,
        exclude_cols=[res["target"]]
    )

    for col, r in res["cat_results"].items():
        if col not in DF.columns:
            continue
        s = DF[col].astype("object").where(DF[col].notna(), "__MISSING__")
        mapped = s.map(r["mapping"]).astype("Int64").fillna(-2).astype("Int64")
        DF[col + bin_col_suffix] = mapped

    for col, r in res["num_results"].items():
        if col not in DF.columns:
            continue
        s = _to_float_series(DF[col])
        e = np.array(r["edges_for_cut"], dtype="float64")
        b = pd.cut(s, bins=e, include_lowest=True, duplicates="drop").cat.codes.astype("Int64")
        if include_missing and s.isna().any():
            b = b.where(~s.isna(), -1).astype("Int64")
        DF[col + bin_col_suffix] = b

    cat_cols = list(res["cat_results"].keys())
    num_cols = list(res["num_results"].keys())
    bin_cols = [c + bin_col_suffix for c in cat_cols + num_cols if c + bin_col_suffix in DF.columns]

    base = DF.drop(columns=cat_cols + num_cols, errors="ignore")
    base = _ensure_sentinel_categories(base, bin_cols)

    df_model = pd.get_dummies(
        base,
        columns=bin_cols,
        prefix={c: c.replace(bin_col_suffix, "") for c in bin_cols},
        dummy_na=False,
        dtype=np.uint8
    )
    df_model = df_model.reindex(sorted(df_model.columns), axis=1)
    df_model = df_model.drop(columns=[c for c in exclude_ids if c in df_model.columns], errors="ignore")
    return df_model

# ============================================
# Plots des courbes (départ à 0,0)
# ============================================
def _curve_from_binned(df, bcol, target):
    y = df[target].astype(int)
    s = df[bcol].astype("Int64").fillna(-1).astype("int64")

    agg = pd.DataFrame({bcol: s, target: y}).groupby(bcol)[target].agg(["sum", "count"])
    agg.columns = ["n_bad", "n_total"]
    agg["n_good"] = agg["n_total"] - agg["n_bad"]

    n_bad = int(agg["n_bad"].sum())
    n_good = int(agg["n_good"].sum())
    if n_bad == 0 or n_good == 0:
        df_cum = pd.DataFrame({"good_client_share_cumsum": [0.0, 1.0],
                               "bad_client_share_cumsum": [0.0, 1.0]})
        return df_cum, 0.0

    agg["bad_rate"] = agg["n_bad"] / agg["n_total"].where(agg["n_total"] > 0, 1)
    agg["bad_share"] = agg["n_bad"] / n_bad
    agg["good_share"] = agg["n_good"] / n_good

    agg = agg.sort_values("bad_rate", kind="mergesort")
    good_cum = np.r_[0.0, agg["good_share"].cumsum().values]
    bad_cum  = np.r_[0.0, agg["bad_share"].cumsum().values]

    df_cum = pd.DataFrame({"good_client_share_cumsum": good_cum,
                           "bad_client_share_cumsum": bad_cum})
    g = gini_trapz(df_cum, signed=False)
    return df_cum, float(g)

def plot_all_concentration_curves_from_binned(res, top_n=None, types=("categorical", "numeric")):
    df_base = res["df_enrichi"]
    target = res["target"]
    rows = []
    for t, store in (("categorical", res["cat_results"]), ("numeric", res["num_results"])):
        if t not in types:
            continue
        for var, info in store.items():
            bcol = f"{var}__BIN"
            if bcol not in df_base.columns:
                continue
            df_cum, g = _curve_from_binned(df_base, bcol, target)
            rows.append((t, var, g, df_cum))
    rows.sort(key=lambda x: x[2], reverse=True)
    if top_n is not None:
        rows = rows[:int(top_n)]
    for t, var, g, df_cum in rows:
        plt.figure(figsize=(6, 6))
        plt.plot(df_cum["good_client_share_cumsum"], df_cum["bad_client_share_cumsum"], marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.title(f"{var} [{t}] — Gini = {g:.4f}")
        plt.xlabel("Cumulative good share")
        plt.ylabel("Cumulative bad share")
        plt.grid(True)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
        print(f"{t} — {var}: Gini = {g:.6f}, nb_points={len(df_cum)}")

# =======================
# Fit + WOE + sélection
# =======================
res = run_full_pipeline_on_onehot_df(
    df_onehot=df_train_imp,
    target_col=TARGET_NAME,
    max_bins_categ=6, min_bin_size_categ=200, min_bin_frac_categ=None,
    max_bins_num=6,   min_bin_size_num=200, min_bin_frac_num=None, n_quantiles_num=50,
    include_missing=True, missing_label="__MISSING__", bin_col_suffix="__BIN",
    n_jobs_categ=-1, n_jobs_num=-1,
    verbose=10,
)

# Jeu final OHE (dispo mais non utilisé ensuite)
cols_id = ["loan_sequence_number", "postal_code", "seller_name", "servicer_name", "msa_md"]
df_final = res["df_ohe"].drop(columns=[c for c in cols_id if c in res["df_ohe"].columns], errors="ignore")
y_train = df_final.pop(res["target"]).astype(int)
X_train = df_final

# WOE à partir des BIN sur TRAIN
def woe_from_bin(df, target, bcol, smooth=0.5):
    if bcol not in df.columns:
        raise KeyError(f"Colonne '{bcol}' absente du DataFrame.")
    tab = df.groupby(bcol, dropna=True)[target].agg(['sum', 'count'])
    tab['good'] = tab['count'] - tab['sum']
    B = float(tab['sum'].sum()); G = float(tab['good'].sum()); K = int(len(tab))
    tab['bad_share']  = (tab['sum']  + smooth) / (B + smooth * K if B + smooth * K > 0 else 1.0)
    tab['good_share'] = (tab['good'] + smooth) / (G + smooth * K if G + smooth * K > 0 else 1.0)
    tab['woe'] = np.log(tab['bad_share'] / tab['good_share']).replace([np.inf, -np.inf], np.nan)
    return df[bcol].map(tab['woe']).astype(float)

train_enrichi = res["df_enrichi"].copy()
target = res["target"]
bin_cols = [c for c in train_enrichi.columns if c.endswith("__BIN")]
woe_train = pd.DataFrame({c.replace("__BIN", "_WOE"): woe_from_bin(train_enrichi, target, c) for c in bin_cols})

# Classement des WOE par gini_after
order_woe = (res["summary"]
             .sort_values("gini_after", ascending=False)["variable"]
             .apply(lambda v: f"{v}_WOE")
             .tolist())
order_woe = [v for v in order_woe if v in woe_train.columns]

# Sélection anti-colinéarité
if len(order_woe) == 0:
    raise RuntimeError("Aucune variable WOE disponible pour la sélection.")
corr = woe_train[order_woe].corr().abs().fillna(0.0)

threshold_corr = 0.85
selected = []
for v in order_woe:
    if not selected:
        selected.append(v); continue
    mc = corr.loc[v, corr.columns.intersection(selected)]
    max_corr = float(mc.max()) if len(mc) else 0.0
    if not np.isfinite(max_corr) or np.isnan(max_corr):
        max_corr = 0.0
    if max_corr < threshold_corr:
        selected.append(v)

selected_woe_cols = selected
selected_vars     = [c.removesuffix("_WOE") for c in selected_woe_cols]
print(f"{len(selected_woe_cols)} variables retenues sur {len(order_woe)}")

def summarize_selection(order_woe, selected_woe_cols, corr, threshold=0.85, save_csv=False, csv_prefix="selection"):
    all_woe     = list(order_woe)
    kept_woe    = list(selected_woe_cols)
    dropped_woe = [v for v in all_woe if v not in set(kept_woe)]
    kept_raw    = [v.removesuffix("_WOE") for v in kept_woe]
    dropped_raw = [v.removesuffix("_WOE") for v in dropped_woe]

    print(f"\nRésumé sélection : {len(kept_woe)} retenues / {len(all_woe)} évaluées (seuil corr = {threshold})")
    print("\n— Variables retenues (WOE) —")
    for v in kept_woe: print("  •", v)
    print("\n— Variables écartées (WOE) —")
    for v in dropped_woe: print("  •", v)

    df_vars = pd.DataFrame(list(zip_longest(kept_raw, dropped_raw, fillvalue="")),
                           columns=["kept_raw", "dropped_raw"])

    diag_rows = []
    if len(kept_woe) and len(dropped_woe):
        corr_f = corr.fillna(0.0)
        for v in dropped_woe:
            if (v in corr_f.index) and set(kept_woe).intersection(corr_f.columns):
                ser_abs = corr_f.loc[v, list(kept_woe)].abs()
                if len(ser_abs):
                    top_kept = ser_abs.idxmax()
                    top_val  = float(ser_abs.loc[top_kept])
                else:
                    top_kept, top_val = None, np.nan
            else:
                top_kept, top_val = None, np.nan
            diag_rows.append({
                "dropped_raw": v.removesuffix("_WOE"),
                "kept_reason_raw": (top_kept.removesuffix("_WOE") if isinstance(top_kept, str) else None),
                "abs_corr_with_kept": top_val
            })
    df_diag = pd.DataFrame(diag_rows).sort_values("abs_corr_with_kept", ascending=False, na_position="last")

    if save_csv:
        df_vars.to_csv(f"{csv_prefix}_kept_vs_dropped.csv", index=False)
        df_diag.to_csv(f"{csv_prefix}_dropped_diagnostics.csv", index=False)
        print(f"\nFichiers écrits :\n - {csv_prefix}_kept_vs_dropped.csv\n - {csv_prefix}_dropped_diagnostics.csv")

    display(df_vars.head(20))
    display(df_diag.head(20))
    return df_vars, df_diag

df_vars, df_diag = summarize_selection(order_woe, selected_woe_cols, corr, threshold=threshold_corr)

# ----------------------------------------------------------
# WOE pipeline (maps, application train/val)
# ----------------------------------------------------------
def build_woe_maps(df_enrichi, target, smooth=0.5):
    maps = {}
    y = df_enrichi[target].astype(int)
    B_all = float(y.sum()); G_all = float(len(y) - y.sum())
    global_woe = np.log((B_all + smooth) / (G_all + smooth))
    for bcol in [c for c in df_enrichi.columns if c.endswith("__BIN")]:
        tab = df_enrichi.groupby(bcol, dropna=True)[target].agg(['sum','count'])
        tab['good'] = tab['count'] - tab['sum']
        B = float(tab['sum'].sum()); G = float(tab['good'].sum()); K = int(len(tab)) if len(tab) > 0 else 1
        denom_bad  = (B + smooth * K) if (B + smooth * K) > 0 else 1.0
        denom_good = (G + smooth * K) if (G + smooth * K) > 0 else 1.0
        w = np.log(((tab['sum']+smooth)/denom_bad) / ((tab['good']+smooth)/denom_good)).replace([np.inf, -np.inf], np.nan)
        maps[bcol] = {"map": w.to_dict(), "default": global_woe}
    return maps

def make_enriched_with_bins(df_raw_onehot, res, include_missing=True, bin_col_suffix="__BIN"):
    DF = deonehot_categoricals(
        df_raw_onehot,
        allow_singleton=False,
        exclude_cols=[res["target"]] if res["target"] in df_raw_onehot.columns else None
    )
    for col, r in res["cat_results"].items():
        if col not in DF.columns:
            continue
        s = DF[col].astype("object").where(DF[col].notna(), "__MISSING__")
        DF[col + bin_col_suffix] = s.map(r["mapping"]).astype("Int64").fillna(-2).astype("Int64")
    for col, r in res["num_results"].items():
        if col not in DF.columns:
            continue
        s = _to_float_series(DF[col])
        e = np.array(r["edges_for_cut"], dtype="float64")
        b = pd.cut(s, bins=e, include_lowest=True, duplicates="drop").cat.codes.astype("Int64")
        if include_missing and s.isna().any():
            b = b.where(~s.isna(), -1).astype("Int64")
        DF[col + bin_col_suffix] = b
    return DF

def apply_woe(df_enrichi_with_bins, woe_maps, kept_vars_raw, bin_col_suffix="__BIN"):
    cols = []
    for v in kept_vars_raw:
        bcol = f"{v}{bin_col_suffix}"
        if bcol not in df_enrichi_with_bins.columns or bcol not in woe_maps:
            continue
        ser = df_enrichi_with_bins[bcol].astype("Int64")
        wmap = woe_maps[bcol]["map"]; wdef = float(woe_maps[bcol]["default"])
        x = ser.map(wmap).astype(float).fillna(wdef)
        cols.append((f"{v}_WOE", x))
    if not cols:
        return pd.DataFrame(index=df_enrichi_with_bins.index)
    return pd.concat([s for _, s in cols], axis=1)

# Variables retenues
kept_woe = list(selected_woe_cols)
kept_vars_raw = [c.removesuffix("_WOE") for c in kept_woe]

# TRAIN / VAL WOE
woe_maps = build_woe_maps(res["df_enrichi"], res["target"])
X_train_woe_full = apply_woe(res["df_enrichi"], woe_maps, kept_vars_raw)
y_train_full = res["df_enrichi"][res["target"]].astype(int)

df_val_enrichi = make_enriched_with_bins(df_val_imp, res)
X_val_woe_full = apply_woe(df_val_enrichi, woe_maps, kept_vars_raw)
y_val = df_val_enrichi[res["target"]].astype(int) if res["target"] in df_val_enrichi.columns else None

# Aligne colonnes
X_val_woe_full = X_val_woe_full.reindex(columns=X_train_woe_full.columns, fill_value=0.0)

# ----------------------------------------------------------
# Entraînement : logistique + tuning AUC + calibration isotonic
# ----------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = {
    "C": [0.03, 0.1, 0.3, 1, 3, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "class_weight": [None, "balanced"],
    "max_iter": [2000],
}
gs = GridSearchCV(LogisticRegression(), grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)

def fit_calibrate(Xtr, ytr, Xva):
    gs.fit(Xtr, ytr)
    lr = gs.best_estimator_
    cal = CalibratedClassifierCV(lr, method="isotonic", cv=cv).fit(Xtr, ytr)
    p_tr = cal.predict_proba(Xtr)[:,1]
    p_va = cal.predict_proba(Xva)[:,1]
    return cal, lr, p_tr, p_va

def ks_best_threshold(y, p):
    y = pd.Series(y).astype(int).to_numpy()
    p = pd.Series(p).astype(float).to_numpy()
    if np.unique(y).size < 2:
        return np.nan, np.nan
    fpr, tpr, thr = roc_curve(y, p)
    ks_arr = tpr - fpr
    i = int(np.nanargmax(ks_arr))
    return float(ks_arr[i]), float(thr[i])

def decile_table(y, p, q=10):
    df = pd.DataFrame({"y": pd.Series(y).astype(int), "p": pd.Series(p).astype(float)})
    try:
        df["decile"] = pd.qcut(df["p"], q=q, labels=False, duplicates="drop")
    except Exception:
        n = len(df)
        if n == 0: return pd.DataFrame()
        ranks = df["p"].rank(method="first") / max(n, 1)
        df["decile"] = pd.cut(ranks, bins=np.linspace(0, 1, q+1), labels=False, include_lowest=True)
    tab = (df.groupby("decile", dropna=True)
             .agg(events=("y", "sum"), count=("y", "size"), avg_p=("p", "mean"))
             .sort_index(ascending=False))
    if tab.empty: return tab
    tab["rate"] = tab["events"] / tab["count"].where(tab["count"] > 0, 1)
    tab["cum_events"] = tab["events"].cumsum()
    tab["cum_count"]  = tab["count"].cumsum()
    total_events = float(tab["events"].sum())
    total_count  = float(tab["count"].sum())
    tab["capture"]   = tab["cum_events"] / (total_events if total_events > 0 else 1.0)
    tab["cum_share"] = tab["cum_count"]  / (total_count  if total_count  > 0 else 1.0)
    cum_good = tab["count"] - tab["events"]
    denom_good = float(cum_good.sum()) if float(cum_good.sum()) > 0 else 1.0
    tab["TPR"] = tab["cum_events"] / (total_events if total_events > 0 else 1.0)
    tab["FPR"] = cum_good.cumsum() / denom_good
    tab["KS"]  = tab["TPR"] - tab["FPR"]
    return tab

def calibration_slope_intercept(y, p, eps=1e-9):
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return np.nan, np.nan
    p_clip = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    try:
        lr = LogisticRegression(penalty="none", solver="lbfgs", max_iter=2000)
        lr.fit(logit_p, y)
    except Exception:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        lr.fit(logit_p, y)
    a = float(lr.intercept_[0]); b = float(lr.coef_[0][0])
    return a, b

def psi(a, b, bins=10, eps=1e-9):
    a = np.asarray(a, dtype="float64"); b = np.asarray(b, dtype="float64")
    a_f = a[np.isfinite(a)]; b_f = b[np.isfinite(b)]
    if a_f.size == 0 or b_f.size == 0: return np.nan
    q = np.quantile(a_f, np.linspace(0, 1, bins + 1))
    q = np.unique(q)
    if q.size < 2: return 0.0
    q[0], q[-1] = -np.inf, np.inf
    for i in range(1, len(q)):
        if not (q[i] > q[i-1]): q[i] = np.nextafter(q[i-1], np.inf)
    ca, _ = np.histogram(a_f, bins=q); cb, _ = np.histogram(b_f, bins=q)
    pa = ca / max(ca.sum(), 1); pb = cb / max(cb.sum(), 1)
    return float(np.sum((pa - pb) * np.log((pa + eps) / (pb + eps))))

def psi_classes(y_tr, y_va, eps=1e-9):
    y_tr = np.asarray(y_tr, int); y_va = np.asarray(y_va, int)
    pa1 = np.mean(y_tr == 1); pa0 = 1 - pa1
    pb1 = np.mean(y_va == 1); pb0 = 1 - pb1
    return float((pa0 - pb0) * np.log((pa0 + eps) / (pb0 + eps)) +
                 (pa1 - pb1) * np.log((pa1 + eps) / (pb1 + eps)))

# Fit initial (full WOE sélectionné)
cal0, lr0, p_tr0, p_va0 = fit_calibrate(X_train_woe_full, y_train_full, X_val_woe_full)
if y_val is not None:
    auc0 = roc_auc_score(y_val, p_va0); gini0 = 2*auc0 - 1
    brier0 = brier_score_loss(y_val, p_va0); ll0 = log_loss(y_val, p_va0)
    ks0, thr0 = ks_best_threshold(y_val, p_va0)
    psi0 = psi(p_tr0, p_va0, bins=10)
    print(f"AUC_val={auc0:.4f} | Gini_val={gini0:.4f} | Brier={brier0:.5f} | LogLoss={ll0:.5f}")
    print(f"KS_val={ks0:.4f} | seuil_KS={thr0:.4f}")
    print(f"\nPSI (train→val, probas) = {psi0:.4f}  (≈ <0.10 faible, 0.10–0.25 modéré, >0.25 fort)")
    print(f"PSI (distribution des classes TRAIN→VAL) = {psi_classes(y_train_full, y_val):.4f}")

# ----------------------------------------------------------
# PSI par feature (sur WOE) + drop des proxies temporels instables
# ----------------------------------------------------------
def psi_by_feature(a, b, bins=10, eps=1e-9):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size==0 or b.size==0: return np.nan
    q = np.quantile(a, np.linspace(0,1,bins+1))
    q = np.unique(q); q[0], q[-1] = -np.inf, np.inf
    for i in range(1,len(q)):
        if not (q[i] > q[i-1]): q[i] = np.nextafter(q[i-1], np.inf)
    ca,_ = np.histogram(a, bins=q); cb,_ = np.histogram(b, bins=q)
    pa, pb = ca/ca.sum(), cb/cb.sum()
    return float(np.sum((pa-pb)*np.log((pa+eps)/(pb+eps))))

psi_feat = pd.Series({c: psi_by_feature(X_train_woe_full[c], X_val_woe_full[c]) for c in X_train_woe_full.columns}).sort_values(ascending=False)
print("\nTop 15 PSI(feature):")
print(psi_feat.head(15))

# Drop automatique des proxys temporels si PSI(feature) > cutoff
drop_conditional = []
for raw in conditional_proxies:
    c = f"{raw}_WOE"
    if c in psi_feat.index and psi_feat.loc[c] > PSI_CUTOFF_PROXY:
        drop_conditional.append(c)

X_train_curr = X_train_woe_full.copy()
X_val_curr   = X_val_woe_full.copy()
if drop_conditional:
    print("\nDrop proxies instables (PSI>cutoff):", drop_conditional)
    X_train_curr = X_train_curr.drop(columns=drop_conditional, errors="ignore")
    X_val_curr   = X_val_curr.drop(columns=drop_conditional, errors="ignore")

# Refit après éventuel drop proxy
cal_curr, lr_curr, p_tr_curr, p_va_curr = fit_calibrate(X_train_curr, y_train_full, X_val_curr)
best_auc  = roc_auc_score(y_val, p_va_curr) if y_val is not None else np.nan
best_psi  = psi(p_tr_curr, p_va_curr) if y_val is not None else np.nan
keep_cols = list(X_train_curr.columns)
print(f"\nStart ablation: AUC={best_auc:.4f} | PSI(probas)={best_psi:.4f}")

# ----------------------------------------------------------
# Ablation gloutonne (peut être désactivée en mettant ABLATION_MAX_STEPS=0)
# ----------------------------------------------------------
for step in range(min(ABLATION_MAX_STEPS, len(keep_cols))):
    # recompute PSI(feature) sur l'ensemble courant
    psi_feat_now = pd.Series({c: psi_by_feature(X_train_curr[c], X_val_curr[c]) for c in keep_cols}).sort_values(ascending=False)
    cand = psi_feat_now.index[0]  # feature avec PSI max
    Xtr_try = X_train_curr.drop(columns=[cand])
    Xva_try = X_val_curr.drop(columns=[cand], errors="ignore")
    cal2, lr2, p_tr2, p_va2 = fit_calibrate(Xtr_try, y_train_full, Xva_try)
    auc2  = roc_auc_score(y_val, p_va2) if y_val is not None else np.nan
    psi2  = psi(p_tr2, p_va2) if y_val is not None else np.nan
    gain_psi = (best_psi - psi2) if (not np.isnan(best_psi) and not np.isnan(psi2)) else np.nan
    loss_auc = (best_auc - auc2) if (not np.isnan(best_auc) and not np.isnan(auc2)) else np.nan
    print(f"Try drop {cand:35s} | AUC={auc2:.4f} (Δ{loss_auc:+.4f}) | PSI={psi2:.4f} (Δ{gain_psi:+.4f})")

    # garde la suppression si PSI baisse et perte AUC tolérée
    if (not np.isnan(psi2)) and (psi2 + 1e-6) < best_psi and (np.isnan(loss_auc) or loss_auc <= ABLATION_MAX_AUC_LOSS):
        X_train_curr, X_val_curr = Xtr_try, Xva_try
        cal_curr, lr_curr, p_tr_curr, p_va_curr = cal2, lr2, p_tr2, p_va2
        best_auc, best_psi = auc2, psi2
        keep_cols.remove(cand)
        print("  -> kept removal")
    else:
        print("  -> revert")
        break

# Scores finaux post-ablation
if y_val is not None:
    auc = roc_auc_score(y_val, p_va_curr); gini = 2*auc - 1
    brier = brier_score_loss(y_val, p_va_curr); ll = log_loss(y_val, p_va_curr)
    print(f"\nAfter ablation: AUC_val={auc:.4f} | Gini={gini:.4f} | Brier={brier:.5f} | LogLoss={ll:.5f} | PSI(probas)={best_psi:.4f}")
    print("Features finales :", len(keep_cols))

# ----------------------------------------------------------
# Prior-shift adjust (correction d'intercept) + déciles APRÈS
# ----------------------------------------------------------
def prior_shift_adjust(p, base_train, base_val, eps=1e-9):
    p = np.clip(np.asarray(p, float), eps, 1-eps)
    logit = np.log(p/(1-p))
    delta = np.log((base_val+eps)/(1-base_val+eps)) - np.log((base_train+eps)/(1-base_train+eps))
    z = logit + delta
    return 1 / (1 + np.exp(-z))

if y_val is not None:
    base_train = float(np.mean(y_train_full)); base_val = float(np.mean(y_val))
    p_val_adj = prior_shift_adjust(p_va_curr, base_train, base_val)

    auc_adj = roc_auc_score(y_val, p_val_adj); gini_adj = 2*auc_adj - 1
    brier_adj = brier_score_loss(y_val, p_val_adj); ll_adj = log_loss(y_val, p_val_adj)
    ks_adj, thr_adj = ks_best_threshold(y_val, p_val_adj)
    psi_adj = psi(p_tr_curr, p_val_adj, bins=10)

    print(f"\nAfter prior-shift adjust:")
    print(f"AUC_val={auc_adj:.4f} | Gini_val={gini_adj:.4f} | Brier={brier_adj:.5f} | LogLoss={ll_adj:.5f}")
    print(f"KS_val={ks_adj:.4f} | seuil_KS={thr_adj:.4f}")
    print(f"PSI (train→val, probas) après ajustement = {psi_adj:.4f}")

    dec_val_after = decile_table(y_val, p_val_adj, q=10)
    if not dec_val_after.empty:
        print("\nDéciles — APRÈS prior-shift :")
        print(dec_val_after[["count","events","rate","avg_p","capture","KS"]])
        print(f"KS_val (déciles, après) = {float(dec_val_after['KS'].max()):.4f}")

# ----------------------------------------------------------
# Importance des variables (coeffs standardisés) — ROBUSTE
# ----------------------------------------------------------
def compute_standardized_importance(estimator, X_ref, fallback_X=None, topn=15):
    """
    estimator : LogisticRegression déjà fit (le dernier utilisé)
    X_ref     : DataFrame aligné avec le fit final (idéalement la matrice passée au dernier .fit())
    fallback_X: DataFrame de secours si besoin (ex: X_train_woe_full)
    """
    if not hasattr(estimator, "coef_"):
        print("Importance non calculée : l'estimateur n'a pas d'attribut coef_.")
        return

    beta = np.asarray(estimator.coef_).ravel()

    # 1) Colonnes réellement utilisées pour le dernier fit
    if X_ref is not None and isinstance(X_ref, pd.DataFrame):
        feat_cols_used = list(X_ref.columns)
    elif hasattr(estimator, "feature_names_in_"):
        feat_cols_used = list(estimator.feature_names_in_)
    elif fallback_X is not None:
        feat_cols_used = list(fallback_X.columns[:len(beta)])
    else:
        print("Impossible de déterminer les noms de variables utilisés.")
        return

    # 2) Alignement par troncature prudente si tailles décalées
    if len(beta) != len(feat_cols_used):
        logger.warning("Taille coef_ (%d) ≠ nb colonnes (%d). Alignement par troncature.",
                       len(beta), len(feat_cols_used))
        m = min(len(beta), len(feat_cols_used))
        beta = beta[:m]
        feat_cols_used = feat_cols_used[:m]

    # 3) Construire le X pour std, en réindexant sur les features attendues
    if X_ref is None and fallback_X is not None:
        X_for_std = fallback_X.reindex(columns=feat_cols_used)
    else:
        X_for_std = X_ref.reindex(columns=feat_cols_used)

    missing = [c for c in feat_cols_used if c not in (X_for_std.columns)]
    if missing:
        logger.warning("Features absentes pour std (ignorées): %s", missing)

    coefs = pd.Series(beta, index=feat_cols_used)
    stds  = X_for_std.std(ddof=0).replace(0, np.nan)
    std_coef = coefs * stds

    imp = (pd.DataFrame({"coef": coefs, "std": stds, "std_coef": std_coef})
           .dropna(subset=["std_coef"])
           .sort_values("std_coef", key=lambda s: s.abs(), ascending=False))

    print("\nTop variables (|coef|*sd) :")
    print(imp.head(topn))
    return imp

# Importance sur le modèle final
lr_final   = lr_curr
X_train_fit= X_train_curr
_ = compute_standardized_importance(
    estimator=lr_final,
    X_ref=X_train_fit,
    fallback_X=X_train_woe_full,
    topn=15
)

# ================================
# Reporting trimestriel complet
# ================================
OUT_DIR = Path("artifacts/pipeline")
(OUT_DIR / "quarterly").mkdir(parents=True, exist_ok=True)

def _safe_monotonic_edges(v, q=10):
    """Déciles robustes + bornes -inf/+inf, monotones strictes."""
    v = np.asarray(v, float)
    qs = np.linspace(0, 1, q + 1)
    raw = np.quantile(v[np.isfinite(v)], qs)
    raw = np.unique(raw)
    if raw.size < 2:
        # cas dégénéré
        lo = np.nanmin(v) if np.isfinite(v).any() else 0.0
        hi = np.nanmax(v) if np.isfinite(v).any() else 1.0
        raw = np.array([lo, hi], float)
    # étends aux infinis et force la stricte augmentation
    edges = np.r_[-np.inf, raw[1:-1], np.inf].astype(float)
    for i in range(1, len(edges)):
        if not (edges[i] > edges[i-1]):
            edges[i] = np.nextafter(edges[i-1], np.inf)
    return edges

def _risk_classes_from_train(p_train, q=10):
    """Construit les classes sur TRAIN + stats min/max/mean train par classe."""
    edges = _safe_monotonic_edges(p_train, q=q)
    bins_train = pd.cut(p_train, bins=edges, include_lowest=True, duplicates="drop")
    tab_train = (pd.DataFrame({
                    "class": bins_train.cat.codes.astype("Int64"),
                    "p": p_train
                 })
                 .groupby("class", dropna=True)
                 .agg(pd_min_train=("p", "min"),
                      pd_max_train=("p", "max"),
                      pd_ttc_train=("p", "mean"),
                      n_train=("p", "size"))
                 .reset_index()
                 .sort_values("class", ascending=False))
    tab_train["class_label"] = tab_train["class"].apply(lambda c: f"C{int(c)}")
    return edges, tab_train

def _classify_with_edges(p, edges):
    return pd.cut(p, bins=edges, include_lowest=True, duplicates="drop").cat.codes.astype("Int64")

def _metrics_for_group(y_true, p_pred, p_train_baseline):
    has_y = y_true is not None and pd.Series(y_true).dropna().nunique() >= 2
    auc = gini = brier = ll = ks = ks_thr = np.nan
    a_cal = b_cal = np.nan
    if has_y:
        auc = float(roc_auc_score(y_true, p_pred))
        gini = 2*auc - 1
        brier = float(brier_score_loss(y_true, p_pred))
        ll = float(log_loss(y_true, p_pred))
        ks, ks_thr = ks_best_threshold(y_true, p_pred)
        a_cal, b_cal = calibration_slope_intercept(y_true, p_pred)
    # drift proba vs TRAIN
    psi_prob = float(psi(p_train_baseline, p_pred))
    # drift classes vs TRAIN si y dispo
    psi_cls = float(psi_classes(y_train_full, y_true)) if has_y else np.nan
    return {
        "auc": auc, "gini": gini, "brier": brier, "logloss": ll,
        "ks": ks, "ks_thr": ks_thr, "cal_intercept": a_cal, "cal_slope": b_cal,
        "psi_prob": psi_prob, "psi_classes": psi_cls
    }

def _woe_for(df_enrichi, keep_vars_raw):
    return apply_woe(df_enrichi, woe_maps, keep_vars_raw).reindex(columns=X_train_curr.columns, fill_value=0.0)

def _psi_feat_summary(X_train_ref, X_group):
    vals = {c: psi_by_feature(X_train_ref[c], X_group[c]) for c in X_train_ref.columns}
    s = pd.Series(vals, dtype="float64").sort_values(ascending=False)
    top_feat = s.index[0] if len(s) else None
    top_val = float(s.iloc[0]) if len(s) else np.nan
    mean_val = float(s.replace([np.inf, -np.inf], np.nan).dropna().mean()) if len(s) else np.nan
    return top_feat, top_val, mean_val, s

def _quarter_series_to_str(s):
    if pd.api.types.is_period_dtype(s.dtype):
        return s.astype(str)
    try:
        return s.astype(str)
    except Exception:
        return s

def make_quarterly_reports(
    train_enrichi, train_quarter,
    val_enrichi,   val_quarter,
    model_calibrated,   # cal_curr
    p_train_baseline,   # p_tr_curr
    risk_q=10,
    out_dir=OUT_DIR
):
    # 1) Edges + table train (bornes)
    edges, tab_train_bins = _risk_classes_from_train(pd.Series(p_train_baseline, dtype="float64"), q=risk_q)
    (out_dir / "quarterly").mkdir(parents=True, exist_ok=True)
    tab_train_bins.to_csv(out_dir / "quarterly" / "risk_classes_train.csv", index=False)

    # 2) Construit un DF "all" pour scorer par trimestre
    tr = train_enrichi.copy()
    tr["_quarter"] = _quarter_series_to_str(train_quarter).values
    tr["_split"] = "train"
    va = val_enrichi.copy()
    va["_quarter"] = _quarter_series_to_str(val_quarter).values
    va["_split"] = "val"
    all_enrichi = pd.concat([tr, va], ignore_index=True)

    # 3) Score + WOE aligné
    X_all = _woe_for(all_enrichi, kept_vars_raw)
    p_all = model_calibrated.predict_proba(X_all)[:, 1]
    y_all = all_enrichi[res["target"]] if res["target"] in all_enrichi.columns else None

    # 4) Assemble pour groupby trimestre
    df_score = pd.DataFrame({
        "_quarter": all_enrichi["_quarter"],
        "_split": all_enrichi["_split"],
        "p": p_all
    })
    if y_all is not None:
        df_score["y"] = y_all.astype(int).to_numpy()

    # 5) métriques par trimestre
    rows_metrics = []
    risk_tables = []
    deciles_tables = []

    for qk, g in df_score.groupby("_quarter", dropna=False):
        idx = g.index
        p = g["p"].to_numpy()
        y = g["y"].to_numpy() if "y" in g.columns else None

        # WOE group pour PSI(feature)
        Xg = X_all.loc[idx]
        top_feat, top_val, mean_val, s_feat = _psi_feat_summary(X_train_curr, Xg)

        # métriques globales
        m = _metrics_for_group(y, p, p_train_baseline)
        base_rate = float(np.mean(y)) if y is not None else np.nan

        rows_metrics.append({
            "quarter": str(qk),
            "n": int(len(g)),
            "base_rate": base_rate,
            **m,
            "psi_feat_top_name": top_feat,
            "psi_feat_top_value": top_val,
            "psi_feat_mean": mean_val
        })

        # déciles (sur le trimestre, juste descriptif)
        deci = decile_table(y, p, q=10) if y is not None else pd.DataFrame()
        if not deci.empty:
            deci = deci.reset_index().rename(columns={"decile": "decile_rank"})
            deci.insert(0, "quarter", str(qk))
            deciles_tables.append(deci)

        # table de classes (avec bornes train)
        cls = _classify_with_edges(pd.Series(p, dtype="float64"), edges)
        tab = (pd.DataFrame({"class": cls, "y": y if y is not None else np.nan})
               .groupby("class", dropna=True)
               .agg(n=("class","size"),
                    defaults=("y","sum"))
               .reset_index())
        if "defaults" in tab.columns:
            tab["defaults"] = tab["defaults"].fillna(0).astype("Int64")
            tab["rate"] = (tab["defaults"] / tab["n"].where(tab["n"] > 0, 1)).astype(float)
        else:
            tab["defaults"] = pd.NA
            tab["rate"] = np.nan

        # joint la table train pour les bornes et la PD TTC de classe
        tab = tab.merge(tab_train_bins[["class","pd_min_train","pd_max_train","pd_ttc_train"]],
                        on="class", how="right").sort_values("class", ascending=False)
        tab.insert(0, "quarter", str(qk))
        risk_tables.append(tab)

    # 6) Sauvegardes
    met = pd.DataFrame(rows_metrics).sort_values("quarter")
    met.to_csv(out_dir / "quarterly" / "metrics_by_quarter.csv", index=False)
    if deciles_tables:
        pd.concat(deciles_tables, ignore_index=True).to_csv(out_dir / "quarterly" / "deciles_by_quarter.csv", index=False)
    if risk_tables:
        pd.concat(risk_tables, ignore_index=True).to_csv(out_dir / "quarterly" / "risk_table_by_quarter.csv", index=False)

    print(f"[OK] Quarterly reports écrits dans {out_dir/'quarterly'}")
    return met

# ===== Lance le reporting sur train+val =====
train_enrichi_with_q = res["df_enrichi"].copy()
val_enrichi_with_q   = df_val_enrichi.copy()

_ = make_quarterly_reports(
    train_enrichi=train_enrichi_with_q, train_quarter=df_train_quarter,
    val_enrichi=val_enrichi_with_q,     val_quarter=df_val_quarter,
    model_calibrated=cal_curr,
    p_train_baseline=pd.Series(p_tr_curr, dtype="float64"),
    risk_q=10,
    out_dir=OUT_DIR
)
