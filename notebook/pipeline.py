# -*- coding: utf-8 -*-

# ================================
# Imports
# ================================
import re
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

# ================================
# Options
# ================================
# Par défaut on NE supprime PAS les *_missing : on laissera la sélection (Gini) décider
no_flag_missing = False  # ne supprime plus les colonnes *_missing par défaut

# ================================
# Chargement unique des données
# ================================
df_train_imp = pd.read_parquet("../data/processed/merged/imputed/train.parquet")
df_val_imp   = pd.read_parquet("../data/processed/merged/imputed/validation.parquet")

# Ajustements spécifiques (idéalement dans l'imputer)
# 👉 On ne remplace plus les NaN de 'first_time_homebuyer_flag' par False.
#    On laisse la mécanique de binning gérer les NaN via __MISSING__.
# for _df in (df_train_imp, df_val_imp):
#     if "first_time_homebuyer_flag" in _df.columns:
#         _df["first_time_homebuyer_flag"] = _df["first_time_homebuyer_flag"].fillna(False)

# Suppression optionnelle des flags *_missing (insensible à la casse) sur train+val
if no_flag_missing:
    rx = re.compile(r"_missing$", re.I)
    missing_cols = sorted(set(
        [c for c in df_train_imp.columns if rx.search(c)] +
        [c for c in df_val_imp.columns if rx.search(c)]
    ))
    if missing_cols:
        print("Suppression des colonnes *_missing :", missing_cols)
        df_train_imp.drop(columns=missing_cols, inplace=True, errors="ignore")
        df_val_imp.drop(columns=missing_cols, inplace=True, errors="ignore")

print("train shape:", df_train_imp.shape)
print("val   shape:", df_val_imp.shape)

# ============================================
# Utils Gini (X=Good, Y=Bad)
# ============================================
def gini_trapz(df_cum,
               y_col="bad_client_share_cumsum",
               x_col="good_client_share_cumsum",
               signed=False):
    """
    Gini = 1 - 2 * aire(y vs x).
    Par défaut on renvoie |Gini| (signed=False).
    Sécurise les endpoints (0,0) et (1,1) et clamp dans [0,1].
    """
    df = df_cum[[x_col, y_col]].astype(float).copy().sort_values(x_col)
    # clamp
    df[x_col] = df[x_col].clip(0, 1)
    df[y_col] = df[y_col].clip(0, 1)
    # endpoints
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
def detect_onehot_groups(df, allow_singleton=True, exclude_cols=None):
    """
    Détecte les groupes one-hot en scindant au DERNIER underscore.
    Accepte tout suffixe (state_CA, grade_A, ...). 0/1 ou bool requis.
    exclude_cols : colonnes à ignorer (ex: la cible).
    """
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
            clean[base] = items
    return clean

def deonehot_categoricals(df, allow_singleton=False, exclude_cols=None, ambiguous_label=None):
    """
    Recompose des colonnes one-hot en une seule catégorie (dtype category).
    - multi-colonnes -> fusion
    - singleton -> fusion seulement si allow_singleton=True
    - "<NA>" (texte) -> NaN, rangé en dernier
    - exclude_cols protégées (ex: cible)
    - contrôle d'exclusivité: si plusieurs 1 sur une même ligne -> valeur ambiguë (NaN par défaut)
    """
    groups = detect_onehot_groups(df, allow_singleton=allow_singleton, exclude_cols=exclude_cols)
    out = df.copy()

    def label_sort_key(lab):
        return (1, "") if lab == "<NA>" else (0, str(lab))

    for base, items in groups.items():
        items_sorted = sorted(items, key=lambda x: label_sort_key(x[1]))
        cols_sorted = [c for c, _ in items_sorted]
        labels = [lab for _, lab in items_sorted]

        # Contrôle d'exclusivité (somme par ligne des OHE du groupe)
        group_vals = df[cols_sorted].apply(pd.to_numeric, errors="coerce")
        row_sum = group_vals.fillna(0).astype("Int64").sum(axis=1)
        n_amb = int((row_sum > 1).sum())
        n_any = int((row_sum >= 1).sum())
        if n_amb > 0:
            rate = n_amb / max(len(df), 1)
            print(f"[WARN] deonehot: groupe '{base}' ambigu sur {n_amb} lignes "
                  f"({rate:.2%} des lignes du DF). Ces lignes seront mises en NaN.")

        if len(items_sorted) == 1 and allow_singleton:
            col, lab = items_sorted[0]
            ser = pd.Series("__OTHER__", index=df.index, dtype="object")
            mask = (df[col] == 1)
            ser[mask] = (pd.NA if lab == "<NA>" else lab)
            # Ambiguïtés (théoriquement impossibles en singleton) -> NaN/ambiguous_label
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
            # Ambiguïtés -> NaN/ambiguous_label
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
    return candidates[0][1]

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
    max_bins=6, min_bin_size=200,
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

    # Contraintes
    def constraints_ok(groups_):
        if max_bins is not None and len(groups_) > max_bins:
            return False
        if min_bin_size and min_bin_size > 0:
            for mods in groups_:
                if int(stats_df[stats_df["modality"].isin(mods)]["n_total"].sum()) < min_bin_size:
                    return False
        return True

    # Merge glouton tant que les contraintes ne sont pas respectées
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
    # Period -> début de période -> jours depuis epoch
    if _is_period_dtype(s.dtype):
        ts = s.dt.to_timestamp(how="start")
        days = (ts.astype("int64") // 86_400_000_000_000)  # ns -> jours
        return days.astype("float64")
    # Datetime -> jours depuis epoch (protège les tz)
    if pd.api.types.is_datetime64_any_dtype(s):
        s_dt = s
        # si tz-aware -> rendre naïf
        try:
            if getattr(s_dt.dt, "tz", None) is not None:
                s_dt = s_dt.dt.tz_convert(None)
        except Exception:
            pass
        s_dt = s_dt.astype("datetime64[ns]")
        days = (s_dt.astype("int64") // 86_400_000_000_000)
        return days.astype("float64")
    # Numérique
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype("float64")
    # Objet -> numérique (coerce)
    return pd.to_numeric(s, errors="coerce").astype("float64")

def _safe_edges_for_cut(edges, s_float):
    """
    edges: liste triée [-inf, t1, ..., +inf] -> array strictement croissante
    élargit extrémités et corrige les égalités numériques.
    """
    e = np.array(edges, dtype="float64")
    for i in range(1, len(e)):
        if not (e[i] > e[i - 1]):
            e[i] = np.nextafter(e[i - 1], np.inf)

    s_vals = s_float.to_numpy()
    try:
        s_min = float(np.nanmin(s_vals))
        s_max = float(np.nanmax(s_vals))
    except ValueError:
        # tout NaN
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
    max_bins=6, min_bin_size=200,
    n_quantiles=50, q_low=0.02, q_high=0.98,
    include_missing=True, min_gain=1e-5
):
    s = _to_float_series(df[col])
    y = df[target_col].astype(int)
    nunique = s.dropna().nunique()
    if nunique < 2:
        # une seule modalité -> un bin
        g0, _ = _gini_from_numeric_bins(y, s, [-np.inf, np.inf], include_missing)
        return {"edges": [-np.inf, np.inf], "edges_for_cut": [-1.0, 1.0], "labels": ["(-inf, inf]"],
                "gini_before": float(g0), "gini_after": float(g0), "bins_table": pd.DataFrame()}

    qs = np.linspace(q_low, q_high, n_quantiles)
    cand_vals = s.quantile(qs).dropna().unique()
    cand_vals = np.unique(cand_vals)
    edges = [-np.inf, np.inf]

    def edges_ok(e):
        arr = s.to_numpy()
        bins_idx = np.digitize(arr, e[1:-1], right=True)
        for k in range(len(e) - 1):
            if int(((bins_idx == k) & ~np.isnan(arr)).sum()) < min_bin_size:
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
            # ignore seuils quasi-identiques
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
                            max_bins, min_bin_size,
                            order_key_for_curve, nominal_order_key):
    # df_small contient uniquement [col, target_col]
    res = maximize_gini_via_merging(
        df=df_small, col=col, target_col=target_col,
        include_missing=include_missing, missing_label=missing_label,
        ordered=is_ord, explicit_order=explicit_order,
        max_bins=max_bins, min_bin_size=min_bin_size,
        order_key_for_curve=order_key_for_curve, nominal_order_key=nominal_order_key
    )
    return col, res

def _compute_num_bin_result(df_small, col, target_col,
                            max_bins, min_bin_size, n_quantiles,
                            include_missing):
    # df_small contient uniquement [col, target_col]
    res = optimize_numeric_binning_by_quantiles(
        df=df_small, col=col, target_col=target_col,
        max_bins=max_bins, min_bin_size=min_bin_size,
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
    max_bins=6, min_bin_size=200,
    order_key_for_curve="bad_rate", nominal_order_key="bad_rate",
    add_binned_columns=True, bin_col_suffix="__BIN",
    n_jobs=1, verbose=0
):
    ordinal_cols = set(ordinal_cols or [])
    explicit_orders = explicit_orders or {}
    df_out = df.copy()
    results, summary_rows = {}, []

    # 1) calcule les binnings en parallèle (pas de mapping ici)
    if n_jobs != 1 and Parallel is not None and len(cat_columns) > 0:
        tasks = (
            delayed(_compute_cat_bin_result)(
                df_out[[col, target_col]].copy(),
                col, target_col,
                include_missing, missing_label,
                (col in ordinal_cols), explicit_orders.get(col),
                max_bins, min_bin_size,
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
                max_bins=max_bins, min_bin_size=min_bin_size,
                order_key_for_curve=order_key_for_curve, nominal_order_key=nominal_order_key
            )
            results[col] = res

    # 2) mapping (série, pour limiter la conso mémoire)
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
    max_bins=6, min_bin_size=200,
    n_quantiles=50, include_missing=True,
    add_binned_columns=True, bin_col_suffix="__BIN",
    exclude_ids=None,
    n_jobs=1, verbose=0
):
    exclude_ids = set(exclude_ids or [])
    df_out = df.copy()
    results, summary_rows = {}, []

    # Détection colonnes numériques/period/datetime
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

    # 1) calcule les binnings en parallèle (pas de cut ici)
    if n_jobs != 1 and Parallel is not None and len(numeric_cols) > 0:
        tasks = (
            delayed(_compute_num_bin_result)(
                df_out[[col, target_col]].copy(),
                col, target_col,
                max_bins, min_bin_size, n_quantiles,
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
                max_bins=max_bins, min_bin_size=min_bin_size,
                n_quantiles=n_quantiles, include_missing=include_missing
            )
            results[col] = res

    # 2) application des cuts (série, mémoire friendly)
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
# Assemblage final + One-Hot des BIN (corrigé)
# ============================================
def build_final_datasets(out_cat, out_num, drop_original=True, bin_col_suffix="__BIN",
                         keep_vars=None):
    """
    keep_vars: iterable de noms de variables (sans suffixe __BIN) à conserver.
               Si None -> conserve toutes les variables binned.
    """
    df_enrichi = out_num["df"].copy()
    # récupère aussi les BIN caté ajoutés dans out_cat
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

    # BIN à garder / à supprimer
    all_bin_cols  = [c for c in df_enrichi.columns if c.endswith(bin_col_suffix)]
    keep_bin_cols = [c + bin_col_suffix for c in (cat_keep + num_keep)]
    drop_bin_cols = [c for c in all_bin_cols if c not in keep_bin_cols]

    # Dataset binned (on enlève les brutes + les BIN non gardés)
    base_drop = cat_all + num_all + drop_bin_cols
    if drop_original:
        df_binned = (df_enrichi
                     .drop(columns=base_drop, errors="ignore")
                     .rename(columns={c: c.replace(bin_col_suffix, "") for c in keep_bin_cols}))
    else:
        df_binned = df_enrichi.drop(columns=drop_bin_cols, errors="ignore").copy()

    # OHE final uniquement des BIN retenus
    base = df_enrichi.drop(columns=base_drop, errors="ignore")
    df_ohe = pd.get_dummies(
        base,
        columns=keep_bin_cols,
        prefix={c: c.replace(bin_col_suffix, "") for c in keep_bin_cols},
        dummy_na=False,
        dtype=np.uint8
    )
    # Option : ordre stable des colonnes
    df_ohe = df_ohe.reindex(sorted(df_ohe.columns), axis=1)
    return df_enrichi, df_binned, df_ohe

# ============================================
# LANCEUR complet (protège la cible, parallélisé)
# ============================================
def run_full_pipeline_on_onehot_df(
    df_onehot,
    target_col=None,                         # passe "default_24m" pour forcer
    max_bins_categ=6, min_bin_size_categ=200,
    max_bins_num=6,   min_bin_size_num=200, n_quantiles_num=50,
    include_missing=True, missing_label="__MISSING__", max_levels_object=50,
    bin_col_suffix="__BIN",
    exclude_ids=("loan_sequence_number", "postal_code", "seller_name", "servicer_name", "msa_md"),
    n_jobs_categ=-1, n_jobs_num=-1, verbose=0,
    min_gini_keep=None   # ⇦ optionnel: filtre les variables à faible Gini (ex: 1e-6)
):
    # 0) dé-one-hot en protégeant la cible si fournie (contrôle exclusivité intégré)
    DF = deonehot_categoricals(
        df_onehot,
        allow_singleton=False,                         # évite d'avaler des singletons ambigus
        exclude_cols=[target_col] if target_col else None
    )

    # 1) cible
    if target_col is not None and target_col not in DF.columns:
        print(f"[INFO] Colonne cible '{target_col}' introuvable après préparation. Inférence automatique...")
        TARGET = infer_binary_target(DF)
    else:
        TARGET = target_col if target_col is not None else infer_binary_target(DF)

    # 2) catégorielles
    cat_cols = find_categorical_columns(DF, target_col=TARGET, max_levels_object=max_levels_object,
                                        exclude_ids=exclude_ids)
    ordinal_cols, explicit_orders = extract_ordinal_info(DF, cat_cols)
    out_cat = auto_bin_all_categoricals(
        df=DF, cat_columns=cat_cols, target_col=TARGET,
        include_missing=include_missing, missing_label=missing_label,
        ordinal_cols=ordinal_cols, explicit_orders=explicit_orders,
        max_bins=max_bins_categ, min_bin_size=min_bin_size_categ,
        order_key_for_curve="bad_rate", nominal_order_key="bad_rate",
        add_binned_columns=True, bin_col_suffix=bin_col_suffix,
        n_jobs=n_jobs_categ, verbose=verbose
    )

    # 3) numériques
    out_num = auto_bin_all_numerics(
        df=out_cat["df"], target_col=TARGET,
        max_bins=max_bins_num, min_bin_size=min_bin_size_num,
        n_quantiles=n_quantiles_num, include_missing=include_missing,
        add_binned_columns=True, bin_col_suffix=bin_col_suffix,
        exclude_ids=exclude_ids,
        n_jobs=n_jobs_num, verbose=verbose
    )

    # 4) datasets finaux (+ filtrage Gini optionnel)
    summary = (pd.concat([out_cat["summary"], out_num["summary"]], ignore_index=True)
               .sort_values(["type", "gini_after"], ascending=[True, False])
               .reset_index(drop=True))
    keep_vars = None
    if min_gini_keep is not None:
        keep_vars = summary.loc[summary["gini_after"] >= float(min_gini_keep), "variable"].tolist()
        if verbose:
            nb_drop = (summary["gini_after"] < float(min_gini_keep)).sum()
            print(f"[INFO] min_gini_keep={min_gini_keep} -> exclusion de {nb_drop} variables.")

    df_enrichi, df_binned, df_ohe = build_final_datasets(
        out_cat, out_num,
        drop_original=True,
        bin_col_suffix=bin_col_suffix,
        keep_vars=keep_vars
    )

    return {
        "target": TARGET,
        "summary": summary,
        "df_enrichi": df_enrichi,    # contient les colonnes *_BIN
        "df_binned": df_binned,      # (optionnel) DF avec colonnes BIN renommées
        "df_ohe": df_ohe,            # OHE final prêt pour le modèle
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
        exclude_cols=[res["target"]]  # protège la colonne cible
    )

    # 1) catégorielles (mappings appris)
    for col, r in res["cat_results"].items():
        if col not in DF.columns:
            continue
        s = DF[col].astype("object").where(DF[col].notna(), "__MISSING__")
        mapped = s.map(r["mapping"]).astype("Int64")
        mapped = mapped.fillna(-2).astype("Int64")  # catégories jamais vues -> -2
        DF[col + bin_col_suffix] = mapped

    # 2) numériques (edges appris)
    for col, r in res["num_results"].items():
        if col not in DF.columns:
            continue
        s = _to_float_series(DF[col])
        e = np.array(r["edges_for_cut"], dtype="float64")
        b = pd.cut(s, bins=e, include_lowest=True, duplicates="drop")
        b = b.cat.codes.astype("Int64")
        if include_missing and s.isna().any():
            b = b.where(~s.isna(), -1).astype("Int64")
        DF[col + bin_col_suffix] = b

    # 3) One-hot final des colonnes BIN
    cat_cols = list(res["cat_results"].keys())
    num_cols = list(res["num_results"].keys())
    bin_cols = [c + bin_col_suffix for c in cat_cols + num_cols if c + bin_col_suffix in DF.columns]

    df_model = pd.get_dummies(
        DF.drop(columns=cat_cols + num_cols, errors="ignore"),
        columns=bin_cols,
        prefix={c: c.replace(bin_col_suffix, "") for c in bin_cols},
        dummy_na=False,
        dtype=np.uint8
    )
    # Option : ordre stable
    df_model = df_model.reindex(sorted(df_model.columns), axis=1)
    # Retire IDs
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
    df_base = res["df_enrichi"]  # contient *_BIN
    target = res["target"]

    # calcul des courbes
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

    # plots
    for t, var, g, df_cum in rows:
        plt.figure(figsize=(6, 6))
        plt.plot(df_cum["good_client_share_cumsum"], df_cum["bad_client_share_cumsum"], marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--")  # pas de couleur spécifique
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
# Exemple d'utilisation :
# =======================

# 1) Fit sur train (DF "one-hot" ou mélange)
res = run_full_pipeline_on_onehot_df(
    df_onehot=df_train_imp,
    target_col="default_24m",   # 👈 sera PROTÉGÉ du dé-one-hot
    max_bins_categ=6, min_bin_size_categ=200,
    max_bins_num=6,   min_bin_size_num=200, n_quantiles_num=50,
    include_missing=True, missing_label="__MISSING__", bin_col_suffix="__BIN",
    n_jobs_categ=-1, n_jobs_num=-1,  # ← utilise tous les cœurs disponibles
    verbose=10,
    # min_gini_keep=1e-6,        # ← décommente pour exclure auto les variables trop faibles
)

# 2) Jeu final pour le modèle (X, y)
cols_id = ["loan_sequence_number", "postal_code", "seller_name", "servicer_name", "msa_md"]
df_final = res["df_ohe"].drop(columns=[c for c in cols_id if c in res["df_ohe"].columns], errors="ignore")
y_train = df_final.pop(res["target"]).astype(int)
X_train = df_final

# 3) Transformer validation/test avec les bins appris
df_val_final = transform_with_learned_bins(df_val_imp, res)
y_val = df_val_final.pop(res["target"]).astype(int) if res["target"] in df_val_final.columns else None
X_val = df_val_final.reindex(columns=X_train.columns, fill_value=0)  # aligne les colonnes

## 4) (Optionnel) Plots des 30 meilleures courbes
# plot_all_concentration_curves_from_binned(res, top_n=30)



# --- Imports (nettoyés) ---
from itertools import zip_longest
import numpy as np
import pandas as pd
from IPython.display import display  # pour les aperçus à la fin

# --- WOE par colonne de bins ---
def woe_from_bin(df, target, bcol, smooth=0.5):
    """
    Calcule le WOE par bin pour la colonne bcol (déjà en codes entiers),
    avec lissage additif 'smooth' (>=0) pour éviter 0/0.
    Renvoie une Series alignée sur df.index, contenant le WOE de chaque ligne.
    """
    if bcol not in df.columns:
        raise KeyError(f"Colonne '{bcol}' absente du DataFrame.")

    # Groupby sur le code de bin (les NaN sont ignorés par groupby -> WOE sera NaN sur ces lignes)
    tab = df.groupby(bcol, dropna=True)[target].agg(['sum', 'count'])
    tab['good'] = tab['count'] - tab['sum']

    B = float(tab['sum'].sum())
    G = float(tab['good'].sum())
    K = int(len(tab))

    # Lissage additif -> tout reste strictement > 0 si smooth > 0
    tab['bad_share']  = (tab['sum']  + smooth) / (B + smooth * K if B + smooth * K > 0 else 1.0)
    tab['good_share'] = (tab['good'] + smooth) / (G + smooth * K if G + smooth * K > 0 else 1.0)

    # WOE = ln(bad_share / good_share)
    tab['woe'] = np.log(tab['bad_share'] / tab['good_share']).replace([np.inf, -np.inf], np.nan)

    # Map sur la colonne de bins (les lignes dont le bin n'est pas dans l'index -> NaN)
    return df[bcol].map(tab['woe']).astype(float)

# 1) WOE train après le pipeline
train_enrichi = res["df_enrichi"].copy()
target = res["target"]
bin_cols = [c for c in train_enrichi.columns if c.endswith("__BIN")]

# Construction du DataFrame des WOE (une colonne par variable)
woe_train = pd.DataFrame({
    c.replace("__BIN", "_WOE"): woe_from_bin(train_enrichi, target, c)
    for c in bin_cols
})

# 2) Ordre par “qualité” (Gini train comme proxy ; idéalement Gini OOT)
order_woe = (res["summary"]
             .sort_values("gini_after", ascending=False)["variable"]
             .apply(lambda v: f"{v}_WOE")
             .tolist())
# On garde seulement les colonnes effectivement présentes
order_woe = [v for v in order_woe if v in woe_train.columns]

# 3) Corrélation absolue entre WOE (colonnes constantes -> NaN -> 0)
if len(order_woe) == 0:
    raise RuntimeError("Aucune variable WOE disponible pour la sélection.")
corr = woe_train[order_woe].corr().abs().fillna(0.0)

# 4) Greedy : on garde une seule variable par groupe très corrélé
threshold = 0.85   # ajuste 0.80–0.90 selon ta tolérance
selected = []

for v in order_woe:
    if not selected:
        selected.append(v)
        continue
    # max corr avec le set déjà retenu
    # (utilise .reindex pour robustesse si certaines colonnes manquent)
    mc = corr.loc[v, corr.columns.intersection(selected)]
    max_corr = float(mc.max()) if len(mc) else 0.0
    if not np.isfinite(max_corr) or np.isnan(max_corr):
        max_corr = 0.0
    if max_corr < threshold:
        selected.append(v)

selected_woe_cols = selected                        # ex. ['credit_score_WOE', ...]
selected_vars     = [c.removesuffix("_WOE") for c in selected_woe_cols]  # versions "brutes"
print(f"{len(selected_woe_cols)} variables retenues sur {len(order_woe)}")

# --- Récap + diagnostics ---
def summarize_selection(order_woe, selected_woe_cols, corr, threshold=0.85, save_csv=True, csv_prefix="selection"):
    # Listes WOE
    all_woe     = list(order_woe)
    kept_woe    = list(selected_woe_cols)
    dropped_woe = [v for v in all_woe if v not in set(kept_woe)]

    # Versions "brutes" (sans suffixe _WOE)
    kept_raw    = [v.removesuffix("_WOE") for v in kept_woe]
    dropped_raw = [v.removesuffix("_WOE") for v in dropped_woe]

    # --- Affichages simples
    print(f"\nRésumé sélection : {len(kept_woe)} retenues / {len(all_woe)} évaluées (seuil corr = {threshold})")
    print("\n— Variables retenues (WOE) —")
    for v in kept_woe:
        print("  •", v)
    print("\n— Variables écartées (WOE) —")
    for v in dropped_woe:
        print("  •", v)

    # --- Table récap (alignée)
    df_vars = pd.DataFrame(
        list(zip_longest(kept_raw, dropped_raw, fillvalue="")),
        columns=["kept_raw", "dropped_raw"]
    )

    # --- Diag : pour chaque variable écartée, la retenue la plus corrélée et |ρ|
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

    # Sauvegardes optionnelles
    if save_csv:
        df_vars.to_csv(f"{csv_prefix}_kept_vs_dropped.csv", index=False)
        df_diag.to_csv(f"{csv_prefix}_dropped_diagnostics.csv", index=False)
        print(f"\nFichiers écrits :\n - {csv_prefix}_kept_vs_dropped.csv\n - {csv_prefix}_dropped_diagnostics.csv")

    # Aperçu notebook
    display(df_vars.head(20))
    display(df_diag.head(20))

    return df_vars, df_diag

# === Appel ===
threshold = 0.85  # par exemple
df_vars, df_diag = summarize_selection(order_woe, selected_woe_cols, corr,
                                       threshold=threshold)
# ==== 0) Utilitaires WOE sur les BIN appris ====
import numpy as np
import pandas as pd

def build_woe_maps(df_enrichi, target, smooth=0.5):
    """
    Construit les mappings WOE à partir du TRAIN (df_enrichi contient déjà les *_BIN).
    Retourne un dict: {bcol: {"map": {bin_id: woe}, "default": global_woe}}
    - 'default' = WOE global basé sur y (utile pour bins inconnus en val/test, ex. -2).
    """
    maps = {}
    y = df_enrichi[target].astype(int)
    B_all = float(y.sum())
    G_all = float(len(y) - y.sum())
    # WOE global (avec smooth)
    global_woe = np.log((B_all + smooth) / (G_all + smooth))

    for bcol in [c for c in df_enrichi.columns if c.endswith("__BIN")]:
        if bcol not in df_enrichi.columns:
            continue
        tab = df_enrichi.groupby(bcol, dropna=True)[target].agg(['sum','count'])
        tab['good'] = tab['count'] - tab['sum']

        B = float(tab['sum'].sum())
        G = float(tab['good'].sum())
        K = int(len(tab)) if len(tab) > 0 else 1

        # shares lissées
        denom_bad  = (B + smooth * K) if (B + smooth * K) > 0 else 1.0
        denom_good = (G + smooth * K) if (G + smooth * K) > 0 else 1.0
        tab['bad_share']  = (tab['sum']  + smooth) / denom_bad
        tab['good_share'] = (tab['good'] + smooth) / denom_good

        w = np.log(tab['bad_share'] / tab['good_share']).replace([np.inf, -np.inf], np.nan)
        maps[bcol] = {"map": w.to_dict(), "default": global_woe}
    return maps

def make_enriched_with_bins(df_raw_onehot, res, include_missing=True, bin_col_suffix="__BIN"):
    """
    Reconstruit les colonnes *_BIN sur un DF brut (train-like ou val/test) avec les binnings appris.
    - catégorielles jamais vues -> -2
    - numériques NaN -> -1
    """
    # 1) dé-one-hot (protège la cible si elle existe dans df_raw_onehot)
    DF = deonehot_categoricals(
        df_raw_onehot,
        allow_singleton=False,
        exclude_cols=[res["target"]] if res["target"] in df_raw_onehot.columns else None
    )
    # 2) Catégorielles
    for col, r in res["cat_results"].items():
        if col not in DF.columns:
            continue
        s = DF[col].astype("object").where(DF[col].notna(), "__MISSING__")
        mapped = s.map(r["mapping"]).astype("Int64")
        DF[col + bin_col_suffix] = mapped.fillna(-2).astype("Int64")  # -2 = catégorie jamais vue
    # 3) Numériques
    for col, r in res["num_results"].items():
        if col not in DF.columns:
            continue
        s = _to_float_series(DF[col])
        e = np.array(r["edges_for_cut"], dtype="float64")
        b = pd.cut(s, bins=e, include_lowest=True, duplicates="drop").cat.codes.astype("Int64")
        if include_missing and s.isna().any():
            b = b.where(~s.isna(), -1).astype("Int64")  # -1 = NaN
        DF[col + bin_col_suffix] = b
    return DF

def apply_woe(df_enrichi_with_bins, woe_maps, kept_vars_raw, bin_col_suffix="__BIN"):
    """
    Fabrique la matrice X WOE à partir des BIN + mappings WOE appris sur TRAIN.
    Remplit les NaN / bins inconnus avec le WOE global appris (clé 'default').
    """
    cols = []
    for v in kept_vars_raw:
        bcol = f"{v}{bin_col_suffix}"
        if bcol not in df_enrichi_with_bins.columns or bcol not in woe_maps:
            continue
        ser = df_enrichi_with_bins[bcol].astype("Int64")
        wmap = woe_maps[bcol]["map"]
        wdef = float(woe_maps[bcol]["default"])
        x = ser.map(wmap).astype(float).fillna(wdef)
        cols.append((f"{v}_WOE", x))
    if not cols:
        # aucune variable conservée -> DataFrame vide avec bons index
        return pd.DataFrame(index=df_enrichi_with_bins.index)
    return pd.concat([s for _, s in cols], axis=1)

# ==== 1) Figer la liste des variables retenues (noms bruts) ====
if 'selected_woe_cols' not in globals() or len(selected_woe_cols) == 0:
    raise RuntimeError("La sélection WOE est vide ou non définie. Exécute d'abord la cellule de sélection.")
kept_woe = list(selected_woe_cols)  # ex. ['credit_score_WOE', ...]
kept_vars_raw = [c.removesuffix("_WOE") for c in kept_woe]

# ==== 2) Construire les mappings WOE sur TRAIN puis X/y WOE (train) ====
woe_maps = build_woe_maps(res["df_enrichi"], res["target"])
X_train_woe = apply_woe(res["df_enrichi"], woe_maps, kept_vars_raw)
y_train = res["df_enrichi"][res["target"]].astype(int)

if X_train_woe.shape[1] == 0:
    raise RuntimeError("Aucune variable WOE disponible dans X_train_woe après application des mappings.")

# ==== 3) Préparer la validation : BIN -> WOE en réutilisant les mappings du TRAIN ====
df_val_enrichi = make_enriched_with_bins(df_val_imp, res)     # reconstruit *_BIN sur val/test
X_val_woe = apply_woe(df_val_enrichi, woe_maps, kept_vars_raw)
y_val = df_val_enrichi[res["target"]].astype(int) if res["target"] in df_val_enrichi.columns else None

# Sanity check colonnes (aligne les features)
X_val_woe = X_val_woe.reindex(columns=X_train_woe.columns, fill_value=0.0)

# ==== 4) Entraînement : logistique + tuning AUC + calibration isotonic ====
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = {
    "C": [0.03, 0.1, 0.3, 1, 3, 10],
    "penalty": ["l2"],
    "solver": ["lbfgs"],
    "class_weight": [None, "balanced"],
    "max_iter": [2000],
}
base = LogisticRegression()
gs = GridSearchCV(base, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
gs.fit(X_train_woe, y_train)
best_lr = gs.best_estimator_

# Calibration (isotonic sur folds internes)
cal = CalibratedClassifierCV(best_lr, method="isotonic", cv=cv)
cal.fit(X_train_woe, y_train)

# (optionnel) Prépare aussi p_train pour les cellules PSI/diagnostics ensuite
p_train = cal.predict_proba(X_train_woe)[:, 1]

# ==== 5) Évaluation validation ====
if y_val is not None:
    p_val = cal.predict_proba(X_val_woe)[:, 1]
    auc = roc_auc_score(y_val, p_val)
    gini = 2*auc - 1
    brier = brier_score_loss(y_val, p_val)
    ll = log_loss(y_val, p_val)
    print(f"AUC_val={auc:.4f} | Gini_val={gini:.4f} | Brier={brier:.5f} | LogLoss={ll:.5f}")
else:
    print("Attention : la cible n'est pas présente sur le jeu de validation, métriques non calculées.")







# === Diagnostics post-modèle (KS, déciles, calibration, PSI, importance) ===
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.linear_model import LogisticRegression

# ---------------------------
# Fonctions utilitaires
# ---------------------------
def ks_best_threshold(y, p):
    """
    Renvoie (KS, seuil) basés sur max(TPR-FPR) du ROC.
    Si y est monoclassé, retourne (np.nan, np.nan).
    """
    y = pd.Series(y).astype(int).to_numpy()
    p = pd.Series(p).astype(float).to_numpy()
    # Cas dégénéré : une seule classe
    if np.unique(y).size < 2:
        return np.nan, np.nan
    fpr, tpr, thr = roc_curve(y, p)  # thresholds décroissants
    ks_arr = tpr - fpr
    i = int(np.nanargmax(ks_arr))
    return float(ks_arr[i]), float(thr[i])

def decile_table(y, p, q=10):
    """
    Table déciles (lift, capture, KS cumulatif).
    Robuste aux distributions peu variées (fallback si qcut échoue).
    Déciles ordonnés du plus risqué (haut) au moins risqué (bas).
    """
    df = pd.DataFrame({"y": pd.Series(y).astype(int), "p": pd.Series(p).astype(float)})

    # Tentative principale: qcut (quantiles)
    try:
        df["decile"] = pd.qcut(df["p"], q=q, labels=False, duplicates="drop")
    except Exception:
        # Fallback: découpe sur les rangs (utile si peu de valeurs distinctes)
        # On crée des bornes sur le rang, puis on mappe vers des déciles 0..q-1
        n = len(df)
        if n == 0:
            return pd.DataFrame()
        ranks = df["p"].rank(method="first") / max(n, 1)
        df["decile"] = pd.cut(ranks, bins=np.linspace(0, 1, q+1), labels=False, include_lowest=True)

    # Agrégation par décile
    tab = (df.groupby("decile", dropna=True)
             .agg(events=("y", "sum"), count=("y", "size"), avg_p=("p", "mean"))
             .sort_index(ascending=False))  # haut = probas élevées
    if tab.empty:
        return tab

    tab["rate"] = tab["events"] / tab["count"].where(tab["count"] > 0, 1)
    tab["cum_events"] = tab["events"].cumsum()
    tab["cum_count"]  = tab["count"].cumsum()

    total_events = float(tab["events"].sum())
    total_count  = float(tab["count"].sum())
    tab["capture"]   = tab["cum_events"] / (total_events if total_events > 0 else 1.0)
    tab["cum_share"] = tab["cum_count"]  / (total_count  if total_count  > 0 else 1.0)

    # KS cumulatif (du haut vers le bas)
    cum_good = tab["count"] - tab["events"]
    denom_good = float(cum_good.sum()) if float(cum_good.sum()) > 0 else 1.0
    tab["TPR"] = tab["cum_events"] / (total_events if total_events > 0 else 1.0)
    tab["FPR"] = cum_good.cumsum() / denom_good
    tab["KS"]  = tab["TPR"] - tab["FPR"]
    return tab

def calibration_slope_intercept(y, p, eps=1e-9):
    """
    Estime a,b de : logit(E[y]) = a + b * logit(p).
    Idéalement a≈0, b≈1 si la calibration est parfaite.
    """
    p = np.asarray(p, dtype="float64")
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return np.nan, np.nan
    p_clip = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p_clip / (1 - p_clip)).reshape(-1, 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)  # penalty=None (sklearn>=1.2)
    lr.fit(logit_p, y)
    a = float(lr.intercept_[0])   # idéal ~ 0
    b = float(lr.coef_[0][0])     # idéal ~ 1
    return a, b

def psi(a, b, bins=10, eps=1e-9):
    """
    Population Stability Index entre distributions a (réf: train) et b (ex: val),
    avec bords fondés sur les quantiles de a.
    """
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    a_f = a[np.isfinite(a)]
    b_f = b[np.isfinite(b)]
    if a_f.size == 0 or b_f.size == 0:
        return np.nan

    # Bords par quantiles de a
    q = np.quantile(a_f, np.linspace(0, 1, bins + 1))
    q = np.unique(q)
    if q.size < 2:
        return 0.0
    q[0], q[-1] = -np.inf, np.inf
    for i in range(1, len(q)):
        if not (q[i] > q[i - 1]):
            q[i] = np.nextafter(q[i - 1], np.inf)

    ca, _ = np.histogram(a_f, bins=q)
    cb, _ = np.histogram(b_f, bins=q)
    pa = ca / max(ca.sum(), 1)
    pb = cb / max(cb.sum(), 1)

    ratio = (pa + eps) / (pb + eps)
    return float(np.sum((pa - pb) * np.log(ratio)))

# ---------------------------
# Prérequis
# ---------------------------
if not ('y_val' in globals() and 'p_val' in globals()):
    raise RuntimeError("y_val et p_val doivent être définis (voir la cellule d'entraînement/calibration).")

# ---------------------------
# 1) KS + seuil optimal
# ---------------------------
ks_val, thr_val = ks_best_threshold(y_val, p_val)
if np.isnan(ks_val):
    print("KS non calculable (y_val monoclassé).")
else:
    print(f"KS_val={ks_val:.4f} | seuil_KS={thr_val:.4f}")

# ---------------------------
# 2) Table des déciles + KS (déciles)
# ---------------------------
dec_val = decile_table(y_val, p_val, q=10)
if dec_val.empty:
    print("Impossible de construire la table des déciles (distribution dégénérée).")
else:
    print(dec_val[["count","events","rate","avg_p","capture","KS"]])
    print(f"KS_val (déciles) = {float(dec_val['KS'].max()):.4f}")

# ---------------------------
# 3) Calibration (intercept/slope)
# ---------------------------
a_val, b_val = calibration_slope_intercept(y_val, p_val)
if np.isnan(a_val) or np.isnan(b_val):
    print("Calibration slope/intercept non calculable (y_val monoclassé).")
else:
    print(f"Calibration (val) : intercept={a_val:+.4f} | slope={b_val:.4f}")

# ---------------------------
# 4) PSI train→val (probas)
# ---------------------------
if 'p_train' in globals():
    psi_scores = psi(p_train, p_val, bins=10)
    if np.isnan(psi_scores):
        print("PSI non calculable (distributions vides).")
    else:
        print(f"PSI (train→val, probas) = {psi_scores:.4f}  (≈ <0.1 faible, 0.1–0.25 modéré, >0.25 fort)")
else:
    print("PSI non calculé : variable 'p_train' non définie dans l'environnement.")

# ---------------------------
# 5) Importance des variables (coeffs standardisés)
# ---------------------------
if 'best_lr' in globals() and 'X_train_woe' in globals():
    if getattr(best_lr, "coef_", None) is None:
        print("Importance non calculée : best_lr n'a pas d'attribut coef_.")
    else:
        coefs = pd.Series(best_lr.coef_.ravel(), index=X_train_woe.columns)
        stds  = X_train_woe.std(ddof=0).replace(0, np.nan)
        std_coef = coefs * stds
        imp = (pd.DataFrame({"coef": coefs, "std": stds, "std_coef": std_coef})
               .sort_values("std_coef", key=lambda s: s.abs(), ascending=False))
        print("\nTop 15 variables (|coef|*sd) :")
        print(imp.head(15))
else:
    print("Importance non calculée : 'best_lr' ou 'X_train_woe' manquant.")
