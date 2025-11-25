# src/features/binning.py
# -*- coding: utf-8 -*-

"""
Module de binning (discrétisation) optimisé pour le risque de crédit.
Implémente :
- Maximisation du Gini (Catégoriel & Numérique)
- Contrainte de Monotonicité (Post-processing)
- Gestion robuste des valeurs manquantes et OOS (Edges infinis)
- Sérialisation JSON robuste aux types NumPy
"""

from __future__ import annotations
import json
import math
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# Parallélisation
try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    def delayed(f): return f

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*is_period_dtype is deprecated.*")

# =============================
# CONSTANTES & CONFIG
# =============================
DENYLIST_STRICT_DEFAULT = [
    "first_payment_date", "maturity_date", "vintage", "mi_cancellation_indicator"
]

EXCLUDE_IDS_DEFAULT: Tuple[str, ...] = (
    "loan_sequence_number", "postal_code", "seller_name", "servicer_name", "msa_md"
)

# =============================
# UTILITAIRES DE CALCUL
# =============================
def gini_trapz(df_cum, y_col="bad_client_share_cumsum", x_col="good_client_share_cumsum", signed=False):
    """Calcule l'indice de Gini via la méthode des trapèzes."""
    df = df_cum[[x_col, y_col]].astype(float).copy().sort_values(x_col)
    # Clip pour éviter les erreurs numériques
    df[x_col] = df[x_col].clip(0, 1)
    df[y_col] = df[y_col].clip(0, 1)
    
    # Ajout des points (0,0) et (1,1) si absents
    if df[x_col].iloc[0] > 0 or df[y_col].iloc[0] > 0:
        df = pd.concat([pd.DataFrame({x_col: [0.0], y_col: [0.0]}), df], ignore_index=True)
    if df[x_col].iloc[-1] < 1 - 1e-12 or df[y_col].iloc[-1] < 1 - 1e-12:
        df = pd.concat([df, pd.DataFrame({x_col: [1.0], y_col: [1.0]})], ignore_index=True)
        
    # Calcul de l'aire sous la courbe ROC (AUC)
    y_vals = df[y_col].to_numpy()
    x_vals = df[x_col].to_numpy()
    
    # Compatibilité numpy versions récentes
    try:
        area = np.trapezoid(y_vals, x_vals)
    except AttributeError:
        area = np.trapz(y_vals, x_vals)
        
    g = 1 - 2 * area
    return g if signed else abs(g)

def to_float_series(s: pd.Series) -> pd.Series:
    """Convertit dates/periods en float (jours), et numeric en float64."""
    if pd.api.types.is_object_dtype(s) or str(s.dtype) == 'category':
         # Tente de convertir les objets numériques
        return pd.to_numeric(s, errors="coerce").astype("float64")
        
    if pd.api.types.is_period_dtype(s):
        ts = s.dt.to_timestamp(how="start")
        # Conversion nanosecondes -> jours (approximatif mais suffisant pour l'ordre)
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

# =============================
# PRÉ-TRAITEMENTS (OneHot, Denylist)
# =============================
def drop_missing_flag_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    mask = cols.str.startswith("was_missing_") | cols.str.endswith("_missing")
    return df.drop(columns=cols[mask], errors="ignore")

def apply_denylist(df: pd.DataFrame, denylist: List[str]) -> pd.DataFrame:
    if not denylist: return df
    return df.drop(columns=[c for c in denylist if c in df.columns], errors="ignore")

def deonehot(df, exclude_cols=None, ambiguous_label=None):
    """Recombine les colonnes One-Hot (ex: 'color_red', 'color_blue') en une seule ('color')."""
    exclude = set(exclude_cols or [])
    groups = {}
    for c in df.columns:
        if c in exclude or "_" not in c: continue
        base, lab = c.rsplit("_", 1)
        s = df[c]
        # Vérification loose pour booléens ou 0/1
        if (pd.api.types.is_bool_dtype(s) or (pd.api.types.is_numeric_dtype(s) and s.dropna().isin([0, 1]).all())):
            groups.setdefault(base, []).append((c, lab))
            
    out = df.copy()
    for base, items in groups.items():
        cols = [c for c, _ in items]
        labels = [lab for _, lab in items]
        # Vérif somme <= 1
        gvals = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        if gvals.sum(axis=1).max() <= 1: # On ne fusionne que si c'est mutuellement exclusif
             ser = pd.Series(pd.NA, index=df.index, dtype="object")
             for c, lab in zip(cols, labels):
                 ser[df[c] == 1] = lab
             out[base] = ser.astype("category")
             out.drop(columns=cols, inplace=True, errors="ignore")
    return out


# =============================
# LOGIQUE DE MONOTONICITÉ
# =============================
def _check_monotonicity(stats_df: pd.DataFrame) -> bool:
    """Retourne True si le bad_rate est monotone (croissant OU décroissant)."""
    if len(stats_df) < 2:
        return True
    rates = stats_df["bad_rate"].values
    # On ignore les bins vides pour la vérif (devraient être rares)
    valid_rates = rates[~np.isnan(rates)]
    if len(valid_rates) < 2:
        return True
        
    diffs = np.diff(valid_rates)
    is_increasing = np.all(diffs >= -1e-9) # tolérance float
    is_decreasing = np.all(diffs <= 1e-9)
    return is_increasing or is_decreasing

def force_monotonicity_numeric(
    df: pd.DataFrame, 
    col: str, 
    target_col: str, 
    edges: List[float], 
    include_missing: bool
) -> List[float]:
    """
    Post-processing : Fusionne itérativement les bins adjacents pour forcer la monotonicité.
    """
    current_edges = sorted(list(set(edges)))
    
    # Sécurité boucle infinie
    max_iter = len(current_edges) + 5
    for _ in range(max_iter):
        # 1. Calculer les stats sur les bins actuels
        # On utilise _gini_from_numeric_bins pour récupérer le dataframe stats
        _, gdf = _gini_from_numeric_bins(
            df[target_col], to_float_series(df[col]), current_edges, include_missing
        )
        
        # On ne garde que les bins numériques pour la monotonicité (le bin missing est à part)
        # _gini_from_numeric_bins met le missing à la fin avec bin=K
        gdf_num = gdf[gdf["bin"] < (len(current_edges) - 1)].copy()
        
        if _check_monotonicity(gdf_num):
            return current_edges

        # 2. Identifier le "violator" (le creux ou la bosse)
        rates = gdf_num["bad_rate"].values
        if len(rates) < 2:
            return current_edges
            
        trend_up = rates[-1] > rates[0]
        
        best_merge_idx = -1
        diffs = np.diff(rates)
        for i, d in enumerate(diffs):
            # Si tendance UP, on cherche un d < 0 (une baisse)
            if trend_up and d < 0:
                best_merge_idx = i + 1
                break
            # Si tendance DOWN, on cherche un d > 0 (une hausse)
            elif not trend_up and d > 0:
                best_merge_idx = i + 1
                break
        
        if best_merge_idx == -1:
            # Cas subtil : zigzag au milieu, on fusionne le premier zigzag trouvé
            for i, d in enumerate(diffs):
                if (trend_up and d < 0) or (not trend_up and d > 0):
                    best_merge_idx = i + 1
                    break
        
        if best_merge_idx != -1 and 0 < best_merge_idx < len(current_edges) - 1:
            # Suppression de la borne
            current_edges.pop(best_merge_idx)
        else:
            break
            
    return current_edges

# =============================
# COEUR DU BINNING (Gini + Numérique)
# =============================

def _safe_edges_for_cut(edges, s_float):
    """Garantit des edges [-inf, ..., inf] pour la robustesse OOS."""
    e = sorted(list(set(edges)))
    if len(e) < 2:
        return np.array([-np.inf, np.inf])
    
    # Remplacement robuste
    e[0] = -np.inf
    e[-1] = np.inf
    return np.array(e, dtype="float64")

def _gini_from_numeric_bins(y, x, edges, include_missing=True):
    """Calcule le Gini et retourne aussi le DataFrame détaillé des stats par bin."""
    y_arr = np.array(y, dtype=int)
    x_arr = np.array(x, dtype=float)
    
    # Digitization
    idx = np.digitize(x_arr, edges[1:-1], right=True)
    
    K = len(edges) - 1
    rows = []
    
    # Stats par bin
    for k in range(K):
        m = (idx == k) & ~np.isnan(x_arr)
        nk = m.sum()
        nb = y_arr[m].sum() if nk > 0 else 0
        rows.append({
            "bin": k, 
            "n_total": nk, 
            "n_bad": nb, 
            "n_good": nk - nb,
            "bad_rate": nb / nk if nk > 0 else 0.0
        })

    # Gestion Missing
    if include_missing:
        m_nan = np.isnan(x_arr)
        nk = m_nan.sum()
        if nk > 0:
            nb = y_arr[m_nan].sum()
            rows.append({
                "bin": K, # Bin spécial à la fin
                "n_total": nk, 
                "n_bad": nb, 
                "n_good": nk - nb,
                "bad_rate": nb / nk
            })
            
    gdf = pd.DataFrame(rows)
    if gdf.empty or gdf["n_total"].sum() == 0:
        return 0.0, gdf

    n_bad_total = gdf["n_bad"].sum()
    n_good_total = gdf["n_good"].sum()
    
    if n_bad_total == 0 or n_good_total == 0:
        return 0.0, gdf

    gdf_sorted = gdf.sort_values("bad_rate").copy()
    gdf_sorted["bad_share"] = gdf_sorted["n_bad"] / n_bad_total
    gdf_sorted["good_share"] = gdf_sorted["n_good"] / n_good_total
    gdf_sorted["bad_cum"] = gdf_sorted["bad_share"].cumsum()
    gdf_sorted["good_cum"] = gdf_sorted["good_share"].cumsum()
    
    g = gini_trapz(gdf_sorted, y_col="bad_cum", x_col="good_cum")
    return g, gdf


def maximize_gini_numeric(
    df, col, target_col,
    max_bins=6, min_bin_size=200, min_bin_frac=None,
    n_quantiles=50, include_missing=True,
    force_mono=True
):
    s = to_float_series(df[col])
    y = df[target_col]
    
    # 1. Initialisation : Bins basés sur les quantiles
    qs = np.linspace(0, 1, n_quantiles + 1)
    cand_vals = np.unique(s.quantile(qs).dropna().values)
    
    if len(cand_vals) > 0:
        min_diff = (cand_vals.max() - cand_vals.min()) * 0.001
        cand_vals = cand_vals[np.concatenate(([True], np.diff(cand_vals) > min_diff))]
    
    # Algo Glouton (Greedy)
    edges = [-np.inf, np.inf]
    best_g, _ = _gini_from_numeric_bins(y, s, edges, include_missing)
    
    n_total = len(s)
    min_size = min_bin_size
    if min_bin_frac:
        min_size = max(min_size, int(n_total * min_bin_frac))

    improved = True
    while improved and (len(edges) - 1) < max_bins:
        improved = False
        best_t = None
        current_best_g = best_g
        
        for t in cand_vals:
            if t <= edges[0] or t >= edges[-1] or t in edges:
                continue
                
            trial_edges = sorted(edges + [t])
            g_try, gdf_try = _gini_from_numeric_bins(y, s, trial_edges, include_missing)
            
            # Vérif taille minimale
            min_n = gdf_try.loc[gdf_try["bin"] < (len(trial_edges)-1), "n_total"].min()
            if min_n < min_size:
                continue
                
            if g_try > current_best_g + 1e-5:
                current_best_g = g_try
                best_t = t
                improved = True
        
        if improved and best_t is not None:
            edges = sorted(edges + [best_t])
            best_g = current_best_g

    # 2. Forçage de la Monotonicité
    if force_mono and (len(edges) - 1) > 1:
        edges = force_monotonicity_numeric(df, col, target_col, edges, include_missing)
        best_g, _ = _gini_from_numeric_bins(y, s, edges, include_missing)

    # 3. Finalisation des edges
    e_cut = _safe_edges_for_cut(edges, s)
    
    return {
        "edges": edges,
        "edges_for_cut": e_cut.tolist(),
        "gini_final": float(best_g)
    }


# =============================
# BINNING CATÉGORIEL
# =============================
def _cat_stats(df, col, target_col, include_missing=True, missing_label="__MISSING__"):
    y = df[target_col].astype(int)
    s = df[col]
    if include_missing:
        # Warning fix: use infer_objects
        s = s.astype("object").fillna(missing_label).infer_objects(copy=False)
    
    tmp = pd.DataFrame({"mod": s, "tgt": y})
    agg = tmp.groupby("mod", observed=True)["tgt"].agg(["sum", "count"])
    agg.columns = ["n_bad", "n_total"]
    agg["n_good"] = agg["n_total"] - agg["n_bad"]
    agg["bad_rate"] = agg["n_bad"] / agg["n_total"]
    
    return agg.reset_index().rename(columns={"mod": "modality"})

def maximize_gini_categorical(
    df, col, target_col,
    include_missing=True, missing_label="__MISSING__",
    max_bins=6, min_bin_size=200, min_bin_frac=None
):
    stats = _cat_stats(df, col, target_col, include_missing, missing_label)
    stats = stats.sort_values("bad_rate").reset_index(drop=True)
    
    bins = [[m] for m in stats["modality"]]
    
    while len(bins) > max_bins:
        best_i = -1
        min_diff = float('inf')
        
        current_stats = []
        for b in bins:
            sub = stats[stats["modality"].isin(b)]
            nb = sub["n_bad"].sum()
            nt = sub["n_total"].sum()
            current_stats.append(nb/nt if nt>0 else 0)
            
        for i in range(len(current_stats)-1):
            d = abs(current_stats[i] - current_stats[i+1])
            if d < min_diff:
                min_diff = d
                best_i = i
        
        if best_i != -1:
            bins[best_i] = bins[best_i] + bins[best_i+1]
            bins.pop(best_i+1)
        else:
            break
            
    mapping = {}
    for i, mod_list in enumerate(bins):
        for m in mod_list:
            # FIX JSON SERIALIZATION: Convert numpy types to native types for keys
            key = m.item() if hasattr(m, "item") else m
            mapping[key] = i
            
    return {
        "mapping": mapping,
        "bins": bins,
        "gini_final": 0.0 # Placeholder
    }


# =============================
# PIPELINE PRINCIPAL & WRAPPERS
# =============================
@dataclass
class LearnedBins:
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
    **kwargs
):
    DF = df.copy()
    if drop_missing_flags:
        DF = drop_missing_flag_columns(DF)
    if denylist_strict:
        DF = apply_denylist(DF, denylist_strict)
    
    DF = deonehot(DF, exclude_cols=[target_col])
    
    cat_cols = []
    num_cols = []
    exclude_set = set(list(exclude_ids) + [target_col])
    
    for c in DF.columns:
        if c in exclude_set: continue
        s = DF[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 10:
            num_cols.append(c)
        else:
            cat_cols.append(c)

    cat_results = {}
    for c in cat_cols:
        cat_results[c] = maximize_gini_categorical(
            DF[[c, target_col]], c, target_col, 
            include_missing, missing_label, 
            max_bins_categ, min_bin_size_categ
        )
        
    num_results = {}
    for c in num_cols:
        num_results[c] = maximize_gini_numeric(
            DF[[c, target_col]], c, target_col,
            max_bins_num, min_bin_size_num, None, 
            n_quantiles_num, include_missing, 
            force_mono=True
        )

    enriched = DF.copy()
    
    for c, res in cat_results.items():
        # Warning fix
        s = enriched[c].astype("object").fillna(missing_label).infer_objects(copy=False)
        mapped = s.map(res["mapping"]).fillna(-1).astype("Int64")
        enriched[c + bin_col_suffix] = mapped
        
    for c, res in num_results.items():
        s = to_float_series(enriched[c])
        e = res["edges_for_cut"]
        b = pd.cut(s, bins=e, labels=False, include_lowest=True).astype("Int64")
        if include_missing and s.isna().any():
            b = b.fillna(-1)
        enriched[c + bin_col_suffix] = b

    bin_cols = [c for c in enriched.columns if c.endswith(bin_col_suffix)]
    final_df = enriched[bin_cols + [target_col]].copy()
    
    learned = LearnedBins(
        target=target_col,
        include_missing=include_missing,
        missing_label=missing_label,
        bin_col_suffix=bin_col_suffix,
        cat_results=cat_results,
        num_results=num_results
    )
    
    return learned, enriched, final_df


def transform_with_learned_bins(df: pd.DataFrame, learned: LearnedBins) -> pd.DataFrame:
    """Applique les bins appris sur un nouveau dataset (Val / Test / OOS)."""
    DF = df.copy()
    suffix = learned.bin_col_suffix
    
    for c, res in learned.cat_results.items():
        if c in DF.columns:
            # Warning fix
            s = DF[c].astype("object").fillna(learned.missing_label).infer_objects(copy=False)
            DF[c + suffix] = s.map(res["mapping"]).fillna(-1).astype("Int64")
            
    for c, res in learned.num_results.items():
        if c in DF.columns:
            s = to_float_series(DF[c])
            e = res["edges_for_cut"]
            b = pd.cut(s, bins=e, labels=False, include_lowest=True).astype("Int64")
            if learned.include_missing:
                b = b.fillna(-1)
            DF[c + suffix] = b
            
    cols = [c for c in DF.columns if c.endswith(suffix)]
    if learned.target in DF.columns:
        cols.append(learned.target)
        
    return DF[cols]


# =============================
# I/O JSON ROBUSTE
# =============================
def save_bins_json(learned: LearnedBins, path: str):
    
    # 1. Helper pour convertir les VALEURS (numpy -> python)
    def default(o):
        if hasattr(o, "item"): return o.item()
        if isinstance(o, (np.ndarray,)): return o.tolist()
        return str(o)

    # 2. Helper pour convertir les CLÉS de dictionnaires (recursif)
    def clean_keys(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                # Convertir la clé si c'est un numpy scalar
                k_clean = k.item() if hasattr(k, "item") else k
                new_dict[k_clean] = clean_keys(v)
            return new_dict
        elif isinstance(obj, list):
            return [clean_keys(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(clean_keys(i) for i in obj)
        return obj

    data = {
        "target": learned.target,
        "include_missing": learned.include_missing,
        "missing_label": learned.missing_label,
        "bin_col_suffix": learned.bin_col_suffix,
        "cat_results": learned.cat_results,
        "num_results": learned.num_results
    }
    
    # Nettoyage profond des clés
    clean_data = clean_keys(data)

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=2, default=default)

def load_bins_json(path: str) -> LearnedBins:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return LearnedBins(**data)