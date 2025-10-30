# src/features/binning.py
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin


@dataclass
class VarBinningInfo:
    kind: str  # "numeric" | "categorical"
    iv: float
    # numeric
    edges: Optional[List[float]] = None     # len = nbins+1
    woe_per_interval: Optional[Dict[pd.Interval, float]] = None
    # categorical
    woe_per_category: Optional[Dict[Union[str, int, float], float]] = None
    # specials
    woe_nan: float = 0.0
    woe_unknown: float = 0.0


class BinningTransformer(BaseEstimator, TransformerMixin):
    """
    Binning + WOE pour variables num (qcut) & cat (woe catégorie).
    - fit(X, y): apprend edges / woe / iv
    - transform(X): renvoie DataFrame WOE-encodé pour les variables ciblées
    """
    def __init__(
        self,
        variables: Optional[List[str]] = None,
        n_bins: int = 10,
        output: str = "woe",            # "woe" | "bin_index" | "both"
        min_unique: int = 2,           # min de modalités/valeurs pour tenter un binning
        smoothing: float = 0.5,        # lissage add-eps pour woe
        include_categorical: bool = True,
        drop_original: bool = False,    # True -> ne garder que les features encodées
        prefix: str = "woe__",          # préfixe des colonnes de sortie
    ):
        self.variables = variables
        self.n_bins = n_bins
        self.output = output
        self.min_unique = min_unique
        self.smoothing = smoothing
        self.include_categorical = include_categorical
        self.drop_original = drop_original
        self.prefix = prefix

        # learned
        self.vars_: List[str] = []
        self.info_: Dict[str, VarBinningInfo] = {}
        self.target_rate_: Optional[float] = None
        self.classes_: Optional[np.ndarray] = None

    # --- utils ---
    @staticmethod
    def _safe_qcut(x, q, duplicates="drop"):
        # qcut peut échouer si beaucoup de doublons: on catch et on degrade en cut
        try:
            return pd.qcut(x, q=q, duplicates=duplicates)
        except Exception:
            # fallback: bins égaux (si distrib très plate)
            try:
                return pd.cut(x, bins=q, duplicates=duplicates)
            except Exception:
                return pd.Series(pd.IntervalIndex([], closed="right"), index=x.index, dtype="category")

    @staticmethod
    def _woe_iv(event, non_event, tot_event, tot_non_event, eps):
        # distributions
        pe = (event + eps) / (tot_event + eps * 2)
        pne = (non_event + eps) / (tot_non_event + eps * 2)
        w = np.log(pe / pne)
        iv = (pe - pne) * w
        return float(w), float(iv)

    def _fit_numeric(self, x: pd.Series, y: pd.Series, name: str, tot_event: int, tot_non_event: int) -> VarBinningInfo:
        # bins (qcut -> fréquences ~ égales)
        bins = self._safe_qcut(x, q=self.n_bins, duplicates="drop")
        if bins.dtype == "category" and len(bins.cat.categories) == 0:
            # variable quasi-constante -> pas de binning
            # WOE unique = WOE global
            eps = self.smoothing
            info = VarBinningInfo(kind="numeric", iv=0.0, edges=None, woe_per_interval={})
            pe = (y.sum() + eps) / (tot_event + eps * 2)
            pne = ((y.shape[0]-y.sum()) + eps) / (tot_non_event + eps * 2)
            w_global = float(np.log(pe / pne))
            info.woe_nan = w_global
            info.woe_unknown = w_global
            info.iv = 0.0
            return info

        # stats par intervalle
        df = pd.DataFrame({"bin": bins, "y": y})
        grp = df.groupby("bin", observed=True)
        cnt = grp["y"].count()
        ev = grp["y"].sum()
        ne = cnt - ev

        eps = self.smoothing
        iv_total = 0.0
        w_map = {}
        for interval in cnt.index:
            w, iv = self._woe_iv(ev.loc[interval], ne.loc[interval], tot_event, tot_non_event, eps)
            w_map[interval] = w
            iv_total += iv

        edges = [c.left for c in cnt.index] + [cnt.index[-1].right]

        # woe pour NaN / unknown = woe global
        pe = (y.sum() + eps) / (tot_event + eps * 2)
        pne = ((y.shape[0]-y.sum()) + eps) / (tot_non_event + eps * 2)
        w_global = float(np.log(pe / pne))

        return VarBinningInfo(
            kind="numeric",
            iv=float(iv_total),
            edges=[float(e) if pd.notna(e) else -np.inf for e in edges],
            woe_per_interval=w_map,
            woe_nan=w_global,
            woe_unknown=w_global,
        )

    def _fit_categorical(self, x: pd.Series, y: pd.Series, name: str, tot_event: int, tot_non_event: int) -> VarBinningInfo:
        df = pd.DataFrame({"cat": x.astype("object"), "y": y})
        grp = df.groupby("cat", dropna=False)
        cnt = grp["y"].count()
        ev = grp["y"].sum()
        ne = cnt - ev

        eps = self.smoothing
        iv_total = 0.0
        w_map = {}
        for cat in cnt.index:
            # cat peut être NaN -> on traite séparément
            e = ev.loc[cat]
            n = ne.loc[cat]
            w, iv = self._woe_iv(e, n, tot_event, tot_non_event, eps)
            if pd.isna(cat):
                nan_woe = w
                continue
            w_map[cat] = w
            iv_total += iv

        # woe global comme fallback unknown
        pe = (y.sum() + eps) / (tot_event + eps * 2)
        pne = ((y.shape[0]-y.sum()) + eps) / (tot_non_event + eps * 2)
        w_global = float(np.log(pe / pne))

        return VarBinningInfo(
            kind="categorical",
            iv=float(iv_total),
            edges=None,
            woe_per_interval=None,
            woe_per_category=w_map,
            woe_nan=nan_woe if "nan_woe" in locals() else w_global,
            woe_unknown=w_global,
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X = X.copy()
        y = pd.Series(y).astype(int)
        self.classes_ = np.unique(y.values)

        tot_event = int(y.sum())
        tot_non_event = int((y == 0).sum())
        self.target_rate_ = float(tot_event / max(len(y), 1))

        # variables cibles
        if self.variables is None:
            # par défaut: num + cat (si include_categorical)
            vars_num = [c for c in X.columns if is_numeric_dtype(X[c]) and X[c].nunique(dropna=True) >= self.min_unique]
            vars_cat = []
            if self.include_categorical:
                vars_cat = [c for c in X.columns if (not is_numeric_dtype(X[c])) and X[c].nunique(dropna=True) >= self.min_unique]
            variables = vars_num + vars_cat
        else:
            variables = [c for c in self.variables if c in X.columns]

        self.vars_ = variables
        self.info_ = {}

        for name in variables:
            s = X[name]
            if is_numeric_dtype(s):
                info = self._fit_numeric(pd.to_numeric(s, errors="coerce"), y, name, tot_event, tot_non_event)
            else:
                info = self._fit_categorical(s, y, name, tot_event, tot_non_event)
            self.info_[name] = info

        return self

    def _transform_numeric(self, s: pd.Series, info: VarBinningInfo):
        # si pas d'edges => constante: woe global
        if info.edges is None or info.woe_per_interval is None:
            w = pd.Series(info.woe_unknown, index=s.index)
            w[s.isna()] = info.woe_nan
            return w, pd.Series(-1, index=s.index, dtype="Int16")

        # couper selon edges appris
        edges = info.edges
        # assurer bornes infinies
        edges = [(-np.inf if i == 0 else edges[i]) for i in range(len(edges))]  # on gardera -inf via cut anyway
        cat = pd.cut(s, bins=info.edges, include_lowest=True)

        # map WOE
        w_map = info.woe_per_interval
        w = cat.map(w_map)
        # NaN (valeur manquante initiale)
        w[s.isna()] = info.woe_nan
        # hors bornes (très rare si edges couvrent tout)
        w = w.fillna(info.woe_unknown)

        # indice de bin (optionnel)
        if cat.dtype == "category":
            bin_idx = pd.Series(cat.cat.codes, index=s.index).astype("Int16")
        else:
            bin_idx = pd.Series(-1, index=s.index, dtype="Int16")

        return w, bin_idx

    def _transform_categorical(self, s: pd.Series, info: VarBinningInfo):
        w = s.map(info.woe_per_category)
        w[s.isna()] = info.woe_nan
        w = w.fillna(info.woe_unknown)
        # Pas de notion d'index de bin pour cat -> codes de catégorie si besoin
        bin_idx = pd.Series(s.astype("category").cat.codes, index=s.index).astype("Int16")
        return w, bin_idx

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        out = pd.DataFrame(index=X.index)

        for name in self.vars_:
            if name not in X.columns:
                # colonne absente -> woe unknown
                info = self.info_[name]
                w = pd.Series(info.woe_unknown, index=X.index)
                b = pd.Series(-1, index=X.index, dtype="Int16")
            else:
                s = X[name]
                info = self.info_[name]
                if info.kind == "numeric":
                    w, b = self._transform_numeric(pd.to_numeric(s, errors="coerce"), info)
                else:
                    w, b = self._transform_categorical(s, info)

            if self.output in ("woe", "both"):
                out[f"{self.prefix}{name}"] = w.astype("float32")
            if self.output in ("bin_index", "both"):
                out[f"bin__{name}"] = b

        if not self.drop_original:
            # concaténer avec X
            out = pd.concat([X, out], axis=1)

        return out

    # Qualité / reporting
    def iv_summary_(self) -> pd.DataFrame:
        rows = []
        for name, info in self.info_.items():
            rows.append({"variable": name, "kind": info.kind, "iv": info.iv})
        return pd.DataFrame(rows).sort_values("iv", ascending=False)

    # Compat Pipeline
    def get_feature_names_out(self, input_features=None):
        if self.output == "woe":
            return np.array([f"{self.prefix}{v}" for v in self.vars_], dtype=object)
        if self.output == "bin_index":
            return np.array([f"bin__{v}" for v in self.vars_], dtype=object)
        both = [f"{self.prefix}{v}" for v in self.vars_] + [f"bin__{v}" for v in self.vars_]
        return np.array(both, dtype=object)
