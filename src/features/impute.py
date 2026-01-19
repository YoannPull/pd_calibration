# src/features/impute.py
# -*- coding: utf-8 -*-

"""
=========

Train-only imputation and robust schema alignment for the credit-risk pipeline.

This module defines :class:`DataImputer`, a scikit-learn compatible transformer
(:class:`sklearn.base.BaseEstimator`, :class:`sklearn.base.TransformerMixin`)
designed to be fitted on the TRAIN sample only and then applied consistently to
TRAIN/VALIDATION/OOS datasets without leakage.

Core ideas
----------
- Fit on TRAIN only: all statistics (medians, modes, cohort maps) are learned
  on the training set, then reused as fixed artifacts at inference time.
- Schema stability: at transform-time, missing columns are created and
  unexpected columns are safely ignored, ensuring robust application across
  time/quarters.
- Domain-aware imputation: a small set of features receives specialized
  business logic (credit score, MI%, DTI, CLTV), then a generic safety net
  imputes remaining columns using global TRAIN statistics.
- Optional cohort imputation: if ``use_cohort=True``, the transformer can
  impute using cohort-specific statistics (e.g., by loan purpose and/or year),
  with transparent fallbacks to global TRAIN values.
- Optional missingness indicators: if ``missing_flag=True``, the transformer
  adds binary flags describing which values were missing before imputation.

What it imputes (special rules)
-------------------------------
- ``credit_score``:
  - coerces to numeric and clips to [300, 850]
  - imputes by (year, loan_purpose) if available, then by loan_purpose, then
    global TRAIN median
  - adds ``cs_missing`` flag
- ``mi_percent`` (requires ``original_ltv``):
  - sets MI to 0 when LTV <= 80 and MI is missing (business rule)
  - imputes by (year, LTV bin) if available, then by LTV bin, then 0
  - adds ``mi_missing`` and ``has_mi`` flags
- ``original_dti``:
  - imputes by (year, loan_purpose) then by loan_purpose then global TRAIN median
  - adds ``dti_missing`` flag
- ``original_cltv``:
  - imputes missing CLTV with LTV when available, and enforces CLTV >= LTV
  - optionally imputes by year, then global TRAIN median
  - adds ``cltv_missing`` flag
- Small ordinal features (e.g., ``original_loan_term``, ``number_of_borrowers``):
  - imputes with TRAIN mode after numeric coercion
  - adds ``*_missing`` flags

Generic safety net
------------------
After specialized rules, remaining numeric columns are filled with TRAIN medians,
and non-numeric columns are filled with TRAIN modes. Categorical columns are
expanded to include the chosen fill value when needed.

Input/Output contract
---------------------
Input: pandas DataFrame with raw loan-level features (may include ``vintage``).
Output: DataFrame with the same (aligned) schema and no missing values for the
handled features, plus optional missingness indicators.

Notes
-----
- The transformer intentionally drops ``pre_relief_refi_loan_seq_number`` (if
  present) to avoid using it as a feature.
- Year extraction is robust to Period, datetime, and common string encodings
  (e.g., '2015Q1', '2015-03-31') via a lightweight regex fallback.

"""


import numpy as np
import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_float_dtype,
)
from sklearn.base import BaseEstimator, TransformerMixin


class DataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, use_cohort=True, missing_flag=False, ltv_bins=(0, 80, 90, 95, 100, np.inf)):
        self.use_cohort = use_cohort
        self.ltv_bins = ltv_bins
        self.missing_flag = missing_flag

    # ---------- helpers ----------
    @staticmethod
    def _mode(x):
        try:
            return x.mode(dropna=True).iloc[0]
        except Exception:
            return np.nan

    @staticmethod
    def _to_year(series):
        """
        Return a year Series (Int16) in a deterministic and fast way.

        Handles:
        - Period (M/Q/...)        -> .dt.year
        - datetime64[ns]          -> .dt.year
        - strings: 'YYYYQn', 'YYYY-MM', 'YYYYMM', 'YYYY...' -> first 4 digits

        If nothing is interpretable, returns None (disables any 'by year' logic).
        """
        try:
            s = pd.Series(series)
        except Exception:
            return None

        # Period -> year
        if pd.api.types.is_period_dtype(s):
            return s.dt.year.astype("Int16")

        # Datetime-like -> year
        if pd.api.types.is_datetime64_any_dtype(s):
            return pd.to_datetime(s, errors="coerce").dt.year.astype("Int16")

        # Categorical/object/string -> extract first 4 digits (YYYY)
        ss = s.astype("string").str.strip()
        year_str = ss.str.extract(r"^\s*(\d{4})")[0]
        year = pd.to_numeric(year_str, errors="coerce").astype("Int16")

        # If everything is NA, nothing usable -> disable "by year" logic
        if year.isna().all():
            return None
        return year

    def _map_fill(self, keys_tuple_series, mapping):
        """keys_tuple_series = Series of tuples; mapping = dict {tuple: value}"""
        if mapping is None:
            return pd.Series(np.nan, index=keys_tuple_series.index)
        return keys_tuple_series.map(mapping)

    # ---------- fit on TRAIN ONLY ----------
    def fit(self, X, y=None):
        df = X.copy()

        # Reference columns (exclude a future dropped column)
        self.columns_fit_ = [c for c in df.columns if c != 'pre_relief_refi_loan_seq_number']
        self.dtypes_ = df.dtypes.to_dict()

        # Precompute year if present
        vyear = self._to_year(df['vintage']) if 'vintage' in df.columns else None

        # STORAGE
        self.stats_ = {}

        # CREDIT SCORE medians
        if 'credit_score' in df.columns:
            cs = pd.to_numeric(df['credit_score'], errors='coerce').clip(300, 850)
            self.stats_['credit_score_global'] = float(cs.median())
            self.stats_['credit_score_by_lp'] = None
            self.stats_['credit_score_by_year_lp'] = None

            if self.use_cohort and 'loan_purpose' in df.columns:
                med_lp = cs.groupby(df['loan_purpose'], observed=True).median()
                self.stats_['credit_score_by_lp'] = med_lp.to_dict()

                if vyear is not None:
                    med_y_lp = pd.Series(
                        cs.values,
                        index=pd.MultiIndex.from_arrays([vyear, df['loan_purpose']])
                    ).groupby(level=[0, 1], observed=True).median()
                    self.stats_['credit_score_by_year_lp'] = {k: float(v) for k, v in med_y_lp.items()}

        # MI% medians by LTV bins (and year)
        if 'mi_percent' in df.columns and 'original_ltv' in df.columns:
            mi = pd.to_numeric(df['mi_percent'], errors='coerce')
            ltv = pd.to_numeric(df['original_ltv'], errors='coerce').clip(lower=0)
            ltv_bins = pd.cut(ltv, self.ltv_bins, include_lowest=True, right=True)

            self.stats_['mi_by_bin'] = mi.groupby(ltv_bins, observed=True).median().to_dict()
            self.stats_['mi_by_year_bin'] = None
            if self.use_cohort and vyear is not None:
                idx = pd.MultiIndex.from_arrays([vyear, ltv_bins])
                med = pd.Series(mi.values, index=idx).groupby(level=[0, 1], observed=True).median()
                self.stats_['mi_by_year_bin'] = {k: float(v) for k, v in med.items()}

        # DTI medians
        if 'original_dti' in df.columns:
            dti = pd.to_numeric(df['original_dti'], errors='coerce')
            self.stats_['dti_global'] = float(dti.median())
            self.stats_['dti_by_lp'] = None
            self.stats_['dti_by_year_lp'] = None

            if self.use_cohort and 'loan_purpose' in df.columns:
                self.stats_['dti_by_lp'] = dti.groupby(df['loan_purpose'], observed=True).median().to_dict()
                if vyear is not None:
                    med = pd.Series(
                        dti.values,
                        index=pd.MultiIndex.from_arrays([vyear, df['loan_purpose']])
                    ).groupby(level=[0, 1], observed=True).median()
                    self.stats_['dti_by_year_lp'] = {k: float(v) for k, v in med.items()}

        # CLTV medians by year (fallback global)
        if 'original_cltv' in df.columns:
            cltv = pd.to_numeric(df['original_cltv'], errors='coerce')
            self.stats_['cltv_global'] = float(cltv.median())
            self.stats_['cltv_by_year'] = None
            if self.use_cohort and vyear is not None:
                med = pd.Series(cltv.values, index=vyear).groupby(level=0).median()
                self.stats_['cltv_by_year'] = {int(k): float(v) for k, v in med.items() if pd.notna(v)}

        # Modes for small ordinal variables
        for col in ['original_loan_term', 'number_of_borrowers']:
            if col in df.columns:
                self.stats_[f'{col}_mode'] = self._mode(df[col])

        # Generic fallbacks learned on train
        num_cols = df.select_dtypes(include='number').columns.tolist()
        self.stats_['global_num_median'] = {
            c: float(pd.to_numeric(df[c], errors='coerce').median()) for c in num_cols
        }
        nonnum_cols = [c for c in self.columns_fit_ if c not in num_cols]
        self.stats_['global_nonnum_mode'] = {c: self._mode(df[c]) for c in nonnum_cols}

        return self

    # ---------- transform (apply to TRAIN and TEST) ----------
    def transform(self, X):
        df = X.copy()

        # 0) Drop unused column
        df.drop(columns=['pre_relief_refi_loan_seq_number'], errors='ignore', inplace=True)

        # 0-bis) Schema alignment: add columns seen at fit-time but missing here
        if hasattr(self, 'columns_fit_'):
            for c in self.columns_fit_:
                if c not in df.columns:
                    df[c] = pd.NA

        # "Was missing" flags BEFORE any imputation
        if self.missing_flag:
            cols_impute = df.columns
            missing0 = df[cols_impute].isna().add_prefix('was_missing_').astype('int8')

        # Helpers
        vyear = self._to_year(df['vintage']) if 'vintage' in df.columns else None

        # 1) Categoricals: Unknown / NotApplicable conventions
        if 'channel' in df.columns and isinstance(df['channel'].dtype, CategoricalDtype):
            df['channel'] = df['channel'].cat.add_categories(['Unknown']).fillna('Unknown')

        if 'property_valuation_method' in df.columns:
            pvm = pd.to_numeric(df['property_valuation_method'].astype('string'), errors='coerce')
            if vyear is not None:
                pvm = pvm.where(vyear >= 2017, 99)  # 99 = NotApplicable before 2017
            pvm = pvm.fillna(9)  # 9 = NotAvailable
            df['property_valuation_method'] = pvm.astype('Int16').astype('category')

        if 'special_eligibility_program' in df.columns and isinstance(df['special_eligibility_program'].dtype, CategoricalDtype):
            df['special_eligibility_program'] = df['special_eligibility_program'].cat.add_categories(['Unknown']).fillna('Unknown')
            df['has_special_program'] = df['special_eligibility_program'].isin(['H', 'F', 'R']).astype('int8')

        if 'msa_md' in df.columns:
            df['msa_md'] = df['msa_md'].fillna(0)

        # 2) CREDIT SCORE
        if 'credit_score' in df.columns:
            df['cs_missing'] = df['credit_score'].isna().astype('int8')
            cs = pd.to_numeric(df['credit_score'], errors='coerce').clip(300, 850)

            # Cohort fill
            if self.use_cohort and 'loan_purpose' in df.columns:
                if vyear is not None and self.stats_.get('credit_score_by_year_lp'):
                    keys = pd.Series(list(zip(vyear, df['loan_purpose'])), index=df.index)
                    mapped = self._map_fill(keys, self.stats_['credit_score_by_year_lp'])
                    cs = cs.fillna(mapped)
                if self.stats_.get('credit_score_by_lp'):
                    mapped = df['loan_purpose'].map(self.stats_['credit_score_by_lp'])
                    cs = cs.fillna(mapped)

            # Global fallback
            cs = cs.fillna(self.stats_.get('credit_score_global', float(np.nan)))
            df['credit_score'] = pd.Series(cs, index=df.index).round().astype('Int16')

        # 3) MI%
        if 'mi_percent' in df.columns:
            df['mi_missing'] = df['mi_percent'].isna().astype('int8')
            mi = pd.to_numeric(df['mi_percent'], errors='coerce').astype('Float32')

            ltv = (
                pd.to_numeric(df['original_ltv'], errors='coerce').clip(lower=0)
                if 'original_ltv' in df.columns else pd.Series(np.nan, index=df.index)
            )

            # Business rule: if LTV <= 80 and MI is missing, set to 0
            mi = mi.mask(ltv.le(80) & mi.isna(), 0.0)

            # Cohort median by LTV bins (and year)
            if 'original_ltv' in df.columns:
                ltv_bins = pd.cut(ltv, self.ltv_bins, include_lowest=True, right=True)

                if self.use_cohort and vyear is not None and self.stats_.get('mi_by_year_bin'):
                    keys = pd.Series(list(zip(vyear, ltv_bins)), index=df.index)
                    mapped = self._map_fill(keys, self.stats_['mi_by_year_bin'])
                    mi = mi.fillna(mapped)

                if self.stats_.get('mi_by_bin'):
                    mapped = ltv_bins.map(self.stats_['mi_by_bin'])
                    mi = mi.fillna(mapped)

            df['mi_percent'] = mi.fillna(0.0).astype('Float32')
            df['has_mi'] = (df['mi_percent'] > 0).astype('int8')

        # 4) DTI
        if 'original_dti' in df.columns:
            df['dti_missing'] = df['original_dti'].isna().astype('int8')
            dti = pd.to_numeric(df['original_dti'], errors='coerce')

            if self.use_cohort and 'loan_purpose' in df.columns:
                if vyear is not None and self.stats_.get('dti_by_year_lp'):
                    keys = pd.Series(list(zip(vyear, df['loan_purpose'])), index=df.index)
                    mapped = self._map_fill(keys, self.stats_['dti_by_year_lp'])
                    dti = dti.fillna(mapped)
                if self.stats_.get('dti_by_lp'):
                    mapped = df['loan_purpose'].map(self.stats_['dti_by_lp'])
                    dti = dti.fillna(mapped)

            dti = dti.fillna(self.stats_.get('dti_global', float(np.nan)))
            df['original_dti'] = pd.Series(dti, index=df.index).round().astype('Int16')

        # 5) CLTV
        if 'original_cltv' in df.columns:
            df['cltv_missing'] = df['original_cltv'].isna().astype('int8')
            cltv = pd.to_numeric(df['original_cltv'], errors='coerce').astype('Float32')

            # If CLTV missing, fallback to LTV; also ensure CLTV >= LTV when both are present
            if 'original_ltv' in df.columns:
                ltv = pd.to_numeric(df['original_ltv'], errors='coerce').astype('Float32')
                cltv = pd.Series(cltv, index=df.index).fillna(ltv)
                cltv = pd.Series(np.where(ltv.notna(), np.maximum(cltv, ltv), cltv), index=df.index).astype('Float32')

            if self.use_cohort and vyear is not None and self.stats_.get('cltv_by_year'):
                mapped = pd.Series(vyear, index=df.index).map(self.stats_['cltv_by_year'])
                cltv = pd.Series(cltv, index=df.index).fillna(mapped)

            cltv = pd.Series(cltv, index=df.index).fillna(self.stats_.get('cltv_global', float(np.nan)))
            df['original_cltv'] = cltv.astype('Float32')

        # 6) Small ordinal -> mode (explicit cast to numeric)
        for col in ['original_loan_term', 'number_of_borrowers']:
            if col in df.columns:
                df[col + '_missing'] = df[col].isna().astype('int8')
                mode_val = self.stats_.get(f'{col}_mode', np.nan)

                ser_num = pd.to_numeric(df[col], errors='coerce')
                mode_num = pd.to_numeric(pd.Series([mode_val]), errors='coerce').iloc[0]

                ser_num = ser_num.fillna(mode_num)
                if ser_num.isna().any():
                    ser_num = ser_num.fillna(method='ffill').fillna(method='bfill')

                df[col] = pd.Series(ser_num.round(), index=df.index).astype('Int16')

        # 7) Generic safety net
        num_meds = self.stats_.get('global_num_median', {})
        for c, med in num_meds.items():
            if c in df.columns:
                s = pd.to_numeric(df[c], errors='coerce')
                if pd.notna(med):
                    df[c] = s.fillna(med)
                else:
                    df[c] = s

        non_modes = self.stats_.get('global_nonnum_mode', {})
        for c, mode_val in non_modes.items():
            if c not in df.columns:
                continue

            # Datetime columns
            if is_datetime64_any_dtype(df[c]):
                ser_dt = pd.to_datetime(df[c], errors='coerce')
                if pd.notna(mode_val):
                    df[c] = ser_dt.fillna(mode_val)
                else:
                    df[c] = ser_dt
                continue

            # Categorical columns
            if isinstance(df[c].dtype, CategoricalDtype):
                cat = df[c]
                cats = cat.cat.categories
                cats_dtype = cats.dtype

                if is_integer_dtype(cats_dtype) or is_float_dtype(cats_dtype):
                    if pd.isna(mode_val):
                        if len(cats) == 0:
                            continue
                        fill_val = cats[0]
                    else:
                        try_conv = pd.to_numeric(pd.Series([mode_val]), errors='coerce').iloc[0]
                        if pd.isna(try_conv):
                            fill_val = cats[0] if len(cats) > 0 else None
                        else:
                            if hasattr(cats_dtype, "type"):
                                fill_val = cats_dtype.type(try_conv)
                            elif len(cats) > 0:
                                fill_val = type(cats[0])(try_conv)
                            else:
                                fill_val = try_conv

                    if fill_val is None:
                        continue
                    if fill_val not in cats:
                        cat = cat.cat.add_categories([fill_val])
                    df[c] = cat.fillna(fill_val)
                else:
                    fill_val = mode_val if pd.notna(mode_val) else 'Unknown'
                    if fill_val not in cats:
                        cat = cat.cat.add_categories([fill_val])
                    df[c] = cat.fillna(fill_val)
                continue

            # Other non-numeric columns
            fill_val = mode_val if pd.notna(mode_val) else 'Unknown'
            df[c] = df[c].fillna(fill_val)

        # Add missing-flag features or drop all missing flags depending on configuration
        if self.missing_flag:
            df = pd.concat([df, missing0], axis=1)
        else:
            drop_cols = [c for c in df.columns if c.endswith('_missing') or c.startswith('was_missing_')]
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True, errors='ignore')

        return df
