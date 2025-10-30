import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype, is_datetime64_any_dtype
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
        try:
            ser = pd.to_datetime(series, errors='coerce')
            return ser.dt.year
        except Exception:
            return None

    def _map_fill(self, keys_tuple_series, mapping):
        """keys_tuple_series = Series of tuples; mapping = dict {tuple: value}"""
        if mapping is None:
            return pd.Series(np.nan, index=keys_tuple_series.index)
        return keys_tuple_series.map(mapping)

    # ---------- fit on TRAIN ONLY ----------
    def fit(self, X, y=None):
        df = X.copy()

        # Colonnes de référence (exclure la colonne drop future)
        self.columns_fit_ = [c for c in df.columns if c != 'pre_relief_refi_loan_seq_number']
        self.dtypes_ = df.dtypes.to_dict()  # utile si besoin de debug/trace

        # Precompute year if present
        vyear = self._to_year(df['vintage']) if 'vintage' in df.columns else None

        # STORAGE
        self.stats_ = {}

        # ---- CREDIT SCORE medians
        if 'credit_score' in df.columns:
            cs = pd.to_numeric(df['credit_score'], errors='coerce').clip(300, 850)
            self.stats_['credit_score_global'] = float(cs.median())
            self.stats_['credit_score_by_lp'] = None
            self.stats_['credit_score_by_year_lp'] = None

            if self.use_cohort and 'loan_purpose' in df.columns:
                med_lp = cs.groupby(df['loan_purpose']).median()
                self.stats_['credit_score_by_lp'] = med_lp.to_dict()

                if vyear is not None:
                    med_y_lp = pd.Series(cs.values,
                                         index=pd.MultiIndex.from_arrays([vyear, df['loan_purpose']])
                                         ).groupby(level=[0, 1]).median()
                    self.stats_['credit_score_by_year_lp'] = {k: float(v) for k, v in med_y_lp.items()}

        # ---- MI% medians by LTV bins (and year)
        if 'mi_percent' in df.columns and 'original_ltv' in df.columns:
            mi = pd.to_numeric(df['mi_percent'], errors='coerce')
            ltv = pd.to_numeric(df['original_ltv'], errors='coerce').clip(lower=0)
            ltv_bins = pd.cut(ltv, self.ltv_bins, include_lowest=True, right=True)

            self.stats_['mi_by_bin'] = mi.groupby(ltv_bins).median().to_dict()
            self.stats_['mi_by_year_bin'] = None
            if self.use_cohort and vyear is not None:
                idx = pd.MultiIndex.from_arrays([vyear, ltv_bins])
                med = pd.Series(mi.values, index=idx).groupby(level=[0, 1]).median()
                self.stats_['mi_by_year_bin'] = {k: float(v) for k, v in med.items()}

        # ---- DTI medians
        if 'original_dti' in df.columns:
            dti = pd.to_numeric(df['original_dti'], errors='coerce')
            self.stats_['dti_global'] = float(dti.median())
            self.stats_['dti_by_lp'] = None
            self.stats_['dti_by_year_lp'] = None

            if self.use_cohort and 'loan_purpose' in df.columns:
                self.stats_['dti_by_lp'] = dti.groupby(df['loan_purpose']).median().to_dict()
                if vyear is not None:
                    med = pd.Series(dti.values,
                                    index=pd.MultiIndex.from_arrays([vyear, df['loan_purpose']])
                                    ).groupby(level=[0, 1]).median()
                    self.stats_['dti_by_year_lp'] = {k: float(v) for k, v in med.items()}

        # ---- CLTV medians by year (fallback global)
        if 'original_cltv' in df.columns:
            cltv = pd.to_numeric(df['original_cltv'], errors='coerce')
            self.stats_['cltv_global'] = float(cltv.median())
            self.stats_['cltv_by_year'] = None
            if self.use_cohort and vyear is not None:
                med = pd.Series(cltv.values, index=vyear).groupby(level=0).median()
                self.stats_['cltv_by_year'] = {int(k): float(v) for k, v in med.items() if pd.notna(v)}

        # ---- modes for small ordinal
        for col in ['original_loan_term', 'number_of_borrowers']:
            if col in df.columns:
                self.stats_[f'{col}_mode'] = self._mode(df[col])

        # ---- NEW: Fallbacks génériques appris sur le train
        # Numériques -> médiane globale ; Non-numériques -> mode global
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

        # 0) Drop colonne non utilisée
        df.drop(columns=['pre_relief_refi_loan_seq_number'], errors='ignore', inplace=True)

        # 0-bis) Alignement de schéma : ajouter les colonnes vues au fit mais absentes ici
        if hasattr(self, 'columns_fit_'):
            for c in self.columns_fit_:
                if c not in df.columns:
                    df[c] = pd.NA

        # Flags "était manquant" avant toute imputation
        if self.missing_flag:
            cols_impute = df.columns
            missing0 = df[cols_impute].isna().add_prefix('was_missing_').astype('int8')

        # Helpers
        vyear = self._to_year(df['vintage']) if 'vintage' in df.columns else None

        # 1) Catés : Unknown / NotApplicable
        if 'channel' in df.columns and isinstance(df['channel'].dtype, CategoricalDtype):
            df['channel'] = df['channel'].cat.add_categories(['Unknown']).fillna('Unknown')

        if 'property_valuation_method' in df.columns:
            pvm = pd.to_numeric(df['property_valuation_method'].astype('string'), errors='coerce')
            if vyear is not None:
                pvm = pvm.where(vyear >= 2017, 99)  # 99 NotApplicable avant 2017
            pvm = pvm.fillna(9)  # 9 NotAvailable
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

            # cohort fill
            if self.use_cohort and 'loan_purpose' in df.columns:
                if vyear is not None and self.stats_.get('credit_score_by_year_lp'):
                    keys = pd.Series(list(zip(vyear, df['loan_purpose'])), index=df.index)
                    mapped = self._map_fill(keys, self.stats_['credit_score_by_year_lp'])
                    cs = cs.fillna(mapped)
                if self.stats_.get('credit_score_by_lp'):
                    mapped = df['loan_purpose'].map(self.stats_['credit_score_by_lp'])
                    cs = cs.fillna(mapped)

            # global fallback
            cs = cs.fillna(self.stats_.get('credit_score_global', float(np.nan)))
            df['credit_score'] = pd.Series(cs, index=df.index).round().astype('Int16')

        # 3) MI%
        if 'mi_percent' in df.columns:
            df['mi_missing'] = df['mi_percent'].isna().astype('int8')
            mi = pd.to_numeric(df['mi_percent'], errors='coerce').astype('Float32')

            ltv = pd.to_numeric(df['original_ltv'], errors='coerce').clip(lower=0) if 'original_ltv' in df.columns else pd.Series(np.nan, index=df.index)
            # règle métier
            mi = mi.mask(ltv.le(80) & mi.isna(), 0.0)

            # cohort median by LTV bins (and year)
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

            if 'original_ltv' in df.columns:
                ltv = pd.to_numeric(df['original_ltv'], errors='coerce').astype('Float32')
                cltv = pd.Series(cltv, index=df.index).fillna(ltv)
                cltv = pd.Series(np.where(ltv.notna(), np.maximum(cltv, ltv), cltv), index=df.index).astype('Float32')

            if self.use_cohort and vyear is not None and self.stats_.get('cltv_by_year'):
                mapped = pd.Series(vyear, index=df.index).map(self.stats_['cltv_by_year'])
                cltv = pd.Series(cltv, index=df.index).fillna(mapped)

            cltv = pd.Series(cltv, index=df.index).fillna(self.stats_.get('cltv_global', float(np.nan)))
            df['original_cltv'] = cltv.astype('Float32')

        # 6) Small ordinal → mode
        for col in ['original_loan_term', 'number_of_borrowers']:
            if col in df.columns:
                df[col + '_missing'] = df[col].isna().astype('int8')
                mode_val = self.stats_.get(f'{col}_mode', np.nan)
                df[col] = df[col].fillna(mode_val)
                # si tout était NaN dans le train, sécurité
                if df[col].isna().any():
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        # 7) NEW: Filet de sécurité générique (pour TOUTES les colonnes restantes)
        #    - Numériques: médiane globale apprise sur le train
        #    - Non-numériques: mode global; si pas dispo -> 'Unknown' (sauf datetime qu'on laisse si pas de mode)
        num_meds = self.stats_.get('global_num_median', {})
        for c, med in num_meds.items():
            if c in df.columns:
                s = pd.to_numeric(df[c], errors='coerce')
                if pd.notna(med):
                    df[c] = s.fillna(med)
                else:
                    df[c] = s  # pas de médiane dispo (colonne 100% NaN au train)

        non_modes = self.stats_.get('global_nonnum_mode', {})
        for c, mode_val in non_modes.items():
            if c in df.columns:
                if is_datetime64_any_dtype(df[c]):
                    # Datetime: on remplit avec le mode si dispo, sinon on laisse NaT
                    if pd.notna(mode_val):
                        df[c] = pd.to_datetime(df[c], errors='coerce').fillna(mode_val)
                elif isinstance(df[c].dtype, CategoricalDtype):
                    fill_val = mode_val if pd.notna(mode_val) else 'Unknown'
                    df[c] = df[c].cat.add_categories([fill_val]).fillna(fill_val)
                else:
                    fill_val = mode_val if pd.notna(mode_val) else 'Unknown'
                    df[c] = df[c].fillna(fill_val)

        # Added missing flag
        if self.missing_flag:
            df = pd.concat([df, missing0], axis=1)

        return df
