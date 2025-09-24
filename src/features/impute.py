# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict
import re
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

# ---------------------------------------------------------------------
# Parsing / typing helpers
# ---------------------------------------------------------------------
def yn_space_to_bool(s: pd.Series, na_vals=('9',)):
    """
    Convert FHA-style Y/N/blank flags to pandas 'boolean' dtype.
    - Trims whitespace, uppercases, and maps:
        'Y' -> True, 'N' -> False, '' (blank) -> False (by convention)
    - Any value in `na_vals` (e.g., '9') becomes <NA>.
    """
    s = s.astype('string').str.strip().str.upper()
    s = s.fillna('').replace({' ': ''})
    out = s.map({'Y': True, 'N': False, '': False})
    out[s.isin(na_vals)] = pd.NA
    return out.astype('boolean')

def to_periodM(col: pd.Series) -> pd.PeriodIndex:
    """
    Coerce strings like 'YYYYMM' to a monthly PeriodIndex (period[M]).
    Invalid values are set to <NA>.
    """
    s = col.astype('string').str.strip()
    s = s.where(s.str.fullmatch(r'\d{6}'), pd.NA)   # YYYYMM
    return pd.PeriodIndex(s, freq='M')

def to_periodQ(col: pd.Series) -> pd.PeriodIndex:
    """
    Coerce strings like 'YYYYQ[1-4]' to a quarterly PeriodIndex (period[Q]).
    Invalid values are set to <NA>.
    """
    s = col.astype('string').str.strip().str.upper()
    return pd.PeriodIndex(s.where(s.str.fullmatch(r'\d{4}Q[1-4]'), pd.NA), freq='Q')

# ---------------------------------------------------------------------
# Type coercion & light cleaning (schema normalization)
# ---------------------------------------------------------------------
def coerce_base_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make a safe copy of the input DataFrame and coerce common columns to:
      - Periods for date-like columns
      - Boolean flags for Y/N/blank indicators
      - Nullable integer types for sentinel-coded numerics
      - Categorical types for enumerations
      - String for IDs
    This function does not perform imputation; it only normalizes dtypes and
    replaces known sentinel values with <NA>.
    """
    df = df.copy()

    # ---- Dates / periods (store as period types for monthly/quarterly ops)
    if 'first_payment_date' in df:
        df['first_payment_date'] = pd.to_datetime(df['first_payment_date'], errors="coerce").dt.to_period("M")
    if 'maturity_date' in df:
        df['maturity_date'] = to_periodM(df['maturity_date'])
    if 'vintage' in df:
        df['vintage'] = to_periodQ(df['vintage'])

    # ---- Y/N/blank -> boolean (nullable)
    if 'ppm_flag' in df:
        df['ppm_flag'] = yn_space_to_bool(df['ppm_flag'], na_vals=())
    if 'interest_only_indicator' in df:
        df['interest_only_indicator'] = yn_space_to_bool(df['interest_only_indicator'], na_vals=())
    if 'super_conforming_flag' in df:
        df['super_conforming_flag'] = yn_space_to_bool(df['super_conforming_flag'], na_vals=())
    if 'first_time_homebuyer_flag' in df:
        df['first_time_homebuyer_flag'] = yn_space_to_bool(df['first_time_homebuyer_flag'], na_vals=('9',))
    if 'relief_refinance_indicator' in df:
        df['relief_refinance_indicator'] = yn_space_to_bool(df['relief_refinance_indicator'])

    # ---- Target as boolean if provided as 0/1 (kept optional)
    if 'default_24m' in df and (pd.api.types.is_integer_dtype(df['default_24m']) or pd.api.types.is_bool_dtype(df['default_24m'])):
        df['default_24m'] = df['default_24m'].map({1: True, 0: False}).astype('boolean')

    # ---- Sentinel replacement -> <NA>, then cast to compact nullable integers
    # Common sentinels from GSE specs: 9999 (credit score), 999 (MI, LTV/DTI/CLTV),
    # 99 (num units). Adjust if your upstream changes.
    for col, sent, dtype in [
        ('credit_score', 9999, 'Int16'),
        ('mi_percent', 999, 'Int16'),
        ('number_of_units', 99, 'Int8'),
        ('original_cltv', 999, 'Int16'),
        ('original_dti', 999, 'Int16'),
        ('original_ltv', 999, 'Int16'),
    ]:
        if col in df:
            df[col] = df[col].replace({sent: pd.NA}).astype(dtype)

    # Other numeric ints (nullable)
    if 'original_loan_term' in df:
        df['original_loan_term'] = df['original_loan_term'].astype('Int16')
    if 'number_of_borrowers' in df:
        df['number_of_borrowers'] = df['number_of_borrowers'].astype('Int8')
    if 'original_upb' in df:
        df['original_upb'] = df['original_upb'].astype('Int64')
    if 'msa_md' in df:
        # Keep as Int32 if possible; ignore if coercion would fail due to incompatible values
        df['msa_md'] = df['msa_md'].astype('Int32', errors='ignore')

    # ---- Identifiers / reference keys
    for col in ['loan_sequence_number', 'pre_relief_refi_loan_seq_number']:
        if col in df:
            df[col] = df[col].astype('string')

    # ---- Postal code normalized to 5-char string with left zero-padding
    if 'postal_code' in df:
        df['postal_code'] = (
            df['postal_code']
                .astype('Int64')   # tolerate numeric source
                .astype('string')
                .str.strip().str.upper()
                .str.zfill(5)
        )

    # ---- Categoricals with controlled category sets and NA handling
    if 'occupancy_status' in df:
        df['occupancy_status'] = (df['occupancy_status'].astype('string').str.strip().str.upper()
                                  .replace({'9': pd.NA})
                                  .astype(CategoricalDtype(categories=['P','S','I'], ordered=False)))
    if 'channel' in df:
        df['channel'] = (df['channel'].astype('string').str.strip().str.upper()
                         .replace({'9': pd.NA})
                         .astype(CategoricalDtype(categories=['R','B','C','T'], ordered=False)))
    if 'amortization_type' in df:
        df['amortization_type'] = (df['amortization_type'].astype('string').str.strip().str.upper()
                                   .astype(CategoricalDtype(categories=['FRM','ARM'], ordered=False)))
    if 'property_state' in df:
        df['property_state'] = df['property_state'].astype('string').str.strip().str.upper().astype('category')
    if 'property_type' in df:
        df['property_type'] = (df['property_type'].astype('string').str.strip().str.upper()
                               .replace({'99': pd.NA})
                               .astype(CategoricalDtype(categories=['SF','CO','PU','CP','MH'], ordered=False)))
    if 'loan_purpose' in df:
        df['loan_purpose'] = (df['loan_purpose'].astype('string').str.strip().str.upper()
                              .replace({'9': pd.NA})
                              .astype(CategoricalDtype(categories=['P','C','N','R'], ordered=False)))
    if 'special_eligibility_program' in df:
        df['special_eligibility_program'] = (df['special_eligibility_program'].astype('string').str.strip().str.upper()
                                             .replace({'9': pd.NA})
                                             .astype(CategoricalDtype(categories=['H','F','R'], ordered=False)))
    if 'property_valuation_method' in df:
        # Keep as small int category (codes), 9 means "NotAvailable"
        df['property_valuation_method'] = (
            df['property_valuation_method'].replace({9: pd.NA}).astype('Int8').astype('category')
        )

    return df

# ---------------------------------------------------------------------
# Business imputation (single-frame): fills missing values with rules.
# NOTE: This learns medians/modes on the provided df. To avoid leakage in
# ML experiments, prefer a fit/transform imputer trained on TRAIN only.
# ---------------------------------------------------------------------
def impute(df: pd.DataFrame, imput_cohort: bool = False) -> pd.DataFrame:
    """
    Apply business rules + simple statistical imputations on a single DataFrame:
      - Deterministic recodes (e.g., MI=0 if LTV<=80)
      - Median/mode imputations, optionally stratified by cohorts
        (vintage year, purpose, LTV bins)
      - Add missingness indicator flags for some columns
    Parameters
    ----------
    df : pd.DataFrame
        Input data (will be copied).
    imput_cohort : bool
        If True, use cohort-aware medians (e.g., by vintage year and/or purpose).
        If False, fall back to simpler medians by bin/column.
    """
    df = df.copy()

    # 0) Obvious drop (reference key rarely used downstream)
    df.drop(columns=['pre_relief_refi_loan_seq_number'], errors='ignore', inplace=True)

    vyear = df['vintage'].dt.year if 'vintage' in df.columns else None

    # 1) Categorical tweaks
    # channel: prefer 'Unknown' category instead of filling with mode
    if 'channel' in df and isinstance(df['channel'].dtype, CategoricalDtype):
        df['channel'] = df['channel'].cat.add_categories(['Unknown']).fillna('Unknown')

    # property_valuation_method:
    # - before 2017 -> code 99 ("NotApplicable")
    # - missing -> code 9 ("NotAvailable")
    if 'property_valuation_method' in df:
        pvm = pd.to_numeric(df['property_valuation_method'].astype('string'), errors='coerce')
        if vyear is not None:
            pvm = pvm.where(vyear >= 2017, 99)  # 99 = NotApplicable (pre-2017)
        pvm = pvm.fillna(9)  # 9 = NotAvailable
        df['property_valuation_method'] = pvm.astype('Int16').astype('category')

    # special_eligibility_program: keep original codes but also add a binary
    # convenience flag indicating presence of any special program.
    if 'special_eligibility_program' in df and isinstance(df['special_eligibility_program'].dtype, CategoricalDtype):
        df['special_eligibility_program'] = df['special_eligibility_program'].cat.add_categories(['Unknown']).fillna('Unknown')
        df['has_special_program'] = df['special_eligibility_program'].isin(['H','F','R']).astype('int8')

    # msa_md: group NA & non-MSA under 0 (as a numeric "no MSA" marker)
    if 'msa_md' in df:
        df['msa_md'] = df['msa_md'].fillna(0)

    # 2) Credit score: clipping + cohort median + global median; add missing flag
    if 'credit_score' in df:
        df['cs_missing'] = df['credit_score'].isna().astype('int8')
        cs = pd.to_numeric(df['credit_score'], errors='coerce').clip(lower=300, upper=850)
        if imput_cohort:
            if vyear is not None and 'loan_purpose' in df:
                med = df.groupby([vyear, 'loan_purpose'])['credit_score'].transform('median')
                cs = cs.fillna(med)
            elif 'loan_purpose' in df:
                med = df.groupby(['loan_purpose'])['credit_score'].transform('median')
                cs = cs.fillna(med)
        df['credit_score'] = cs.fillna(cs.median()).round().astype('Int16')

    # 3) MI%: if LTV<=80 and MI is NA -> 0; otherwise median by LTV bins (and cohorts if enabled)
    if 'mi_percent' in df:
        df['mi_missing'] = df['mi_percent'].isna().astype('int8')
        mi = pd.to_numeric(df['mi_percent'], errors='coerce').astype('Float32')
        if 'original_ltv' in df:
            ltv = pd.to_numeric(df['original_ltv'], errors='coerce').clip(lower=0)
            # Business rule: loans with LTV<=80 typically don't carry MI
            mi = mi.mask(ltv.le(80) & mi.isna(), 0.0)
            ltv_bins = pd.cut(ltv, [0, 80, 90, 95, 100, np.inf], include_lowest=True, right=True)
            if vyear is not None and imput_cohort:
                med = df.groupby([vyear, ltv_bins])['mi_percent'].transform('median')
                mi = mi.fillna(med)
            else:
                med = df.groupby(ltv_bins)['mi_percent'].transform('median')
                mi = mi.fillna(med)
        df['mi_percent'] = mi.fillna(0.0).astype('Float32')
        df['has_mi'] = (df['mi_percent'] > 0).astype('int8')

    # 4) DTI: median by (year x purpose) if enabled; else global median; add missing flag
    if 'original_dti' in df:
        df['dti_missing'] = df['original_dti'].isna().astype('int8')
        dti = pd.to_numeric(df['original_dti'], errors='coerce')
        if imput_cohort:
            if vyear is not None and 'loan_purpose' in df:
                med = df.groupby([vyear, 'loan_purpose'])['original_dti'].transform('median')
                dti = dti.fillna(med)
            elif 'loan_purpose' in df:
                med = df.groupby(['loan_purpose'])['original_dti'].transform('median')
                dti = dti.fillna(med)
        df['original_dti'] = dti.fillna(dti.median()).round().astype('Int16')

    # 5) CLTV: lower bound by LTV; then cohort median by year; fallback to global median; add flag
    if 'original_cltv' in df:
        df['cltv_missing'] = df['original_cltv'].isna().astype('int8')
        cltv = pd.to_numeric(df['original_cltv'], errors='coerce').astype('Float32')
        if 'original_ltv' in df:
            ltv = pd.to_numeric(df['original_ltv'], errors='coerce').astype('Float32')
            # If CLTV is missing but LTV exists, use LTV as a lower bound (CLTV >= LTV)
            cltv = cltv.fillna(ltv)
            cltv = np.where(ltv.notna(), np.maximum(cltv, ltv), cltv)
            cltv = pd.Series(cltv, index=df.index).astype('Float32')
        if imput_cohort and vyear is not None:
            med = df.groupby(vyear)['original_cltv'].transform('median')
            cltv = cltv.fillna(med)
        # global fallback
        df['original_cltv'] = pd.Series(cltv, index=df.index).fillna(float(pd.Series(cltv).median())).astype('Float32')

    # 6) Small ordinals: simple mode imputation (or ffill/bfill as a last resort); add flags
    for col in ['original_loan_term', 'number_of_borrowers']:
        if col in df:
            df[col + '_missing'] = df[col].isna().astype('int8')
            try:
                df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
            except Exception:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    return df

# ---------------------------------------------------------------------
# Utility: load quarter files and split into train/val/test by year
# ---------------------------------------------------------------------
READ_KW = dict(engine="pyarrow", dtype_backend="pyarrow")  # CLI may fallback if pyarrow is unavailable

def load_quarter_files(in_dir: Path, window_months: int = 24) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CSVs named like: default_labels_{T}m_YYYYQ*.csv from `in_dir`,
    then split by year using the convention:
        - train: 2020–2022
        - validation: 2023
        - test: 2024
    Returns (df_train, df_val, df_test). Missing buckets yield empty DataFrames.
    """
    pattern = re.compile(rf"default_labels_{window_months}m_(\d{{4}}Q[1-4])\.csv$", re.I)
    files = sorted(
        (p for p in Path(in_dir).glob(f"default_labels_{window_months}m_*.csv") if pattern.match(p.name)),
        key=lambda p: pattern.match(p.name).group(1)
    )

    buckets = {"train": [], "validation": [], "test": []}

    def year_to_split(y: int):
        if 2020 <= y <= 2022: return "train"
        if y == 2023:         return "validation"
        if y == 2024:         return "test"
        return None

    for p in files:
        qstr = pattern.match(p.name).group(1)
        q = pd.Period(qstr, freq="Q")
        split = year_to_split(q.year)
        if split is None:
            print(f"Ignored: {p.name}")
            continue
        df = pd.read_csv(p, **READ_KW)
        df["vintage"] = q  # add the quarter as a Period[Q] (useful downstream)
        buckets[split].append(df)

    df_train = pd.concat(buckets["train"], ignore_index=True) if buckets["train"] else pd.DataFrame()
    df_val   = pd.concat(buckets["validation"], ignore_index=True) if buckets["validation"] else pd.DataFrame()
    df_test  = pd.concat(buckets["test"], ignore_index=True) if buckets["test"] else pd.DataFrame()
    return df_train, df_val, df_test

def coerce_and_impute(df: pd.DataFrame, imput_cohort: bool = False) -> pd.DataFrame:
    """
    Convenience one-liner that first normalizes dtypes (coerce_base_types)
    and then applies business imputation (impute).
    """
    return impute(coerce_base_types(df), imput_cohort=imput_cohort)
