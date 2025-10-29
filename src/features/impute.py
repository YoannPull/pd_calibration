# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
from typing import Tuple, Dict, Optional, List
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
    Normalize dtypes (no imputation here):
      - Periods for date-like columns
      - Boolean flags for Y/N/blank indicators
      - Nullable integers for sentinel-coded numerics
      - Categoricals for enums
      - String for IDs
    Also replaces known sentinel values with <NA>.
    """
    df = df.copy()

    # ---- Dates / periods
    if 'first_payment_date' in df:
        df['first_payment_date'] = to_periodM(df['first_payment_date'])
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

    # ---- Target (optional) -> boolean
    if 'default_24m' in df:
        if pd.api.types.is_bool_dtype(df['default_24m']):
            df['default_24m'] = df['default_24m'].astype('boolean')
        elif pd.api.types.is_integer_dtype(df['default_24m']):
            df['default_24m'] = df['default_24m'].map({1: True, 0: False}).astype('boolean')

    # ---- Sentinel replacement -> <NA>, then compact nullable ints
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
        df['msa_md'] = df['msa_md'].astype('Int32', errors='ignore')

    # ---- Identifiers / reference keys
    for col in ['loan_sequence_number', 'pre_relief_refi_loan_seq_number']:
        if col in df:
            df[col] = df[col].astype('string')

    # ---- Postal code -> '00000' string (left zero-padded)
    if 'postal_code' in df:
        pc = pd.to_numeric(df['postal_code'], errors='coerce').astype('Int64')
        df['postal_code'] = pc.astype('string').str.strip().str.upper().str.zfill(5)

    # ---- Categoricals with controlled sets
    if 'occupancy_status' in df:
        df['occupancy_status'] = (
            df['occupancy_status'].astype('string').str.strip().str.upper()
              .replace({'9': pd.NA})
              .astype(CategoricalDtype(categories=['P','S','I'], ordered=False))
        )
    if 'channel' in df:
        df['channel'] = (
            df['channel'].astype('string').str.strip().str.upper()
              .replace({'9': pd.NA})
              .astype(CategoricalDtype(categories=['R','B','C','T'], ordered=False))
        )
    if 'amortization_type' in df:
        df['amortization_type'] = (
            df['amortization_type'].astype('string').str.strip().str.upper()
              .astype(CategoricalDtype(categories=['FRM','ARM'], ordered=False))
        )
    if 'property_state' in df:
        df['property_state'] = df['property_state'].astype('string').str.strip().str.upper().astype('category')
    if 'property_type' in df:
        df['property_type'] = (
            df['property_type'].astype('string').str.strip().str.upper()
              .replace({'99': pd.NA})
              .astype(CategoricalDtype(categories=['SF','CO','PU','CP','MH'], ordered=False))
        )
    if 'loan_purpose' in df:
        df['loan_purpose'] = (
            df['loan_purpose'].astype('string').str.strip().str.upper()
              .replace({'9': pd.NA})
              .astype(CategoricalDtype(categories=['P','C','N','R'], ordered=False))
        )
    if 'special_eligibility_program' in df:
        df['special_eligibility_program'] = (
            df['special_eligibility_program'].astype('string').str.strip().str.upper()
              .replace({'9': pd.NA})
              .astype(CategoricalDtype(categories=['H','F','R'], ordered=False))
        )
    if 'property_valuation_method' in df:
        # Conserver la valeur codée telle quelle (1/2/3/4/9), SANS convertir 9->NA ici.
        df['property_valuation_method'] = (
            pd.to_numeric(df['property_valuation_method'], errors='coerce')
              .astype('Int8')
              .astype('category')
        )

    return df

# ---------------------------------------------------------------------
# Business imputation (single-frame): fills missing values with rules.
# ---------------------------------------------------------------------
def impute(df: pd.DataFrame, imput_cohort: bool = False) -> pd.DataFrame:
    """
    Apply business rules + simple statistical imputations on a single DataFrame:
      - Deterministic recodes (e.g., MI=0 if LTV<=80)
      - Median/mode imputations, optionally stratified by cohorts
        (vintage year, purpose, LTV bins)
      - Add missingness indicator flags for some columns
    Note: To avoid leakage, prefer a fit/transform imputer trained on TRAIN only.
    """
    df = df.copy()

    # 0) Drop rarely used reference key
    df.drop(columns=['pre_relief_refi_loan_seq_number'], errors='ignore', inplace=True)

    vyear = df['vintage'].dt.year if 'vintage' in df.columns else None

    # 1) Categorical tweaks
    if 'channel' in df and isinstance(df['channel'].dtype, CategoricalDtype):
        if 'Unknown' not in df['channel'].cat.categories:
            df['channel'] = df['channel'].cat.add_categories(['Unknown'])
        df['channel'] = df['channel'].fillna('Unknown')

    # property_valuation_method:
    # - before 2017 -> code 99 ("NotApplicable")
    # - missing -> code 9 ("NotAvailable")
    if 'property_valuation_method' in df:
        pvm = pd.to_numeric(df['property_valuation_method'].astype('string'), errors='coerce')
        if vyear is not None:
            pvm = pvm.where(vyear >= 2017, 99)  # 99 = NotApplicable (pre-2017)
        pvm = pvm.fillna(9)  # 9 = NotAvailable
        df['property_valuation_method'] = pvm.astype('Int16').astype('category')

    # special_eligibility_program: preserve codes + add binary convenience flag
    if 'special_eligibility_program' in df and isinstance(df['special_eligibility_program'].dtype, CategoricalDtype):
        if 'Unknown' not in df['special_eligibility_program'].cat.categories:
            df['special_eligibility_program'] = df['special_eligibility_program'].cat.add_categories(['Unknown'])
        df['special_eligibility_program'] = df['special_eligibility_program'].fillna('Unknown')
        df['has_special_program'] = df['special_eligibility_program'].isin(['H','F','R']).astype('int8')

    # msa_md: group NA & non-MSA under 0
    if 'msa_md' in df:
        df['msa_md'] = df['msa_md'].fillna(0)

    # 2) Credit score
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

    # 3) MI%
    if 'mi_percent' in df:
        df['mi_missing'] = df['mi_percent'].isna().astype('int8')
        mi = pd.to_numeric(df['mi_percent'], errors='coerce').astype('Float32')
        if 'original_ltv' in df:
            ltv = pd.to_numeric(df['original_ltv'], errors='coerce').clip(lower=0)
            mi = mi.mask(ltv.le(80) & mi.isna(), 0.0)  # LTV<=80 -> pas de MI
            ltv_bins = pd.cut(ltv, [0, 80, 90, 95, 100, np.inf], include_lowest=True, right=True)
            if vyear is not None and imput_cohort:
                med = df.groupby([vyear, ltv_bins])['mi_percent'].transform('median')
                mi = mi.fillna(med)
            else:
                med = df.groupby(ltv_bins)['mi_percent'].transform('median')
                mi = mi.fillna(med)
        df['mi_percent'] = mi.fillna(0.0).astype('Float32')
        df['has_mi'] = (df['mi_percent'] > 0).astype('int8')

    # 4) DTI
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
        # Spec Freddie: 0 < DTI <= 65 (999 = NA). Clip par prudence.
        df['original_dti'] = dti.fillna(dti.median()).clip(lower=0, upper=65).round().astype('Int16')

    # 5) CLTV
    if 'original_cltv' in df:
        df['cltv_missing'] = df['original_cltv'].isna().astype('int8')
        cltv = pd.to_numeric(df['original_cltv'], errors='coerce').astype('Float32')
        if 'original_ltv' in df:
            ltv = pd.to_numeric(df['original_ltv'], errors='coerce').astype('Float32')
            cltv = cltv.fillna(ltv)
            cltv = np.where(ltv.notna(), np.maximum(cltv, ltv), cltv)
            cltv = pd.Series(cltv, index=df.index).astype('Float32')
        if imput_cohort and vyear is not None:
            med = df.groupby(vyear)['original_cltv'].transform('median')
            cltv = cltv.fillna(med)
        df['original_cltv'] = pd.Series(cltv, index=df.index)\
                                .fillna(float(pd.Series(cltv).median()))\
                                .clip(lower=0, upper=300).astype('Float32')

    # 6) Small ordinals
    for col in ['original_loan_term', 'number_of_borrowers']:
        if col in df:
            df[col + '_missing'] = df[col].isna().astype('int8')
            try:
                df[col] = df[col].fillna(df[col].mode(dropna=True).iloc[0])
            except Exception:
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    return df

# ---------------------------------------------------------------------
# Utility: load quarter files and split into train/val/test
# ---------------------------------------------------------------------
READ_KW = dict(engine="pyarrow", dtype_backend="pyarrow")  # CLI may fallback if pyarrow is unavailable
_QPAT = re.compile(r"^\d{4}Q[1-4]$")

def _parse_quarter(qstr: str) -> pd.Period:
    qstr = str(qstr).strip().upper()
    if not _QPAT.fullmatch(qstr):
        raise ValueError(f"Quarter must be 'YYYYQ[1-4]'. Got: {qstr}")
    return pd.Period(qstr, freq="Q")

def _parse_range_expr(expr: Optional[str]) -> Tuple[Optional[pd.Period], Optional[pd.Period]]:
    """
    Parse 'YYYYQm:YYYYQn', 'YYYYQm:' (open-ended), ':YYYYQn', or single 'YYYYQm'.
    Returns (start, end) as Periods or None.
    """
    if not expr:
        return (None, None)
    expr = expr.strip()
    if ":" not in expr:
        q = _parse_quarter(expr)
        return (q, q)
    left, right = expr.split(":", 1)
    start = _parse_quarter(left) if left.strip() else None
    end   = _parse_quarter(right) if right.strip() else None
    return (start, end)

def _in_range(q: pd.Period, r: Tuple[Optional[pd.Period], Optional[pd.Period]]) -> bool:
    lo, hi = r
    return (lo is None or q >= lo) and (hi is None or q <= hi)

def _parse_years_expr(expr: Optional[str]) -> List[int]:
    """
    Accepts '2020', '2020-2022', '2018,2020,2022-2023'
    """
    years: List[int] = []
    if not expr:
        return years
    for token in str(expr).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            a, b = int(a), int(b)
            years.extend(range(min(a, b), max(a, b) + 1))
        else:
            years.append(int(token))
    return sorted(set(years))

def load_quarter_files(
    in_dir: Path,
    window_months: int = 24,
    *,
    # Option 1: explicit quarter ranges
    train_range: Optional[str] = None,
    val_range: Optional[str]   = None,
    test_range: Optional[str]  = None,
    # Option 2: years lists/ranges (fallback if no quarter ranges)
    train_years: Optional[str] = None,
    val_years: Optional[str]   = None,
    test_years: Optional[str]  = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load files named default_labels_{T}m_YYYYQ*.csv from `in_dir`
    and split by user-defined temporal rules.

    Precedence:
      1) If any of *_range are provided, split by quarter ranges, e.g.:
         train_range='2018Q1:2021Q4', val_range='2022Q1:2022Q4', test_range='2023Q1:'
      2) Else if *_years provided, split by years, e.g.:
         train_years='2018-2020', val_years='2021', test_years='2022-2023'
      3) Else default:
         train: 2020–2022 ; validation: 2023 ; test: 2024
    """
    pattern = re.compile(rf"default_labels_{window_months}m_(\d{{4}}Q[1-4])\.csv$", re.I)

    files = sorted(
        (p for p in Path(in_dir).glob(f"default_labels_{window_months}m_*.csv") if pattern.match(p.name)),
        key=lambda p: pattern.match(p.name).group(1)
    )
    if not files:
        raise RuntimeError(f"Aucun fichier trouvé dans {in_dir} avec le motif default_labels_{window_months}m_YYYYQ*.csv")

    # Build split decider
    use_ranges = any([train_range, val_range, test_range])
    if use_ranges:
        ranges = {
            "train": _parse_range_expr(train_range),
            "validation": _parse_range_expr(val_range),
            "test": _parse_range_expr(test_range),
        }
        def decide(q: pd.Period) -> Optional[str]:
            for bucket in ("train", "validation", "test"):
                r = ranges[bucket]
                if r != (None, None) and _in_range(q, r):
                    return bucket
            return None
    else:
        # Years-based (or default if none given)
        ty = set(_parse_years_expr(train_years)) or set(range(2020, 2023))
        vy = set(_parse_years_expr(val_years))   or {2023}
        sy = set(_parse_years_expr(test_years))  or {2024}
        def decide(q: pd.Period) -> Optional[str]:
            y = q.year
            if y in ty: return "train"
            if y in vy: return "validation"
            if y in sy: return "test"
            return None

    buckets = {"train": [], "validation": [], "test": []}

    for p in files:
        m = pattern.match(p.name)
        if not m:
            continue
        qstr = m.group(1)
        q = pd.Period(qstr, freq="Q")
        split = decide(q)
        if split is None:
            print(f"Ignored (no split rule matches): {p.name}")
            continue
        # Fallback if pyarrow isn't installed
        read_kw = READ_KW.copy()
        try:
            import pyarrow  # noqa: F401
        except Exception:
            read_kw.pop("engine", None)
            read_kw.pop("dtype_backend", None)
        df = pd.read_csv(p, **read_kw)
        df["vintage"] = q  # Period[Q]
        buckets[split].append(df)

    df_train = pd.concat(buckets["train"], ignore_index=True) if buckets["train"] else pd.DataFrame()
    df_val   = pd.concat(buckets["validation"], ignore_index=True) if buckets["validation"] else pd.DataFrame()
    df_test  = pd.concat(buckets["test"], ignore_index=True) if buckets["test"] else pd.DataFrame()
    return df_train, df_val, df_test

def coerce_and_impute(df: pd.DataFrame, imput_cohort: bool = False) -> pd.DataFrame:
    """
    One-liner: normalize dtypes (coerce_base_types) then apply business imputation (impute).
    """
    return impute(coerce_base_types(df), imput_cohort=imput_cohort)
