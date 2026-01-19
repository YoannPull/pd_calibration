# src/features/labels.py
# -*- coding: utf-8 -*-
"""
Build loan-level default labels from Origination + Performance files.

This module is designed for datasets like Freddie/Fannie loan tapes where:
- ORIG ("origination") contains static loan attributes (one row per loan),
- PERF ("performance") contains monthly status snapshots (many rows per loan).

We create a binary default label at the *loan level* over a fixed horizon:
    default_{window_months}m = 1
if the loan experiences, within the first `window_months` months since first payment date:
- delinquency >= `delinquency_threshold` (e.g., 3 means 90+ DPD), OR
- "RA" status (optional), OR
- a liquidation / termination code (e.g., 02/03/09).

Output
------
A DataFrame equal to df_orig augmented with the label column `default_{window_months}m`.

Implementation notes
--------------------
- Input files are read with explicit column names and dtypes for speed + memory control.
- Dates are stored as monthly Period ("YYYYMM" -> Period[M]) to ease month arithmetic.
- We compute months_since_orig = months(monthly_reporting_period - first_payment_date).
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# RAW SCHEMA (column names as provided by the pipe-delimited files)
# -----------------------------------------------------------------------------

COLS_ORIG = [
    "credit_score",
    "first_payment_date",
    "first_time_homebuyer_flag",
    "maturity_date",
    "msa_md",
    "mi_percent",
    "number_of_units",
    "occupancy_status",
    "original_cltv",
    "original_dti",
    "original_upb",
    "original_ltv",
    "original_interest_rate",
    "channel",
    "ppm_flag",
    "amortization_type",
    "property_state",
    "property_type",
    "postal_code",
    "loan_sequence_number",
    "loan_purpose",
    "original_loan_term",
    "number_of_borrowers",
    "seller_name",
    "servicer_name",
    "super_conforming_flag",
    "pre_relief_refi_loan_seq_number",
    "special_eligibility_program",
    "relief_refinance_indicator",
    "property_valuation_method",
    "interest_only_indicator",
    "mi_cancellation_indicator",
]

COLS_PERF = [
    "loan_sequence_number",
    "monthly_reporting_period",
    "current_actual_upb",
    "current_loan_delinquency_status",
    "loan_age",
    "remaining_months_to_legal_maturity",
    "defect_settlement_date",
    "modification_flag",
    "zero_balance_code",
    "zero_balance_effective_date",
    "current_interest_rate",
    "current_non_interest_bearing_upb",
    "ddlpi",
    "mi_recoveries",
    "net_sale_proceeds",
    "non_mi_recoveries",
    "total_expenses",
    "legal_costs",
    "maintenance_and_preservation_costs",
    "taxes_and_insurance",
    "miscellaneous_expenses",
    "actual_loss_calculation",
    "cumulative_modification_cost",
    "step_modification_flag",
    "payment_deferral",
    "estimated_ltv",
    "zero_balance_removal_upb",
    "delinquent_accrued_interest",
    "delinquency_due_to_disaster",
    "borrower_assistance_status_code",
    "current_month_modification_cost",
    "interest_bearing_upb",
]

# -----------------------------------------------------------------------------
# READ DTYPES (memory-friendly; keep IDs as strings)
# -----------------------------------------------------------------------------

DTYPES_ORIG = {
    "credit_score": "float32",
    "first_payment_date": "string",
    "maturity_date": "string",
    "msa_md": "float32",
    "mi_percent": "float32",
    "number_of_units": "float32",
    "occupancy_status": "string",
    "original_cltv": "float32",
    "original_dti": "float32",
    "original_upb": "float64",
    "original_ltv": "float32",
    "original_interest_rate": "float32",
    "channel": "string",
    "ppm_flag": "string",
    "amortization_type": "string",
    "property_state": "string",
    "property_type": "string",
    "postal_code": "string",
    "loan_sequence_number": "string",
    "loan_purpose": "string",
    "original_loan_term": "float32",
    "number_of_borrowers": "float32",
    "seller_name": "string",
    "servicer_name": "string",
    "super_conforming_flag": "string",
    "pre_relief_refi_loan_seq_number": "string",
    "special_eligibility_program": "string",
    "relief_refinance_indicator": "string",
    "property_valuation_method": "float32",
    "interest_only_indicator": "string",
    "mi_cancellation_indicator": "string",
}

DTYPES_PERF = {
    "loan_sequence_number": "string",
    "monthly_reporting_period": "string",
    "current_actual_upb": "float64",
    "current_loan_delinquency_status": "string",
    "loan_age": "float32",
    "remaining_months_to_legal_maturity": "float32",
    "defect_settlement_date": "string",
    "modification_flag": "string",
    "zero_balance_code": "string",
    "zero_balance_effective_date": "string",
    "current_interest_rate": "float32",
    "current_non_interest_bearing_upb": "float64",
    "ddlpi": "string",
    "mi_recoveries": "float64",
    "net_sale_proceeds": "string",
    "non_mi_recoveries": "float64",
    "total_expenses": "float64",
    "legal_costs": "float64",
    "maintenance_and_preservation_costs": "float64",
    "taxes_and_insurance": "float64",
    "miscellaneous_expenses": "float64",
    "actual_loss_calculation": "float64",
    "cumulative_modification_cost": "float64",
    "step_modification_flag": "string",
    "payment_deferral": "string",
    "estimated_ltv": "float32",
    "zero_balance_removal_upb": "float64",
    "delinquent_accrued_interest": "float64",
    "delinquency_due_to_disaster": "string",
    "borrower_assistance_status_code": "string",
    "current_month_modification_cost": "float64",
    "interest_bearing_upb": "float64",
}

# -----------------------------------------------------------------------------
# DATE / STATUS HELPERS
# -----------------------------------------------------------------------------


def _parse_yyyymm(series: pd.Series) -> pd.PeriodIndex:
    """
    Parse a YYYYMM string column into a monthly PeriodIndex (freq="M").

    Invalid values are coerced to NaT -> will become <NA> periods.
    """
    s = pd.to_datetime(series.astype("string"), format="%Y%m", errors="coerce")
    return s.dt.to_period("M")


def _month_diff(mpr: pd.Series, fpd: pd.Series) -> pd.Series:
    """
    Compute (mpr - fpd) in months, with both inputs as monthly periods.

    Parameters
    ----------
    mpr : monthly_reporting_period (Period[M] or coercible to it)
    fpd : first_payment_date (Period[M] or coercible to it)

    Returns
    -------
    Integer-like Series of month differences.
    """
    if not pd.api.types.is_period_dtype(mpr):
        mpr = pd.PeriodIndex(mpr.astype("string"), freq="M").to_series(index=mpr.index)
    if not pd.api.types.is_period_dtype(fpd):
        fpd = pd.PeriodIndex(fpd.astype("string"), freq="M").to_series(index=fpd.index)

    return (mpr.dt.year - fpd.dt.year) * 12 + (mpr.dt.month - fpd.dt.month)


def _normalize_zb_code(z: pd.Series) -> pd.Series:
    """
    Normalize zero-balance codes to two-digit strings when numeric.

    Example: "2" -> "02"
    """
    z = z.astype("string").str.strip()
    mask_num = z.str.fullmatch(r"\d+")
    z.loc[mask_num] = z.loc[mask_num].str.zfill(2)
    return z


def _make_default_row_flag(
    delinquency_status: pd.Series,
    zero_balance_code: pd.Series,
    delinquency_threshold: int,
    liquidation_codes: Iterable[str],
    include_ra: bool,
) -> pd.Series:
    """
    Build a *row-level* default indicator from monthly performance fields.

    A row is flagged as "default-like" if:
    - delinquency_status is numeric and >= delinquency_threshold, OR
    - delinquency_status == "RA" (if include_ra=True), OR
    - zero_balance_code is in liquidation_codes.

    Returns
    -------
    int8 Series with values {0,1}.
    """
    s = delinquency_status.astype("string").str.strip().str.upper()

    # Some tapes use "RA" as a special status (e.g., repurchase / rep & warranty).
    is_ra = s.eq("RA") if include_ra else pd.Series(False, index=s.index)

    # Convert non-RA statuses to numeric delinquency buckets (coerce invalid to NaN).
    s_num = pd.to_numeric(s.where(~is_ra), errors="coerce")
    is_90dpd = s_num.ge(delinquency_threshold)

    z = _normalize_zb_code(zero_balance_code)
    is_liq = z.isin(list(liquidation_codes))

    return (is_90dpd | is_ra | is_liq).astype("int8")


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------


def build_default_labels(
    path_orig: str,
    path_perf: str,
    window_months: int = 24,
    delinquency_threshold: int = 3,
    liquidation_codes: Tuple[str, ...] = ("02", "03", "09"),
    include_ra: bool = True,
    require_full_window: bool = False,
) -> pd.DataFrame:
    """
    Build loan-level default labels over a fixed horizon.

    Parameters
    ----------
    path_orig : str
        Path to the origination file (pipe-delimited, no header).
    path_perf : str
        Path to the performance file (pipe-delimited, no header).
    window_months : int
        Horizon in months after first payment date (inclusive).
    delinquency_threshold : int
        Numeric delinquency bucket considered as default (3 => 90+ DPD).
    liquidation_codes : tuple[str, ...]
        Zero-balance codes treated as liquidation/termination => default.
    include_ra : bool
        If True, treat delinquency status "RA" as default-like.
    require_full_window : bool
        If True, keep only loans that have an observation exactly at T=window_months
        (i.e., observed through the full horizon). This avoids labeling loans with
        short histories as non-default just because data stops early.

    Returns
    -------
    pd.DataFrame
        df_orig enriched with a label column: default_{window_months}m (Int8 0/1).
    """
    # 1) Load raw files with explicit schemas.
    df_orig = pd.read_csv(
        path_orig, sep="|", header=None, names=COLS_ORIG, dtype=DTYPES_ORIG, engine="c"
    )
    df_perf = pd.read_csv(
        path_perf, sep="|", header=None, names=COLS_PERF, dtype=DTYPES_PERF, engine="c"
    )

    # 2) Parse YYYYMM strings into monthly periods.
    df_orig["first_payment_date"] = _parse_yyyymm(df_orig["first_payment_date"])
    df_perf["monthly_reporting_period"] = _parse_yyyymm(df_perf["monthly_reporting_period"])

    # 3) Attach first_payment_date to each performance row (m:1).
    df_perf = df_perf.merge(
        df_orig[["loan_sequence_number", "first_payment_date"]],
        on="loan_sequence_number",
        how="left",
        copy=False,
        validate="m:1",
    )

    # 4) Compute months since origination (first payment date).
    df_perf["months_since_orig"] = _month_diff(
        df_perf["monthly_reporting_period"], df_perf["first_payment_date"]
    ).astype("Int32")

    # 5) Keep only observations within the labeling window.
    within = df_perf["months_since_orig"].le(window_months)
    dfw = df_perf.loc[
        within,
        ["loan_sequence_number", "current_loan_delinquency_status", "zero_balance_code"],
    ].copy()

    # 6) Row-level default flag, then aggregate to loan-level max.
    dfw["default_row"] = _make_default_row_flag(
        dfw["current_loan_delinquency_status"],
        dfw["zero_balance_code"],
        delinquency_threshold=delinquency_threshold,
        liquidation_codes=liquidation_codes,
        include_ra=include_ra,
    )

    loan_level = (
        dfw.groupby("loan_sequence_number", observed=True)["default_row"]
        .max()
        .rename(f"default_{window_months}m")
        .astype("Int8")
        .reset_index()
    )

    # 7) Optional censoring control: require observed data at exactly T=window_months.
    if require_full_window:
        has_T = (
            df_perf.loc[df_perf["months_since_orig"].eq(window_months), "loan_sequence_number"]
            .dropna()
            .drop_duplicates()
        )
        loan_level = (
            loan_level.merge(
                has_T.to_frame("loan_sequence_number").assign(_ok=1),
                on="loan_sequence_number",
                how="inner",
            )
            .drop(columns="_ok")
        )

    # 8) Merge label back into origination table; missing labels -> 0 (no default observed).
    df_out = df_orig.merge(loan_level, on="loan_sequence_number", how="left")
    label_col = f"default_{window_months}m"
    df_out[label_col] = df_out[label_col].fillna(0).astype("Int8")

    return df_out
