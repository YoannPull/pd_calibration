# -*- coding: utf-8 -*-
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, Tuple

# ---- Column name constants (order must match file spec) ----
COLS_ORIG = [
    "credit_score","first_payment_date","first_time_homebuyer_flag","maturity_date",
    "msa_md","mi_percent","number_of_units","occupancy_status","original_cltv","original_dti",
    "original_upb","original_ltv","original_interest_rate","channel","ppm_flag","amortization_type",
    "property_state","property_type","postal_code","loan_sequence_number","loan_purpose",
    "original_loan_term","number_of_borrowers","seller_name","servicer_name","super_conforming_flag",
    "pre_relief_refi_loan_seq_number","special_eligibility_program","relief_refinance_indicator",
    "property_valuation_method","interest_only_indicator","mi_cancellation_indicator"
]

COLS_PERF = [
    "loan_sequence_number","monthly_reporting_period","current_actual_upb",
    "current_loan_delinquency_status","loan_age","remaining_months_to_legal_maturity",
    "defect_settlement_date","modification_flag","zero_balance_code","zero_balance_effective_date",
    "current_interest_rate","current_non_interest_bearing_upb","ddlpi","mi_recoveries",
    "net_sale_proceeds","non_mi_recoveries","total_expenses","legal_costs",
    "maintenance_and_preservation_costs","taxes_and_insurance","miscellaneous_expenses",
    "actual_loss_calculation","cumulative_modification_cost","step_modification_flag",
    "payment_deferral","estimated_ltv","zero_balance_removal_upb","delinquent_accrued_interest",
    "delinquency_due_to_disaster","borrower_assistance_status_code","current_month_modification_cost",
    "interest_bearing_upb"
]

# ---- Memory-friendly dtypes (tweak if needed) ----
DTYPES_ORIG = {
    "credit_score":"float32","first_payment_date":"string","maturity_date":"string",
    "msa_md":"float32","mi_percent":"float32","number_of_units":"float32","occupancy_status":"string",
    "original_cltv":"float32","original_dti":"float32","original_upb":"float64","original_ltv":"float32",
    "original_interest_rate":"float32","channel":"string","ppm_flag":"string","amortization_type":"string",
    "property_state":"string","property_type":"string","postal_code":"string","loan_sequence_number":"string",
    "loan_purpose":"string","original_loan_term":"float32","number_of_borrowers":"float32",
    "seller_name":"string","servicer_name":"string","super_conforming_flag":"string",
    "pre_relief_refi_loan_seq_number":"string","special_eligibility_program":"string",
    "relief_refinance_indicator":"string","property_valuation_method":"float32",
    "interest_only_indicator":"string","mi_cancellation_indicator":"string"
}

DTYPES_PERF = {
    "loan_sequence_number":"string","monthly_reporting_period":"string",
    "current_actual_upb":"float64","current_loan_delinquency_status":"string","loan_age":"float32",
    "remaining_months_to_legal_maturity":"float32","defect_settlement_date":"string",
    "modification_flag":"string","zero_balance_code":"string","zero_balance_effective_date":"string",
    "current_interest_rate":"float32","current_non_interest_bearing_upb":"float64","ddlpi":"string",
    "mi_recoveries":"float64","net_sale_proceeds":"string","non_mi_recoveries":"float64",
    "total_expenses":"float64","legal_costs":"float64","maintenance_and_preservation_costs":"float64",
    "taxes_and_insurance":"float64","miscellaneous_expenses":"float64","actual_loss_calculation":"float64",
    "cumulative_modification_cost":"float64","step_modification_flag":"string","payment_deferral":"string",
    "estimated_ltv":"float32","zero_balance_removal_upb":"float64","delinquent_accrued_interest":"float64",
    "delinquency_due_to_disaster":"string","borrower_assistance_status_code":"string",
    "current_month_modification_cost":"float64","interest_bearing_upb":"float64"
}

# ---- Helpers ----
def _parse_yyyymm(series: pd.Series) -> pd.PeriodIndex:
    """Parse YYYYMM into monthly Periods quickly & robustly."""
    s = pd.to_datetime(series.astype("string"), format="%Y%m", errors="coerce")
    return s.dt.to_period("M")

def _month_diff(mpr: pd.Series, fpd: pd.Series) -> pd.Series:
    # Ensure period[M] dtype even if input are strings/datetimes by accident
    if not pd.api.types.is_period_dtype(mpr):
        mpr = pd.PeriodIndex(mpr.astype("string"), freq="M").to_series(index=mpr.index)
    if not pd.api.types.is_period_dtype(fpd):
        fpd = pd.PeriodIndex(fpd.astype("string"), freq="M").to_series(index=fpd.index)
    return (mpr.dt.year - fpd.dt.year) * 12 + (mpr.dt.month - fpd.dt.month)


def _normalize_zb_code(z: pd.Series) -> pd.Series:
    """Zero-balance codes as 2-digit strings ('02','03','09',...)."""
    z = z.astype("string").str.strip()
    # if numeric-like, left-pad to 2
    mask_num = z.str.fullmatch(r"\d+")
    z.loc[mask_num] = z.loc[mask_num].str.zfill(2)
    return z

def _make_default_row_flag(
    delinquency_status: pd.Series,
    zero_balance_code: pd.Series,
    delinquency_threshold: int,
    liquidation_codes: Iterable[str],
    include_ra: bool
) -> pd.Series:
    """
    Row-level default:
      - delinquency >= threshold (e.g., 3 -> 90+ DPD) OR RA (if include_ra)
      - OR zero_balance_code in liquidation_codes
    """
    s = delinquency_status.astype("string").str.strip().str.upper()
    is_ra = s.eq("RA") if include_ra else pd.Series(False, index=s.index)
    # numeric delinquency where possible
    s_num = pd.to_numeric(s.where(~is_ra), errors="coerce")
    is_90dpd = s_num.ge(delinquency_threshold)

    z = _normalize_zb_code(zero_balance_code)
    is_liq = z.isin(list(liquidation_codes))

    default_row = (is_90dpd | is_ra | is_liq).astype("int8")
    return default_row

# ---- Public API ----
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
    Build a loan-level dataset with a 'default_{T}m' column joined onto Origination.

    Args:
        path_orig: path to origination file (pipe-delimited, no header, 32 cols).
        path_perf: path to performance file (pipe-delimited, no header, 32 cols).
        window_months: label horizon in months (typical: 12/24/36).
        delinquency_threshold: integer delinquency bucket as default (3 => 90+ DPD).
        liquidation_codes: zero-balance codes considered default events.
        include_ra: treat 'RA' (REO acquisition) in delinquency status as default.
        require_full_window:
            - If True, keep only loans whose performance history covers >= T months after FPD
              (avoids right-censoring). This requires having rows up to FPD+T for the loan.

    Returns:
        df_orig_with_label: Origination columns + 'default_{T}m' (int8).
    """
    # --- Read files with explicit schema (fast & predictable) ---
    df_orig = pd.read_csv(
        path_orig, sep="|", header=None, names=COLS_ORIG, dtype=DTYPES_ORIG, engine="c"
    )
    df_perf = pd.read_csv(
        path_perf, sep="|", header=None, names=COLS_PERF, dtype=DTYPES_PERF, engine="c"
    )

    # --- Parse dates to monthly Periods ---
    fpd = _parse_yyyymm(df_orig["first_payment_date"])
    df_orig["first_payment_date"] = fpd

    mpr = _parse_yyyymm(df_perf["monthly_reporting_period"])
    df_perf["monthly_reporting_period"] = mpr

    # --- Join FPD onto performance for month arithmetic ---
    df_perf = df_perf.merge(
        df_orig[["loan_sequence_number", "first_payment_date"]],
        on="loan_sequence_number",
        how="left",
        copy=False,
        validate="m:1"
    )

    # --- Compute months since origination (vectorized) ---
    df_perf["months_since_orig"] = _month_diff(
        df_perf["monthly_reporting_period"], df_perf["first_payment_date"]
    ).astype("Int32")

    # --- Restrict to the first T months (inclusive) ---
    within = df_perf["months_since_orig"].le(window_months)
    dfw = df_perf.loc[within, ["loan_sequence_number",
                               "current_loan_delinquency_status",
                               "zero_balance_code"]].copy()

    # --- Row-level default flag (in-window) ---
    dfw["default_row"] = _make_default_row_flag(
        dfw["current_loan_delinquency_status"],
        dfw["zero_balance_code"],
        delinquency_threshold=delinquency_threshold,
        liquidation_codes=liquidation_codes,
        include_ra=include_ra
    )

    # --- Aggregate to loan-level default (any default within T months) ---
    loan_level = (
        dfw.groupby("loan_sequence_number", observed=True)["default_row"]
           .max()
           .rename(f"default_{window_months}m")
           .astype("Int8")
           .reset_index()
    )

    # --- Optional: enforce "full observation window" (censoring control) ---
    if require_full_window:
        # A loan has full window if it has at least one perf row at exactly T months since FPD
        has_T = (
            df_perf.loc[df_perf["months_since_orig"].eq(window_months), "loan_sequence_number"]
                  .dropna()
                  .drop_duplicates()
        )
        loan_level = loan_level.merge(
            has_T.to_frame("loan_sequence_number").assign(_ok=1),
            on="loan_sequence_number", how="inner"
        ).drop(columns="_ok")

    # --- Merge back to origination (loan-level, features-only) ---
    df_out = df_orig.merge(loan_level, on="loan_sequence_number", how="left")

    # No default observed in window => 0
    label_col = f"default_{window_months}m"
    df_out[label_col] = df_out[label_col].fillna(0).astype("Int8")

    return df_out
