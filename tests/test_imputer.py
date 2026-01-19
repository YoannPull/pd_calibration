import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype, is_integer_dtype
from features.impute import DataImputer


def make_train_df() -> pd.DataFrame:
    """Mini dataset d'entrainement sans NA sur certaines colonnes,
    pour tester l'apparition de NA 'nouveaux' à l'inférence."""
    n = 5
    df = pd.DataFrame({
        'credit_score': [700, 720, 680, 690, 710],
        'first_payment_date': pd.to_datetime(['2018-01-01'] * n),
        'maturity_date': pd.to_datetime(['2048-01-01'] * n),
        'msa_md': [12345, 23456, 34567, 45678, 56789],
        'mi_percent': [np.nan, 12.0, 0.0, np.nan, 30.0],
        'number_of_units': [1, 1, 1, 2, 1],  # <- aucun NA au train
        'occupancy_status': ['P', 'P', 'I', 'P', 'S'],
        'original_cltv': [80, 85, 90, np.nan, 95],
        'original_dti': [35, 40, 28, 33, 45],
        'original_upb': [200000, 180000, 220000, 150000, 300000],
        'original_ltv': [78, 82, 88, 90, 97],
        'original_interest_rate': [3.5, 3.6, 3.7, 3.8, 4.0],
        'channel': pd.Series(['R', 'B', 'R', 'B', 'R'], dtype='category'),  # cat bien typée
        'ppm_flag': ['N'] * n,
        'amortization_type': ['FRM'] * n,
        'property_state': ['CA'] * n,
        'property_type': ['SF'] * n,
        'postal_code': ['90001'] * n,
        'loan_sequence_number': [f'L{i}' for i in range(n)],
        'loan_purpose': ['P', 'P', 'C', 'P', 'C'],
        'original_loan_term': [360, 360, 360, 360, 360],
        'number_of_borrowers': [1, 1, 2, 1, 1],  # mode = 1
        'seller_name': ['Seller'] * n,
        'servicer_name': ['Servicer'] * n,
        'super_conforming_flag': ['N'] * n,
        'pre_relief_refi_loan_seq_number': [np.nan] * n,
        'special_eligibility_program': pd.Series(['H', 'F', 'H', 'F', 'H'], dtype='category'),
        'relief_refinance_indicator': ['N'] * n,
        # PVM sera normalisé en int cat dans transform ; ici volontairement en str
        'property_valuation_method': ['2', '9', '2', '2', '9'],
        'interest_only_indicator': ['N'] * n,
        'mi_cancellation_indicator': ['N'] * n,
        'default_24m': [0, 0, 0, 0, 1],
        'vintage': pd.to_datetime(['2017-06-01', '2018-03-01', '2018-07-01', '2019-01-01', '2017-10-01']),
    })
    return df


def make_new_df_with_new_missing(df_train: pd.DataFrame) -> pd.DataFrame:
    """Copie du train avec des NA injectés sur des colonnes qui n'en avaient pas,
    + suppression d'une colonne pour tester l'alignement de schéma."""
    df_new = df_train.copy()

    # Injecter des NA "nouveaux"
    df_new.loc[[1, 3], 'number_of_units'] = np.nan               # numeric -> fallback médiane
    df_new.loc[2, 'channel'] = np.nan                             # catégoriel -> fallback mode/'Unknown'
    df_new.loc[0, 'special_eligibility_program'] = np.nan         # catégoriel -> idem
    df_new.loc[4, 'property_valuation_method'] = np.nan           # cat num -> fallback cohérent dtype
    df_new.loc[1, 'credit_score'] = np.nan                        # feature explicitement imputée

    # Supprimer une colonne vue au fit pour tester l'auto-ajout + imputation
    df_new = df_new.drop(columns=['number_of_borrowers'])

    return df_new


def test_imputer_fills_new_missing_and_preserves_types(tmp_path):
    # 1) Fit sur train sans NA pour certaines colonnes
    df_train = make_train_df()
    imputer = DataImputer(use_cohort=True, missing_flag=True)
    imputer.fit(df_train)

    # 2) Nouvelles données avec NA inattendus
    df_new = make_new_df_with_new_missing(df_train)
    df_imp = imputer.transform(df_new)

    # --- A. Les NA injectés sont bien imputés ---
    cols_to_check = [
        'number_of_units', 'channel', 'special_eligibility_program',
        'property_valuation_method', 'credit_score'
    ]
    assert not df_imp[cols_to_check].isna().any().any(), "Des NA subsistent après imputation"

    # --- B. La colonne absente revient bien (alignement de schéma) et est imputée ---
    assert 'number_of_borrowers' in df_imp.columns
    assert not df_imp['number_of_borrowers'].isna().any()

    # --- C. Typage catégoriel numérique préservé pour PVM ---
    pvm = df_imp['property_valuation_method']
    assert isinstance(pvm.dtype, CategoricalDtype), "PVM devrait être de type catégorie"
    assert is_integer_dtype(pvm.cat.categories.dtype), "Les catégories PVM devraient être numériques (int)"

    # --- D. Flags was_missing_* corrects ---
    assert 'was_missing_number_of_units' in df_imp.columns
    assert df_imp.loc[1, 'was_missing_number_of_units'] == 1
    assert df_imp.loc[3, 'was_missing_number_of_units'] == 1
    assert df_imp['was_missing_number_of_units'].dtype == 'int8'

    # --- E. Export Parquet possible (typage propre) ---
    pa = pytest.importorskip("pyarrow")  # skip si pas installé
    out_path = tmp_path / "sample.parquet"
    df_imp.to_parquet(out_path, index=False)  # ne doit pas lever
    assert out_path.exists()
