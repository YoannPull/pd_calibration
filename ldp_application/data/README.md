# `ldp_application/data/`

This folder contains the input data used by the LDP / S&P grade application.

## Source

Rating history data is sourced from:

```text
https://ratingshistory.info
````

The site provides credit rating history files from multiple agencies as **CSV exports**
(converted from the original XBRL disclosures).

## Raw input

We use the **S&P Corporate** history file downloaded from the source above and stored in:

* `ldp_application/data/raw/20220601_SP_Ratings_Services_Corporate.csv` *(example name; depends on the download date)*

The raw file contains instrument- and issuer-related rating actions. To avoid mixing heterogeneous
rating definitions, our pipeline restricts the sample to:

* `rating_type == "Issuer Credit Rating"`
* `rating_sub_type == "Local Currency LT"`
* `rating_agency_name == "Standard & Poor's Ratings Services"`

## Processed input (used by the grade application)

The grade application expects a **monthly snapshot** (one row per `obligor_name × year_month`,
keeping the most recent rating action within each month), produced by:

* `ldp_application/process_sp_base.py`

Default output (CSV only):

* `ldp_application/data/processed/sp_corporate_monthly.csv`

The processed dataset follows the same **schema** as the historical `data_rating_corporate.xlsx`
file (but is generated automatically and does not include enrichment fields):

Columns:

* `rating_agency_name`
* `rating`
* `rating_action_date`
* `legal_entity_identifier` *(empty / NA)*
* `obligor_name`
* `year_month`
* `year` *(July–June convention: months 1–6 belong to year t, months 7–12 to year t+1)*
* `pays` *(empty / NA)*
* `nace` *(empty / NA)*

## Makefile defaults

By default, the Makefile builds the snapshot and then runs the grade tables/plots using:

* `ldp_application/data/processed/sp_corporate_monthly.csv`

## Outputs

Generated tables and plots are written to:

* `ldp_application/outputs/sp_grade_is_oos/`

In particular:

* `sp_grade_table_YYYY.csv` *(one per year)*
* `sp_grade_tables_*.csv` *(combined table)*
* `plots_timeseries/` *(figures from the combined table)*
