# Data Disclaimer (Third-Party / External Data)

This repository contains **code** to process and analyze external datasets.  
It does **not** grant any rights to third-party data and does not redistribute restricted datasets unless explicitly stated.

Users of this repository are responsible for obtaining the data legally and complying with the relevant terms of use.

---

## 1) Freddie Mac Single-Family Loan-Level Dataset (Standard)

The main empirical pipeline (Empirical application #1) uses the **Freddie Mac Single-Family Loan-Level Dataset (Standard)**.

Official access point:
- https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

### Important notes
- The dataset is **owned and distributed by Freddie Mac** and may be subject to
  specific **terms of use, access conditions, and redistribution restrictions**.
- Users must:
  1) download the data through the official channel, and  
  2) comply with the dataset’s terms and any applicable legal/regulatory requirements.
- The authors provide **no warranty** regarding the data (availability, correctness, completeness, suitability).

### Expected local layout
This project expects the Freddie Mac files to be stored locally under:
- `data/raw/mortgage_data/`

See `data/README.md` for the exact directory layout used by the pipeline.

---

## 2) Rating history data for the LDP / S&P grade application (`ldp_application/data/`)

The LDP module (Empirical application #2) relies on **credit rating history inputs** stored under:
- `ldp_application/data/`

### Regulatory background (SEC Rule 17g-7)
In the United States, **Nationally Recognized Statistical Rating Organizations (NRSROs)** are required
to disclose their **credit rating histories** in an interactive data file using **XBRL** under **SEC Rule 17g-7(b)**.
The SEC also provides a publication guide and taxonomy for these disclosures.

### Source used in this project: ratingshistory.info
In this project, rating history data is sourced from:
- https://ratingshistory.info

This website provides:
- rating histories from multiple agencies,
- originally disclosed as XBRL files on rating agency websites,
- converted into **CSV files** for easier use in research workflows (Excel/Access),
- with filenames that typically encode the **as-of date**, the **agency name**, and the **asset category**.

The website references the SEC publication guide for abbreviations and technical conventions.

### Expected local layout (updated workflow)

**Raw download (CSV):**
- `ldp_application/data/raw/20220601_SP_Ratings_Services_Corporate.csv` *(example name; depends on the download date)*

**Processed input used by the grade application (CSV-only):**
The Makefile first builds a monthly snapshot (1 row per obligor × month) using:
- `ldp_application/process_sp_base.py`

Default output:
- `ldp_application/data/processed/sp_corporate_monthly.csv`

This processed file follows the same *schema* as the historical `data_rating_corporate.xlsx`
but is generated automatically and does not include enrichment fields (they are kept empty / NA).

If you replace these files or use another dataset, ensure you comply with the relevant terms.

---

## No affiliation
This repository is an independent research codebase and is **not affiliated with**
Freddie Mac, ratingshistory.info, the SEC, S&P, or any other data provider mentioned above.

---


