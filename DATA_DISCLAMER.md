# Data Disclaimer (Third-Party / External Data)

This repository contains **code** to process and analyse external datasets. It does **not** grant any rights to third-party data and does **not** redistribute restricted datasets unless explicitly stated. Users of this repository are responsible for obtaining the data legally and complying with the relevant terms of use.

---

## 1) Freddie Mac Single-Family Loan-Level Dataset (Standard)

The main empirical pipeline (Empirical application #1) uses the **Freddie Mac Single-Family Loan-Level Dataset (Standard)**.

Official access point:
- https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

### Important notes
- The Freddie Mac dataset is **not included** in this repository and will not be provided by the authors.
- The dataset is **owned and distributed by Freddie Mac** and may be subject to specific **terms of use, access conditions, and redistribution restrictions**.
- Users must:
  1) download the data through the official channel, and
  2) comply with the dataset’s terms and any applicable legal or regulatory requirements.
- The authors provide **no warranty** regarding the data (availability, correctness, completeness, suitability).

### Expected local layout
This project expects the Freddie Mac files to be stored locally under:
- `data/raw/mortgage_data/`

See `data/README.md` for the exact directory layout used by the pipeline.

### Using a shorter time span
The quarters used by the pipeline are configured in `config.yaml`. If you download fewer quarters, update:
- `data.quarters`, and
- the explicit split definitions under `splits.explicit` (design, validation, and out-of-sample quarters).

This allows replication with fewer raw files while keeping the rest of the workflow unchanged.

---

## 2) Rating history data for the LDP / S&P grade application (`ldp_application/data/`)

The LDP module (Empirical application #2) relies on **credit rating history disclosures** published by NRSROs under **SEC Rule 17g-7(b)** in XBRL format.

Regulatory background and technical documentation:
- https://www.sec.gov/about/divisions-offices/office-credit-ratings/disclosure-of-credit-rating-histories
- https://www.sec.gov/data-research/structured-data/rating-history-files-publication-guide

### Important notes (availability and redistribution)
- This repository does **not** redistribute any rating history files.
- In our original experiments, we used a snapshot downloaded in 2022. That exact third-party snapshot may no longer be available from external mirrors. Replication therefore requires re-downloading the underlying disclosures from official sources (or equivalent public disclosures).

### Suggested tool (external, optional)
To facilitate downloading and converting rating history disclosures into sorted CSV files, users may rely on the open-source project `maxonlinux/ratings-history` (external tool, not a dependency of this repository):
- https://github.com/maxonlinux/ratings-history

Note: this external project is released under the **AGPL-3.0** licence. We do not vendor or link it as a library in this repository; we only reference it as an optional standalone tool.

### Expected local layout (updated workflow)

**Raw input (CSV):**
- `ldp_application/data/raw/20220601_SP_Ratings_Services_Corporate.csv` *(example name; depends on the download date / snapshot)*

**Processed input used by the grade application (CSV-only):**
The Makefile first builds a monthly snapshot (1 row per obligor × month) using:
- `ldp_application/process_sp_base.py`

Default output:
- `ldp_application/data/processed/sp_corporate_monthly.csv`

This processed file follows the same schema as the historical `data_rating_corporate.xlsx` but is generated automatically and does not include enrichment fields (they are kept empty / NA).

If you replace these files or use another dataset, ensure you comply with the relevant terms.

---

## No affiliation

This repository is an independent research codebase and is **not affiliated with** Freddie Mac, the SEC, S&P, `maxonlinux/ratings-history`, or any other data provider mentioned above.
