# Data Disclaimer (Third-Party Data)

This repository contains **code** to process and analyze external datasets.  
It does **not** redistribute restricted third-party datasets unless explicitly stated.

## Freddie Mac Single-Family Loan-Level Dataset (Standard)

The main empirical application uses the **Freddie Mac Single-Family Loan-Level Dataset (Standard)**.

Official access point:
- https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset

### Important notes
- The dataset is **owned and distributed by Freddie Mac** and may be subject to
  specific **terms of use, access conditions, and redistribution restrictions**.
- Users of this repository are responsible for:
  1) downloading the data from Freddie Mac through the official channel, and  
  2) complying with the dataset’s terms and any applicable legal/regulatory requirements.
- The authors of this repository do **not** provide any warranty regarding the data
  (availability, correctness, completeness, or suitability).

### What this repo expects locally
This project expects the Freddie Mac files to be stored locally under:
- `data/raw/mortgage_data/`

See `data/README.md` for the exact directory layout used by the pipeline.

## Other data sources
If you add macro series (e.g., from FRED or other providers) under `data/raw/macro/`,
those series remain subject to their respective licenses/terms.

## No affiliation
This repository is an independent research codebase and is **not affiliated with**
Freddie Mac or any data provider mentioned above.
```

---