# `ldp_application/data/`

This folder contains the input data used by the LDP / S&P grade application.

## Source
Rating history data is sourced from:
```text
https://ratingshistory.info
````

The site provides credit rating history files from multiple agencies as **CSV exports**
(converted from the original XBRL disclosures).

## Expected input file

By default, the Makefile expects:

* `ldp_application/data/raw/data_rating_corporate.xlsx`

This file is built from the CSV histories downloaded from the source above.

## Outputs

Generated tables and plots are written to:

* `ldp_application/outputs/sp_grade_is_oos/`