# experiments/beta_binom_jeffreys/make_table_beta_binom.py
# -*- coding: utf-8 -*-
"""
Build a paper-ready LaTeX table for the Beta–Binomial robustness experiment
(fixed (n, p_true) scenarios, varying clustering levels).

Reads:
  - data/beta_binom_results_scenarios.csv if present
  - else data/beta_binom_results.csv filtered to design=="scenarios" (if column exists)

Writes:
  - tables/beta_binom_rho_table.tex
  - tables/beta_binom_rho_table_tidy.csv
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

METHOD_ORDER: List[str] = ["jeffreys", "cp", "normal"]
METHOD_LABELS: Dict[str, str] = {
    "jeffreys": "Jeffreys",
    "cp": "Clopper--Pearson",
    "normal": "Normal approximation",
}

CLUSTER_LEVELS: List[Tuple[str, float]] = [
    ("i.i.d.", 0.00),
    ("Mild", 0.01),
    ("Moderate", 0.05),
    ("Severe", 0.10),
]

SCENARIOS: List[Dict[str, float | str]] = [
    {"scenario": "Baseline", "n": 1000, "p_true": 0.01},
    {"scenario": "Low default", "n": 1000, "p_true": 0.001},
]

DECIMALS = 3


def _get_first_existing(df: pd.DataFrame, candidates: List[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the candidate columns exist: {candidates}")


def _require_cols(df: pd.DataFrame, cols: List[str], context: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        msg = "Missing columns in beta_binom_results.csv: " + ", ".join(missing)
        if context:
            msg += f". {context}"
        raise KeyError(msg)


def _closest_value(target: float, values: np.ndarray) -> float:
    return float(values[np.argmin(np.abs(values - target))])


def _fmt(x: float, d: int = DECIMALS) -> str:
    return f"{x:.{d}f}"


def _cell(lb: float, ub: float, L: float, sd_lb: float, sd_ub: float) -> str:
    return (
        r"\makecell{"
        + rf"$[{_fmt(lb)},\,{_fmt(ub)}]$\\"
        + rf"\footnotesize $L={_fmt(L)}$\\"
        + rf"\footnotesize $\sigma_{{LB}}={_fmt(sd_lb)}$, $\sigma_{{UB}}={_fmt(sd_ub)}$"
        + "}"
    )


def _latex_escape(s: str) -> str:
    return s.replace("&", r"\&")


def build_tidy(df: pd.DataFrame) -> pd.DataFrame:
    _require_cols(df, ["n", "p_true", "rho", "method", "lb_mean", "ub_mean"])

    len_col = _get_first_existing(df, ["len_mean", "avg_length", "length_mean", "interval_len_mean"])

    has_sd = ("lb_sd" in df.columns) and ("ub_sd" in df.columns)
    if not has_sd:
        _require_cols(df, ["lb_var", "ub_var"], context="Need lb_sd/ub_sd or lb_var/ub_var.")
        df = df.copy()
        df["lb_sd"] = np.sqrt(np.maximum(df["lb_var"].astype(float), 0.0))
        df["ub_sd"] = np.sqrt(np.maximum(df["ub_var"].astype(float), 0.0))

    out = df.copy()
    out["n"] = out["n"].astype(int)
    out["p_true"] = out["p_true"].astype(float)
    out["rho"] = out["rho"].astype(float)
    out["method"] = out["method"].astype(str)
    out["lb_mean"] = out["lb_mean"].astype(float)
    out["ub_mean"] = out["ub_mean"].astype(float)
    out["len_mean"] = out[len_col].astype(float)
    out["lb_sd"] = out["lb_sd"].astype(float)
    out["ub_sd"] = out["ub_sd"].astype(float)

    keep = ["n", "p_true", "rho", "method", "lb_mean", "ub_mean", "len_mean", "lb_sd", "ub_sd"]
    for extra in ["conf_level", "n_sim", "scenario", "design"]:
        if extra in out.columns:
            keep.append(extra)
    return out[keep]


def emit_latex_table(tidy: pd.DataFrame, out_path: Path) -> None:
    conf = None
    if "conf_level" in tidy.columns:
        confs = tidy["conf_level"].dropna().unique()
        if len(confs) == 1:
            conf = float(confs[0])
    cap_level = f"{int(round(conf*100))}\\%" if conf is not None else "95\\%"

    avail_rhos = np.sort(tidy["rho"].unique())

    lines: List[str] = []
    lines.append(r"% Requires: \usepackage{booktabs}, \usepackage{threeparttable}, \usepackage{makecell}")
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \begin{threeparttable}")
    lines.append(
        r"  \caption{Beta--Binomial robustness: average interval bounds, lengths, and bound variability ("
        + cap_level
        + r" level)}"
    )
    lines.append(r"  \label{tab:beta_binom_rho}")
    lines.append(r"  \setlength{\tabcolsep}{5pt}")
    lines.append(r"  \renewcommand{\arraystretch}{1.15}")
    lines.append(r"  \small")

    col_spec = "l" + "c" * len(CLUSTER_LEVELS)
    lines.append(rf"  \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    hdr = "    \\textbf{Method}"
    for lab, _ in CLUSTER_LEVELS:
        hdr += rf" & \\textbf{{{_latex_escape(lab)}}}"
    hdr += r" \\"
    lines.append(hdr)
    lines.append(r"    \midrule")

    for s_idx, sc in enumerate(SCENARIOS):
        sc_name = str(sc["scenario"])
        n0 = int(sc["n"])
        p0 = float(sc["p_true"])

        lines.append(rf"    \multicolumn{{{1+len(CLUSTER_LEVELS)}}}{{l}}{{\textit{{{_latex_escape(sc_name)}}}}} \\")
        lines.append(r"    \addlinespace[2pt]")

        sub_sc = tidy[(tidy["n"] == n0) & (np.isclose(tidy["p_true"], p0))].copy()
        if sub_sc.empty:
            raise ValueError(f"No data for scenario {sc_name} (n={n0}, p_true={p0})")

        for m in METHOD_ORDER:
            tmp_m = sub_sc[sub_sc["method"] == m].copy()
            if tmp_m.empty:
                raise ValueError(f"Missing method='{m}' for scenario {sc_name}")

            row = rf"    {METHOD_LABELS.get(m, m)}"
            for _, rho_t in CLUSTER_LEVELS:
                rho_use = _closest_value(rho_t, avail_rhos)
                tmp = tmp_m[np.isclose(tmp_m["rho"], rho_use)].copy()
                if tmp.empty:
                    raise ValueError(f"Missing rho={rho_t} for scenario={sc_name}, method={m}")
                r0 = tmp.iloc[0]
                row += " & " + _cell(
                    lb=float(r0["lb_mean"]),
                    ub=float(r0["ub_mean"]),
                    L=float(r0["len_mean"]),
                    sd_lb=float(r0["lb_sd"]),
                    sd_ub=float(r0["ub_sd"]),
                )
            row += r" \\"
            lines.append(row)

        if s_idx < len(SCENARIOS) - 1:
            lines.append(r"    \addlinespace[6pt]")

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")

    lines.append(r"  \begin{tablenotes}[flushleft]")
    lines.append(r"    \footnotesize")
    lines.append(
        r"    \item \textbf{Notes:} Each cell reports Monte Carlo averages of interval bounds "
        r"$[\overline{\mathrm{LB}},\overline{\mathrm{UB}}]$, the mean length "
        r"$L=\overline{(\mathrm{UB}-\mathrm{LB})}$, and Monte Carlo standard deviations "
        r"$(\sigma_{LB},\sigma_{UB})$ of the bounds across replications. "
        r"Clustering levels correspond to increasing over-dispersion (parameter values are stated in the main text)."
    )
    lines.append(r"  \end{tablenotes}")
    lines.append(r"  \end{threeparttable}")
    lines.append(r"\end{table}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    out_dir = base_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer scenarios-only file to avoid any ambiguity
    scen_path = data_dir / "beta_binom_results_scenarios.csv"
    combined_path = data_dir / "beta_binom_results.csv"

    if scen_path.exists():
        df = pd.read_csv(scen_path)
    elif combined_path.exists():
        df = pd.read_csv(combined_path)
        if "design" in df.columns:
            df = df[df["design"] == "scenarios"].copy()
    else:
        raise FileNotFoundError(f"Missing input CSV in {data_dir}")

    tidy = build_tidy(df)

    tidy_out = out_dir / "beta_binom_rho_table_tidy.csv"
    tidy.to_csv(tidy_out, index=False)

    tex_out = out_dir / "beta_binom_rho_table.tex"
    emit_latex_table(tidy, tex_out)

    print(f"[OK] Saved tidy CSV: {tidy_out}")
    print(f"[OK] Saved LaTeX table: {tex_out}")


if __name__ == "__main__":
    main()
