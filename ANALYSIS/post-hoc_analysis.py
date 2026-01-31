"""This script builds a corrected 6-metric diagnostics table for OMOP Text-to-SQL result files.

How it works:
1) It reads one or more JSONL files where each line contains at least "gold" and "pred".
   Values may be SQL strings or "<NO_SQL>".
2) It normalizes and extracts SQL from predictions, handling cases where model output includes prose or code fences.
3) It computes six diagnostics (plus unanswerable recall) on answerable examples:
   - False abstain (%): predicted "<NO_SQL>" when the gold is answerable
   - Schema drift (%): predicted FROM/JOIN tokens that are not OMOP tables, CTEs, or whitelisted set-returning functions,
     while ignoring "FROM" used inside common SQL functions (e.g., EXTRACT(... FROM ...))
   - Different tables (%): whether the base tables used by gold and prediction differ (CTEs excluded)
   - DATEDIFF usage (%): whether the prediction uses DATEDIFF
   - CTE usage (%): whether the prediction starts with WITH
   - visit_detail miss (%): among questions whose gold SQL uses visit_detail, whether the prediction omits it
4) It writes a CSV file with one row per input JSONL file.

The cleaned version preserves the same computations and outputs as the original script; only emojis, argparse help strings,
and non-section comments are removed, and this module docstring is replaced with this explanation.
"""

from __future__ import annotations

import argparse, json, re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd

# Constants
OMOP_TABLES=set("""
person observation_period visit_occurrence visit_detail condition_occurrence drug_exposure procedure_occurrence device_exposure measurement observation death note note_nlp specimen fact_relationship location care_site provider payer_plan_period cost
drug_era dose_era condition_era cohort cohort_definition metadata
concept concept_ancestor concept_relationship concept_synonym vocabulary domain concept_class relationship drug_strength source_to_concept_map cdm_source
attribute_definition episode episode_event
""".split())

                                                                    
FROM_FUNC_WHITELIST=set("""
unnest generate_series json_each json_each_text jsonb_each jsonb_each_text
json_array_elements jsonb_array_elements json_array_elements_text jsonb_array_elements_text
regexp_split_to_table
""".split())

SQL_START_RE = re.compile(r"\b(with|select|insert|update|delete)\b", re.IGNORECASE)

                                                                                      
                                          
CTE_PAT = re.compile(
    r'(?ix)'
    r'(?:\bwith\b\s+(?:recursive\s+)?)'
    r'("?[A-Za-z_][\w]*"?)\s+as\s*(?:not\s+materialized\s+|materialized\s+)?\('
    r'|,\s*("?[A-Za-z_][\w]*"?)\s+as\s*(?:not\s+materialized\s+|materialized\s+)?\('
)

# SQL parsing
def norm_sql(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"^```[\w]*\s*", "", s.strip())
    s = re.sub(r"\s*```$", "", s.strip())
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def is_no_sql(s: str) -> bool:
    return norm_sql(s).upper() == "<NO_SQL>"

def extract_sql(text: str) -> str:
    """Best-effort extraction from model output that may contain explanation + SQL."""
    if text is None:
        return ""
    t = str(text).strip()
    if not t:
        return ""
    if "```" in t:
        m = re.search(r"```sql\s*(.*?)```", t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s*(.*?)```", t, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
    m = SQL_START_RE.search(t)
    if not m:
        return ""
    sql = t[m.start():].strip()
    if ";" in sql:
        sql = sql[: sql.rfind(";")+1]
    return sql.strip()

def read_jsonl(path: str):
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def extract_cte_names(sql: str):
    s = norm_sql(sql)
    if not re.match(r"(?i)^\s*with\b", s):
        return set()
    names=set()
    for m in CTE_PAT.finditer(s):
        g1, g2 = m.group(1), m.group(2)
        if g1: names.add(g1.strip('"').lower())
        if g2: names.add(g2.strip('"').lower())
    return names

def extract_from_join_tokens(sql: str):
    """
    Extract base identifiers after FROM/JOIN.

    Important: ignores occurrences of "FROM <token>" inside functions like:
      EXTRACT(... FROM ...), DATE_PART, SUBSTRING(... FROM ...), etc.
    """
    s = norm_sql(sql)
    s = re.sub(r"'.*?'", "''", s)                                                 
    toks=[]
    for m in re.finditer(r"\b(from|join)\s+([A-Za-z_][\w\.]*|\"[^\"]+\")", s, flags=re.I):
        pre = s[max(0, m.start()-30):m.start()].lower()
        if any(k in pre for k in ["extract", "date_part", "datepart", "substring", "trim", "overlay", "position"]):
            continue
        tok = m.group(2).strip('"')
        base = tok.split('.')[-1].lower()
        toks.append(base)
    return toks

def base_tables(sql: str):
    s = norm_sql(sql)
    ctes = extract_cte_names(s)
    toks = extract_from_join_tokens(s)
    return set(t for t in toks if t not in ctes)

def schema_drift(sql: str) -> bool:
    """
    Drift = at least one FROM/JOIN 'relation' token is neither:
    - an OMOP table,
    - a CTE name,
    - a whitelisted set-returning function.

    (Computed on answerable, non-abstained predictions.)
    """
    s = norm_sql(sql)
    if not s or is_no_sql(s):
        return False
    ctes = extract_cte_names(s)
    toks = extract_from_join_tokens(s)
    for t in toks:
        if t in ctes:
            continue
        if t in OMOP_TABLES:
            continue
        if t in FROM_FUNC_WHITELIST:
            continue
        return True
    return False

# Metrics
def compute_6_metrics(path: str) -> dict:
    rows=read_jsonl(path)
    gold=[r.get("gold","") for r in rows]
    pred=[r.get("pred","") for r in rows]

    gold_no=np.array([is_no_sql(g) for g in gold], dtype=bool)
    pred_no=np.array([is_no_sql(p) for p in pred], dtype=bool)
    ans_mask=~gold_no

    gold_sql=[norm_sql(g) for g in gold]
    pred_sql=[norm_sql(extract_sql(p)) if not pred_no[i] else "<NO_SQL>" for i,p in enumerate(pred)]

                                                     
    false_abstain = (ans_mask & pred_no).sum() / ans_mask.sum() if ans_mask.sum() else np.nan

                                                             
    unanswerable_recall = (gold_no & pred_no).sum() / gold_no.sum() if gold_no.sum() else np.nan

    drift_flags=[]
    table_mismatch=[]
    datediff_flags=[]
    cte_flags=[]

    for i in range(len(rows)):
        if not ans_mask[i] or pred_no[i]:
            continue
        drift_flags.append(schema_drift(pred_sql[i]))
        table_mismatch.append(base_tables(gold_sql[i]) != base_tables(pred_sql[i]))
        datediff_flags.append(re.search(r"(?i)\bdatediff\b", pred_sql[i]) is not None)
        cte_flags.append(re.match(r"(?i)^\s*with\b", pred_sql[i]) is not None)

    schema_drift_rate = float(np.mean(drift_flags)) if drift_flags else np.nan
    tables_diff_rate = float(np.mean(table_mismatch)) if table_mismatch else np.nan
    datediff_rate = float(np.mean(datediff_flags)) if datediff_flags else np.nan
    cte_rate = float(np.mean(cte_flags)) if cte_flags else np.nan

                                                                                
                               
    vd_miss=[]
    for i in range(len(rows)):
        if not ans_mask[i]:
            continue
        if re.search(r"(?i)\bvisit_detail\b", gold_sql[i]) is None:
            continue
        miss = True if pred_no[i] else (re.search(r"(?i)\bvisit_detail\b", pred_sql[i]) is None)
        vd_miss.append(miss)
    visit_detail_miss_rate = float(np.mean(vd_miss)) if vd_miss else np.nan

    return {
        "N": len(rows),
        "False abstain (%)": false_abstain*100 if not np.isnan(false_abstain) else np.nan,
        "Unanswerable Recall (%)": unanswerable_recall*100 if not np.isnan(unanswerable_recall) else np.nan,
        "Schema drift (%)": schema_drift_rate*100 if not np.isnan(schema_drift_rate) else np.nan,
        "Different tables (%)": tables_diff_rate*100 if not np.isnan(tables_diff_rate) else np.nan,
        "DATEDIFF usage (%)": datediff_rate*100 if not np.isnan(datediff_rate) else np.nan,
        "CTE usage (%)": cte_rate*100 if not np.isnan(cte_rate) else np.nan,
        "visit_detail miss (%)": visit_detail_miss_rate*100 if not np.isnan(visit_detail_miss_rate) else np.nan,
    }

def prettify_model_name(p: Path) -> str:
    return p.stem.replace("results_", "").replace("_", "-")

# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="table_6_metrics_corrected.csv")
    args = ap.parse_args()

    rows=[]
    for inp in args.inputs:
        p=Path(inp)
        m=compute_6_metrics(str(p))
        m["Model"]=prettify_model_name(p)
        rows.append(m)

    df=pd.DataFrame(rows)[[
        "Model","N",
        "False abstain (%)",
        "Unanswerable Recall (%)",
        "Schema drift (%)",
        "Different tables (%)",
        "DATEDIFF usage (%)",
        "CTE usage (%)",
        "visit_detail miss (%)",
    ]].round(2)

    df.to_csv(args.out, index=False)
    print(f"Wrote: {Path(args.out).resolve()}")

if __name__ == "__main__":
    main()
