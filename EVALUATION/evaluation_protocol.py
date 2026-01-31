
"""This script evaluates Text-to-SQL predictions stored in JSONL files for an EHR/OMOP setting.

How it works:
1) It loads one or more JSONL result files. Each line is expected to include:
   - "gold": the reference SQL (or "<NO_SQL>" for unanswerable questions)
   - "pred": the model prediction (or "<NO_SQL>" if the model abstains)
   - "input": the question text (optional for evaluation)
2) It computes text-based metrics on answerable questions:
   - Exact Match (EM)
   - Normalized Exact Match (EMN)
   - ROUGE-L (average token-level F1)
3) It optionally connects to PostgreSQL (via psycopg2) and measures execution-based correctness by running gold and
   predicted SQL in read-only mode with a timeout, then comparing normalized result sets.
4) It computes answerability and execution metrics:
   - F1_ans: how well the model decides whether to answer
   - Execution Accuracy and F1_exe: correctness among executable answers
5) It computes reliability scores RS(c) for multiple penalty values c.
6) It prints per-file results, aggregates metrics across folds by model name, prints summary tables, and exports CSV
   files with per-fold, per-model, and global comparison results.

The evaluation logic and outputs are preserved from the original script; only emojis, argparse help strings, and
non-section comments are removed, and this module docstring is replaced with this explanation.
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple
import csv

try:
    import psycopg2
except Exception:
    psycopg2 = None


                                                                             
# JSONL utilities
                                                                             

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


                                                                             
                                                
                                                                             

# Normalization & text metrics
def normalize_sql_for_emn(sql: str) -> str:
    s = sql.strip().lower()
    if s.endswith(";"):
        s = s[:-1]
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def exact_match(gold: str, pred: str) -> int:
    return int(gold.strip() == pred.strip())


def exact_match_norm(gold: str, pred: str) -> int:
    return int(normalize_sql_for_emn(gold) == normalize_sql_for_emn(pred))


def lcs_length(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1):
        ai = a[i-1]
        row = dp[i]
        prev_row = dp[i-1]
        for j in range(1, m+1):
            if ai == b[j-1]:
                row[j] = prev_row[j-1] + 1
            else:
                row[j] = max(row[j-1], prev_row[j])
    return dp[n][m]


def rouge_l_f1(gold: str, pred: str) -> float:
    gold_toks = gold.strip().split()
    pred_toks = pred.strip().split()
    if not gold_toks and not pred_toks:
        return 1.0
    if not gold_toks or not pred_toks:
        return 0.0
    lcs = lcs_length(gold_toks, pred_toks)
    r = lcs / len(gold_toks)
    p = lcs / len(pred_toks)
    if r + p == 0:
        return 0.0
    return 2 * r * p / (r + p)


                                                                             
# PostgreSQL connection & SQL execution
                                                                             

def make_pg_conn(args):
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is not installed. Install it with 'pip install psycopg2-binary'.")
    if args.pg_conn_str:
        return psycopg2.connect(args.pg_conn_str)
    if not args.pg_db:
        raise ValueError("--pg_db is required if --pg_conn_str is not provided.")
    return psycopg2.connect(
        dbname=args.pg_db,
        user=args.pg_user,
        password=args.pg_password,
        host=args.pg_host,
        port=args.pg_port,
    )


def execute_sql_readonly(
    conn,
    sql_text: str,
    timeout_ms: int = 3000,
    max_rows: Optional[int] = None,
) -> Tuple[Optional[List[str]], Optional[List[Tuple[Any, ...]]], Optional[Exception]]:
    sql = (sql_text or "").strip()
    if not sql:
        return None, None, ValueError("Empty SQL")

    if sql.endswith(";"):
        sql = sql[:-1]

    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL default_transaction_read_only = on;")
            cur.execute(f"SET LOCAL statement_timeout = '{int(timeout_ms)}ms';")
            if max_rows is not None and max_rows > 0:
                wrapped = f"SELECT * FROM ({sql}) AS t LIMIT {int(max_rows)}"
            else:
                wrapped = sql
            cur.execute(wrapped)
            desc = cur.description
            if desc is None:
                cols = []
                rows = []
            else:
                cols = [d.name for d in desc]
                rows = cur.fetchall()
        conn.commit()
        return cols, rows, None
    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        return None, None, e


def normalize_value_for_result(v: Any) -> Any:
    return "" if v is None else str(v)


def normalize_result(
    cols: List[str],
    rows: List[Tuple[Any, ...]],
    sort_rows: bool = True,
) -> Tuple[Tuple[str, ...], List[Tuple[Any, ...]]]:
    col_tuple = tuple(cols)
    norm_rows = [tuple(normalize_value_for_result(v) for v in row) for row in rows]
    if sort_rows:
        norm_rows.sort()
    return col_tuple, norm_rows


def results_equal(
    cols_gold: List[str],
    rows_gold: List[Tuple[Any, ...]],
    cols_pred: List[str],
    rows_pred: List[Tuple[Any, ...]],
) -> bool:
    c_g, r_g = normalize_result(cols_gold, rows_gold, sort_rows=True)
    c_p, r_p = normalize_result(cols_pred, rows_pred, sort_rows=True)
    return (c_g == c_p) and (r_g == r_p)


                                                                             
                                       
                                                                             

# Metrics computation
def evaluate_file(
    jsonl_path: str,
    conn,
    timeout_ms: int = 3000,
    max_rows: Optional[int] = None,
    rs_c_values: Optional[List[float]] = None,
) -> Dict[str, Any]:
    if rs_c_values is None:
        rs_c_values = [0.0, 1.0, 10.0]

    data = read_jsonl(jsonl_path)
    n_total = len(data)

    em_count = 0
    emn_count = 0
    rouge_sum = 0.0
    rouge_n = 0

    tp_ans = 0
    fp_ans = 0
    fn_ans = 0

    actually_answerable = 0
    predicted_answerable = 0
    correct_exec = 0

    gold_exec_errors = 0
    pred_exec_errors = 0

    rs_entries: List[Tuple[int, int, Optional[bool]]] = []

    for ex in data:
        gold = ex.get("gold", "").strip()
        pred = ex.get("pred", "").strip()

        is_answerable = int(gold != "<NO_SQL>")
        model_answers = int(pred != "<NO_SQL>")

        if is_answerable:
            em_count += exact_match(gold, pred)
            emn_count += exact_match_norm(gold, pred)
            rouge_sum += rouge_l_f1(gold, pred)
            rouge_n += 1

        if is_answerable == 1 and model_answers == 1:
            tp_ans += 1
        elif is_answerable == 0 and model_answers == 1:
            fp_ans += 1
        elif is_answerable == 1 and model_answers == 0:
            fn_ans += 1

        exec_correct: Optional[bool] = None

        if is_answerable == 1 and model_answers == 1:
            cols_gold, rows_gold, err_gold = execute_sql_readonly(
                conn,
                gold,
                timeout_ms=timeout_ms,
                max_rows=max_rows,
            )
            if err_gold is not None:
                gold_exec_errors += 1
                continue

            actually_answerable += 1
            predicted_answerable += 1

            cols_pred, rows_pred, err_pred = execute_sql_readonly(
                conn,
                pred,
                timeout_ms=timeout_ms,
                max_rows=max_rows,
            )
            if err_pred is not None:
                pred_exec_errors += 1
                exec_correct = False
            else:
                ok = results_equal(cols_gold, rows_gold, cols_pred, rows_pred)
                exec_correct = ok
                if ok:
                    correct_exec += 1

        elif is_answerable == 1 and model_answers == 0:
            cols_gold, rows_gold, err_gold = execute_sql_readonly(
                conn,
                gold,
                timeout_ms=timeout_ms,
                max_rows=max_rows,
            )
            if err_gold is not None:
                gold_exec_errors += 1
                continue

            actually_answerable += 1

        rs_entries.append((is_answerable, model_answers, exec_correct))

    denom = rouge_n if rouge_n > 0 else 1
    em = em_count / denom
    emn = emn_count / denom
    rouge_l = rouge_sum / denom

    prec_ans = tp_ans / (tp_ans + fp_ans) if (tp_ans + fp_ans) > 0 else 0.0
    rec_ans = tp_ans / (tp_ans + fn_ans) if (tp_ans + fn_ans) > 0 else 0.0
    f1_ans = 2 * prec_ans * rec_ans / (prec_ans + rec_ans) if (prec_ans + rec_ans) > 0 else 0.0

    execution_accuracy = correct_exec / actually_answerable if actually_answerable > 0 else 0.0

    prec_exe = correct_exec / predicted_answerable if predicted_answerable > 0 else 0.0
    rec_exe = correct_exec / actually_answerable if actually_answerable > 0 else 0.0
    f1_exe = 2 * prec_exe * rec_exe / (prec_exe + rec_exe) if (prec_exe + rec_exe) > 0 else 0.0

    rs_scores: Dict[str, float] = {}

    n = len(rs_entries)
    if n > 0 and (float(n) not in rs_c_values):
        rs_c_values = list(rs_c_values) + [float(n)]

    for c in rs_c_values:
        total = 0.0
        denom_rs = 0
        for (y, a, e) in rs_entries:
            if y == 0:
                if a == 0:
                    score = 1.0
                else:
                    score = -float(c)
                total += score
                denom_rs += 1
            else:
                if a == 0:
                    score = 0.0
                    total += score
                    denom_rs += 1
                else:
                    if e is None:
                        continue
                    if e:
                        score = 1.0
                    else:
                        score = -float(c)
                    total += score
                    denom_rs += 1

        rs_value = total / denom_rs if denom_rs > 0 else 0.0
        key = f"RS_{int(c) if float(c).is_integer() else c}"
        rs_scores[key] = rs_value

    model_name = os.path.basename(jsonl_path)

    results: Dict[str, Any] = {
        "file": jsonl_path,
        "model": model_name,
        "n_examples": n_total,
        "em": em,
        "emn": emn,
        "rouge_l": rouge_l,
        "execution_accuracy": execution_accuracy,
        "prec_ans": prec_ans,
        "rec_ans": rec_ans,
        "f1_ans": f1_ans,
        "prec_exe": prec_exe,
        "rec_exe": rec_exe,
        "f1_exe": f1_exe,
        "gold_exec_errors": gold_exec_errors,
        "pred_exec_errors": pred_exec_errors,
    }
    results.update(rs_scores)

    return results


def print_results(res: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"File: {res['file']}")
    print(f"Model (approx.): {res['model']}")
    print("-" * 80)
    print(f"Number of examples: {res['n_examples']}")
    print()
    print("Text metrics (on answerable questions):")
    print(f"  EM   : {res['em']:.4f}")
    print(f"  EMN  : {res['emn']:.4f}")
    print(f"  ROUGE-L (avg F1) : {res['rouge_l']:.4f}")
    print()
    print("Answerability (F1_ans):")
    print(f"  Precision_ans : {res['prec_ans']:.4f}")
    print(f"  Recall_ans    : {res['rec_ans']:.4f}")
    print(f"  F1_ans        : {res['f1_ans']:.4f}")
    print()
    print("Execution Accuracy & F1_exe:")
    print(f"  Execution Accuracy (EA / R_exe) : {res['execution_accuracy']:.4f}")
    print(f"  Precision_exe                  : {res['prec_exe']:.4f}")
    print(f"  Recall_exe (R_exe)             : {res['rec_exe']:.4f}")
    print(f"  F1_exe                         : {res['f1_exe']:.4f}")
    print()
    print("Reliability Scores RS(c):")
    for k, v in sorted((k, v) for k, v in res.items() if k.startswith("RS_")):
        print(f"  {k} : {v:.4f}")
    print()
    print("SQL execution info:")
    print(f"  GOLD execution errors : {res['gold_exec_errors']}")
    print(f"  PRED execution errors : {res['pred_exec_errors']}")
    print("=" * 80 + "\n")


                                                                             
# Main
                                                                             

def main():
    parser = argparse.ArgumentParser(
        description="Complete Text-to-SQL evaluation (EM, ROUGE-L, Execution Accuracy, RS(c), F1_ans, F1_exe)."
    )
    parser.add_argument(
        "--results_jsonl",
        nargs="+",
        required=True,
    )

    parser.add_argument("--pg_conn_str", type=str, default=None)
    parser.add_argument("--pg_host", type=str, default="localhost")
    parser.add_argument("--pg_port", type=int, default=5432)
    parser.add_argument("--pg_db", type=str, default="OMOP")
    parser.add_argument("--pg_user", type=str, default="postgres")
    parser.add_argument("--pg_password", type=str, default=None)
    parser.add_argument("--pg_schema", type=str, default="omop")

    parser.add_argument("--pg_timeout_ms", type=int, default=5000)
    parser.add_argument("--max_rows", type=int, default=None)

    parser.add_argument(
        "--rs_c",
        nargs="*",
        type=float,
        default=[0.0, 1.0, 10.0],
    )

    args = parser.parse_args()
    conn = make_pg_conn(args)

    if args.pg_schema:
        with conn.cursor() as cur:
            cur.execute(f"SET search_path TO {args.pg_schema}, public;")
        conn.commit()

    try:
        results_per_file = []
        
        for path in args.results_jsonl:
            res = evaluate_file(
                jsonl_path=path,
                conn=conn,
                timeout_ms=args.pg_timeout_ms,
                max_rows=args.max_rows,
                rs_c_values=args.rs_c,
            )
            results_per_file.append(res)
            print_results(res)

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for res in results_per_file:
            model = res["model"]
            model = re.sub(r"_fold\d+", "", model)
            model = re.sub(r"results_", "", model)
            model = re.sub(r"\.jsonl$", "", model)
            grouped.setdefault(model, []).append(res)

        print("\n" + "="*120)
        print(" MODEL-WISE AVERAGED RESULTS (across N folds)")
        print("="*120)

        def mean(values):
            return sum(values) / len(values)

        aggregated_models = {}

        count_keys = {"n_examples", "gold_exec_errors", "pred_exec_errors"}

        metric_keys = {
            "em",
            "emn",
            "rouge_l",
            "execution_accuracy",
            "prec_ans",
            "rec_ans",
            "f1_ans",
            "prec_exe",
            "rec_exe",
            "f1_exe",
        }

        for model, entries in grouped.items():
            agg: Dict[str, float] = {}

            for k in count_keys:
                if k in entries[0]:
                    agg[k] = sum(e[k] for e in entries)

            rs_keys = sorted({k for e in entries for k in e.keys() if k.startswith("RS_")})
            for k in rs_keys:
                vals = [e[k] for e in entries if k in e]
                if vals:
                    agg[k] = mean(vals)

            for k in metric_keys:
                if k in entries[0]:
                    agg[k] = mean([e[k] for e in entries])

            aggregated_models[model] = agg

            print(f"\n Model: {model}")
            print(f"  Number of folds: {len(entries)}")
            for k, v in sorted(agg.items()):
                print(f"  {k:25s} : {v:.4f}")

        print("\n" + "="*120)
        print(" GLOBAL TABLE — Model comparison (averaged over folds)")
        print("="*120)

        main_metrics = [
            "execution_accuracy",
            "f1_exe",
            "f1_ans",
            "em",
            "emn",
            "rouge_l",
            "RS_0",
            "RS_1",
            "RS_10"
        ]

        header = "Model".ljust(20) + "".join(m.ljust(15) for m in main_metrics)
        print(header)
        print("-"*120)

        for model, metrics in aggregated_models.items():
            row = model.ljust(20)
            for m in main_metrics:
                val = metrics.get(m, 0.0)
                row += f"{val:.4f}".ljust(15)
            print(row)

        print("\n Evaluation completed.")
# CSV export

                                                                            
                       
                                                                            

        all_fold_keys = sorted({k for res in results_per_file for k in res.keys()})

        with open("evaluation_per_fold.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_fold_keys)
            writer.writeheader()
            for res in results_per_file:
                writer.writerow(res)

        model_rows = []
        all_model_keys = set(["model"])
        for model, metrics in aggregated_models.items():
            row = {"model": model}
            row.update(metrics)
            model_rows.append(row)
            all_model_keys.update(row.keys())

        all_model_keys = sorted(all_model_keys)

        with open("evaluation_per_model.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_model_keys)
            writer.writeheader()
            for row in model_rows:
                writer.writerow(row)

        global_rows = []
        for model, metrics in aggregated_models.items():
            row = {"model": model}
            for m in main_metrics:
                row[m] = metrics.get(m, 0.0)
            global_rows.append(row)

        global_fieldnames = ["model"] + main_metrics

        with open("evaluation_global.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=global_fieldnames)
            writer.writeheader()
            for row in global_rows:
                writer.writerow(row)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
