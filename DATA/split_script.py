
"""This script splits a deduplicated fine-tuning dataset into train/validation/test JSONL files.

How it works:
1) It loads a JSON dataset where each example contains at least: "template", "sql", and "question_concept".
2) It groups examples by "variation", defined as the pair (template, sql). Each variation is expected to have 30
   paraphrases (examples).
3) For each template, it takes the first two variations encountered as "seen" variations:
   - 20 paraphrases go to train
   - 5 paraphrases go to validation
   - 5 paraphrases go to test
4) Any remaining variations for each template are treated as "unseen" candidates. The script shuffles these variations,
   then assigns half to validation and half to test, taking 5 paraphrases from each assigned variation.
5) Optionally, it reads an UNANSWERABLES.md file, shuffles the questions, and adds them to the splits with output
   "<NO_SQL>" using a target distribution relative to each split size.
6) It writes train.jsonl, val.jsonl, and test.jsonl to the chosen output directory. Each JSONL line has:
   - "input": the question text
   - "output": the SQL query (or "<NO_SQL>" for unanswerable questions)

The splitting logic and outputs are preserved from the original script; only comments/help strings/emojis are removed and
the module docstring is replaced with this explanation.
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path


# I/O utilities
def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# Splitting logic
def group_by_variation(data):
    variations = defaultdict(list)
    variation_order_by_template = defaultdict(list)
    seen_by_template = defaultdict(set)

    for ex in data:
        template = ex["template"]
        sql = ex["sql"]
        key = (template, sql)

        variations[key].append(ex)

        if key not in seen_by_template[template]:
            seen_by_template[template].add(key)
            variation_order_by_template[template].append(key)

    return variations, variation_order_by_template


def check_variations(variations, variation_order_by_template):
    nb_variations = len(variations)
    print(f"Total number of variations (template, sql): {nb_variations}")
    assert nb_variations == 223, "Expected 223 variations."

    for key, exs in variations.items():
        assert len(exs) == 30, f"Variation {key} does not have 30 paraphrases (len={len(exs)})"

    sizes = [len(vs) for vs in variation_order_by_template.values()]
    nb_templates_1 = sum(1 for t in sizes if t == 1)
    nb_templates_3 = sum(1 for t in sizes if t == 3)

    print(f"Templates with 1 variation: {nb_templates_1}")
    print(f"Templates with 3 variations: {nb_templates_3}")

    assert nb_templates_1 == 1, "Expected 1 template with 1 variation."
    assert nb_templates_3 == 74, "Expected 74 templates with 3 variations."


def build_split(variations, variation_order_by_template, seed=42):
    random.seed(seed)

    train_vars = []
    remaining_vars = []

    for template, var_list in variation_order_by_template.items():
        if len(var_list) >= 2:
            first_two = var_list[:2]
            train_vars.extend(first_two)
            remaining = var_list[2:]
            remaining_vars.extend(remaining)
        else:
            train_vars.extend(var_list)

    print(f"Number of 'seen' variations (train/val/test): {len(train_vars)}")
    print(f"Number of remaining variations (unseen candidates): {len(remaining_vars)}")

    assert len(train_vars) == 149, "Expected 149 seen variations."
    assert len(remaining_vars) == 74, "Expected 74 remaining variations."

    remaining_vars = list(remaining_vars)
    random.shuffle(remaining_vars)
    val_unseen_vars = set(remaining_vars[:37])
    test_unseen_vars = set(remaining_vars[37:])

    train_examples = []
    val_examples = []
    test_examples = []

    for key in train_vars:
        exs = list(variations[key])
        random.shuffle(exs)

        train_part = exs[:20]
        val_part = exs[20:25]
        test_part = exs[25:30]

        for ex in train_part:
            train_examples.append({
                "input": ex["question_concept"],
                "output": ex["sql"],
            })

        for ex in val_part:
            val_examples.append({
                "input": ex["question_concept"],
                "output": ex["sql"],
            })

        for ex in test_part:
            test_examples.append({
                "input": ex["question_concept"],
                "output": ex["sql"],
            })

    for key in remaining_vars:
        exs = list(variations[key])
        random.shuffle(exs)
        chosen = exs[:5]

        if key in val_unseen_vars:
            for ex in chosen:
                val_examples.append({
                    "input": ex["question_concept"],
                    "output": ex["sql"],
                })
        elif key in test_unseen_vars:
            for ex in chosen:
                test_examples.append({
                    "input": ex["question_concept"],
                    "output": ex["sql"],
                })
        else:
            raise RuntimeError("Remaining variation not assigned to val or test.")

    print(f"[BEFORE UNANSWERABLES] train={len(train_examples)}, val={len(val_examples)}, test={len(test_examples)}")
    return train_examples, val_examples, test_examples


# Unanswerables
def compute_unanswerable_distribution(total_unans, train_base, val_base, test_base):
    target_train = 0.10 / 0.90 * train_base
    target_val = 0.20 / 0.80 * val_base
    target_test = 0.20 / 0.80 * test_base

    t_train = target_train
    t_val = target_val
    t_test = target_test

    sum_targets = t_train + t_val + t_test

    if total_unans <= 0 or sum_targets <= 0:
        return 0, 0, 0

    max_total = sum_targets
    if total_unans >= max_total:
        factor = 1.0
        M = max_total
    else:
        factor = total_unans / max_total
        M = total_unans

    n_train = int(round(t_train * factor))
    n_val = int(round(t_val * factor))
    n_test = int(round(t_test * factor))

    current_sum = n_train + n_val + n_test
    diff = int(round(M - current_sum))

    for _ in range(abs(diff)):
        if diff > 0:
            if t_train >= t_val and t_train >= t_test:
                n_train += 1
            elif t_val >= t_train and t_val >= t_test:
                n_val += 1
            else:
                n_test += 1
        elif diff < 0:
            if n_train >= n_val and n_train >= n_test and n_train > 0:
                n_train -= 1
            elif n_val >= n_train and n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

    if n_train + n_val + n_test > total_unans:
        surplus = n_train + n_val + n_test - total_unans
        for _ in range(surplus):
            if n_train >= n_val and n_train >= n_test and n_train > 0:
                n_train -= 1
            elif n_val >= n_train and n_val >= n_test and n_val > 0:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1

    return n_train, n_val, n_test


def load_unanswerables(path, train_base, val_base, test_base, seed=42):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q:
                continue
            if q.startswith("#"):
                continue
            questions.append(q)

    total = len(questions)
    if total == 0:
        print("UNANSWERABLES.md is empty or contains no valid question.")
        return [], [], []

    random.seed(seed)
    random.shuffle(questions)

    n_train, n_val, n_test = compute_unanswerable_distribution(
        total_unans=total,
        train_base=train_base,
        val_base=val_base,
        test_base=test_base,
    )

    print(
        f"UNANSWERABLES: total={total}, distribution -> train={n_train}, val={n_val}, test={n_test}"
    )

    idx_train_end = n_train
    idx_val_end = n_train + n_val
    train_q = questions[:idx_train_end]
    val_q = questions[idx_train_end:idx_val_end]
    test_q = questions[idx_val_end:idx_val_end + n_test]

    train_ex = [{"input": q, "output": "<NO_SQL>"} for q in train_q]
    val_ex = [{"input": q, "output": "<NO_SQL>"} for q in val_q]
    test_ex = [{"input": q, "output": "<NO_SQL>"} for q in test_q]

    return train_ex, val_ex, test_ex


def write_jsonl(examples, path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str
    )
    parser.add_argument(
        "--unanswerables",
        type=str
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    unans_path = Path(args.unanswerables)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {input_path}...")
    data = load_data(input_path)

    variations, variation_order_by_template = group_by_variation(data)
    check_variations(variations, variation_order_by_template)

    train_ex, val_ex, test_ex = build_split(
        variations, variation_order_by_template, seed=args.seed
    )

    train_base = len(train_ex)
    val_base = len(val_ex)
    test_base = len(test_ex)

    if unans_path.exists():
        print(f"Loading unanswerables from {unans_path}...")
        un_train, un_val, un_test = load_unanswerables(
            unans_path,
            train_base=train_base,
            val_base=val_base,
            test_base=test_base,
            seed=args.seed,
        )
        train_ex.extend(un_train)
        val_ex.extend(un_val)
        test_ex.extend(un_test)
    else:
        print("No UNANSWERABLES.md found, no unanswerable examples added.")

    print(f"[AFTER UNANSWERABLES] train={len(train_ex)}, val={len(val_ex)}, test={len(test_ex)}")

    write_jsonl(train_ex, output_dir / "train.jsonl")
    write_jsonl(val_ex, output_dir / "val.jsonl")
    write_jsonl(test_ex, output_dir / "test.jsonl")

    print(f"Files written to: {output_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
