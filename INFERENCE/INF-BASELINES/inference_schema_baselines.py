
"""This script runs baseline Text-to-SQL inference on a JSONL test set using schema-injected prompts.

How it works:
1) It reads a JSONL file where each line contains an example with fields "input" (natural-language question)
   and "output" (reference SQL).
2) It reads a PostgreSQL DDL schema file and injects the schema into every prompt (enforced for all models).
3) It runs inference for up to three baseline Hugging Face causal language models:
   - Meta-Llama-3-8B-Instruct
   - Mistral-7B-Instruct-v0.3
   - Natural-SQL-7B
4) Prompt formatting is model-specific:
   - Llama-3 uses native Llama-3 chat tokens.
   - Mistral uses tokenizer.apply_chat_template when available (with a plain-text fallback).
   - Natural-SQL uses the Natural-SQL model-card style template and includes the same system instructions
     inside the "# Task" section.
5) Generation is batched, uses left padding, slices generations using the padded prompt length, and optionally
   adds ';' as an EOS token (with a safe regeneration fallback if this yields a too-short output).
6) It writes one JSONL results file per model into the output directory with fields: input, gold, pred.

The script preserves the original inference behavior and outputs, except for removing emojis and removing argparse
help strings.
"""

import os
import json
import argparse
from typing import List, Dict, Callable, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# I/O utilities
def read_jsonl(path: str) -> List[Dict[str, str]]:
    data: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            if "input" in ex and "output" in ex:
                question = ex["input"].strip()
                sql = ex["output"].strip()
                if question:
                    data.append({"input": question, "output": sql})
    print(f" Loaded {len(data):,} examples from {path}")
    return data


def write_results_jsonl(path: str, examples: List[Dict[str, str]], preds: List[str]) -> None:
    assert len(examples) == len(preds)
    with open(path, "w", encoding="utf-8") as f:
        for ex, sql_pred in zip(examples, preds):
            rec = {"input": ex["input"], "gold": ex["output"], "pred": sql_pred}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f" Results written to {path}")


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# Prompting
SYSTEM_MSG = (
    "You are an assistant that generates valid PostgreSQL SQL queries for data stored "
    "in an electronic health record following the OMOP Common Data Model (CDM). "
    "Follow these rules strictly:\n"
    "- Use only OMOP CDM tables, columns, and relationships.\n"
    "- Never invent tables or columns.\n"
    "- Never hallucinate schema details.\n"
    "- Never use SELECT *.\n"
    "- Terminate the SQL query with a semicolon.\n"
    "- Output only the SQL query without prose, comments, or explanations.\n"
    "- If the question cannot be answered from OMOP CDM data, output <NO_SQL>.\n"
    "- Stop after the first semicolon."
)


def _user_content(question: str, schema_ddl: str) -> str:
                                                      
    return (
        "You are given the database schema in PostgreSQL DDL.\n"
        "Use it to write a correct query.\n\n"
        "### Database schema (DDL)\n"
        f"{schema_ddl}\n\n"
        "### Question\n"
        f"{question}\n"
    )


def build_llama3_prompt_with_schema(schema_ddl: str) -> Callable[[str], str]:
                                                                         
    def _fn(question: str) -> str:
        user_block = _user_content(question, schema_ddl=schema_ddl)
        return (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{SYSTEM_MSG}<|eot_id|>\n"
            "<|start_header_id|>user<|end_header_id|>\n"
            f"{user_block}\n"
            "<|eot_id|>\n"
            "<|start_header_id|>assistant<|end_header_id|>\n"
        )
    return _fn


def make_chat_prompt_fn(
    tokenizer,
    schema_ddl: str,
    system_msg: str = SYSTEM_MSG,
) -> Callable[[str], str]:
                                                                               
    def _fn(question: str) -> str:
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": _user_content(question, schema_ddl=schema_ddl)},
        ]
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                  
        return (
            f"System: {system_msg}\n\n"
            f"User: {_user_content(question, schema_ddl=schema_ddl)}\n\n"
            f"Assistant:"
        )
    return _fn


                                                                                                
def build_natural_sql_prompt(question: str, schema_ddl: str, system_msg: str = SYSTEM_MSG) -> str:
                                                                                           
    return (
        "# Task\n"
        f"{system_msg}\n\n"
        f"Generate a SQL query to answer the following question: `{question}`\n\n"
        "### PostgreSQL Database Schema\n"
        "The query will run on a database with the following schema:\n\n"
        f"{schema_ddl}\n\n"
        "# SQL\n"
        f"Here is the SQL query that answers the question: `{question}`\n"
        "```sql\n"
    )


# Model loading
def _select_dtype(dtype: str):
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def load_base_model_and_tokenizer(
    model_name: str,
    dtype: str = "bf16",
    attn_implementation: Optional[str] = None,
):
    torch_dtype = _select_dtype(dtype)

    print(f"\n Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

                                                   
    tokenizer.padding_side = "left"

    kwargs = dict(torch_dtype=torch_dtype, device_map="auto")
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


# Generation
@torch.inference_mode()
def generate_sql_for_examples(
    model,
    tokenizer,
    examples: List[Dict[str, str]],
    prompt_fn: Callable[[str], str],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    eos_token_id: Optional[int] = None,
    batch_size: int = 1,
    stop_at_semicolon_eos: bool = False,
    semicolon_fallback_min_chars: int = 20,
) -> List[str]:
    device = next(model.parameters()).device

    if eos_token_id is None:
                                                                          
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot != tokenizer.unk_token_id:
            eos_token_id = eot
        else:
            eos_token_id = tokenizer.eos_token_id

                                                                    
    eos_ids = eos_token_id
    semi_id = None
    if stop_at_semicolon_eos:
        semi_ids = tokenizer.encode(";", add_special_tokens=False)
        if len(semi_ids) == 1:
            semi_id = semi_ids[0]
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            else:
                eos_ids = list(eos_ids)
            if semi_id not in eos_ids:
                eos_ids.append(semi_id)

    preds: List[str] = []

    for start in tqdm(range(0, len(examples), batch_size), desc="SQL generation", unit="batch"):
        batch = examples[start:start + batch_size]
        prompts = [prompt_fn(ex["input"].strip()) for ex in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs["input_ids"].shape[1]                                       

                                                                       
        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids,
            pad_token_id=tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False, num_beams=1)

        output_ids = model.generate(**inputs, **gen_kwargs)

        batch_texts: List[str] = []
        for i in range(len(batch)):
            gen_ids = output_ids[i][prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                                                                   
            if ";" in gen_text:
                gen_text = gen_text.split(";", 1)[0] + ";"

            if not gen_text:
                gen_text = "<NO_SQL>"

            batch_texts.append(gen_text)

                                                                                                
        if stop_at_semicolon_eos and semi_id is not None:
            redo_idxs = [
                i for i, txt in enumerate(batch_texts)
                if (txt != "<NO_SQL>" and len(txt) < semicolon_fallback_min_chars)
            ]
            for i in redo_idxs:
                single_prompt = prompts[i]
                single_in = tokenizer(single_prompt, return_tensors="pt").to(device)

                single_gen_kwargs = dict(gen_kwargs)
                single_gen_kwargs["eos_token_id"] = eos_token_id                   

                out = model.generate(**single_in, **single_gen_kwargs)
                gen_ids = out[0][single_in["input_ids"].shape[1]:]
                txt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                if ";" in txt:
                    txt = txt.split(";", 1)[0] + ";"
                if not txt:
                    txt = "<NO_SQL>"
                batch_texts[i] = txt

        preds.extend(batch_texts)

    return preds


# Main
def main():
    parser = argparse.ArgumentParser(
        description="Baseline Text-to-SQL inference (schema-injected) for Llama3, Mistral, Natural-SQL."
    )
    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

                                              
    parser.add_argument("--schema_path", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stop_at_semicolon_eos", action="store_true")
    parser.add_argument("--semicolon_fallback_min_chars", type=int, default=20)

    parser.add_argument("--attn_implementation", type=str, default=None)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument(
        "--llama3_model_name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
    )
    parser.add_argument(
        "--mistral_model_name",
        type=str,
        default="mistralai/Mistral-7B-Instruct-v0.3",
    )
    parser.add_argument(
        "--natural_sql_model_name",
        type=str,
        default="chatdb/natural-sql-7b",
    )
    parser.add_argument(
        "--run_models",
        type=str,
        default="llama3,mistral,natural",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    run_models = set(m.strip().lower() for m in args.run_models.split(',') if m.strip())
    allowed_models = {'llama3', 'mistral', 'natural'}
    unknown = run_models - allowed_models
    if unknown:
        raise ValueError(
            f"Unknown models in --run_models: {sorted(unknown)}. Allowed: {sorted(allowed_models)}"
        )


    if args.attn_implementation is not None:
        allowed = {"flash_attention_2", "sdpa", "eager"}
        if args.attn_implementation not in allowed:
            raise ValueError(f"--attn_implementation must be one of {sorted(allowed)} or omitted.")

    os.makedirs(args.output_dir, exist_ok=True)

    schema_ddl = read_text_file(args.schema_path)
    if not schema_ddl:
        raise ValueError("--schema_path is required and must not be empty.")

    test_examples = read_jsonl(args.test_jsonl)

    if "llama3" in run_models:
                                   
                                
                                   
        model_llama, tok_llama = load_base_model_and_tokenizer(
            model_name=args.llama3_model_name,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        prompt_llama = build_llama3_prompt_with_schema(schema_ddl)
        
        preds_llama = generate_sql_for_examples(
            model=model_llama,
            tokenizer=tok_llama,
            examples=test_examples,
            prompt_fn=prompt_llama,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            stop_at_semicolon_eos=args.stop_at_semicolon_eos,
            semicolon_fallback_min_chars=args.semicolon_fallback_min_chars,
        )
        llama_out = os.path.join(args.output_dir, "results_llama3_base.jsonl")
        write_results_jsonl(llama_out, test_examples, preds_llama)
        
        del model_llama
        torch.cuda.empty_cache()
        
    else:
        print("  Skipping Llama-3")
    if "mistral" in run_models:
                                   
                                     
                                   
        model_mistral, tok_mistral = load_base_model_and_tokenizer(
            model_name=args.mistral_model_name,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        prompt_mistral = make_chat_prompt_fn(tok_mistral, schema_ddl=schema_ddl)
        
        preds_mistral = generate_sql_for_examples(
            model=model_mistral,
            tokenizer=tok_mistral,
            examples=test_examples,
            prompt_fn=prompt_mistral,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
            stop_at_semicolon_eos=args.stop_at_semicolon_eos,
            semicolon_fallback_min_chars=args.semicolon_fallback_min_chars,
        )
        mistral_out = os.path.join(args.output_dir, "results_mistral_7b_instruct_v0_3.jsonl")
        write_results_jsonl(mistral_out, test_examples, preds_mistral)
        
        del model_mistral
        torch.cuda.empty_cache()
        
    else:
        print("  Skipping Mistral")
    if "natural" in run_models:
                                   
                           
                                   
        model_nat, tok_nat = load_base_model_and_tokenizer(
            model_name=args.natural_sql_model_name,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
        
        def prompt_nat_fn(q: str) -> str:
                                                                         
            return build_natural_sql_prompt(q, schema_ddl, system_msg=SYSTEM_MSG)
        
                                                                      
        NAT_EOS_ID = 100001
        tok_nat.pad_token_id = NAT_EOS_ID
        
        preds_nat = generate_sql_for_examples(
            model=model_nat,
            tokenizer=tok_nat,
            examples=test_examples,
            prompt_fn=prompt_nat_fn,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=NAT_EOS_ID,
            batch_size=args.batch_size,
            stop_at_semicolon_eos=args.stop_at_semicolon_eos,
            semicolon_fallback_min_chars=args.semicolon_fallback_min_chars,
        )
        nat_out = os.path.join(args.output_dir, "results_natural_sql_7b.jsonl")
        write_results_jsonl(nat_out, test_examples, preds_nat)
        
        del model_nat
        torch.cuda.empty_cache()
        
    else:
        print("  Skipping Natural-SQL")
    print("\n Baseline inference completed.")


if __name__ == "__main__":
    main()
