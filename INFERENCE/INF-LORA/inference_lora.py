"""This script runs Text-to-SQL inference on a JSONL test set using a Llama-3 base model and one or more LoRA adapters.

How it works:
1) It reads a JSONL file where each line contains an example with fields "input" (the natural-language question)
   and "output" (the reference SQL).
2) It loads the base Hugging Face causal language model (default: Meta-Llama-3-8B-Instruct) and its tokenizer.
   The tokenizer is configured with left padding and a pad token (falling back to EOS if needed).
3) It loads LoRA adapter checkpoints and runs batched generation for each adapter:
   - If the installed PEFT version supports multi-adapter loading, all adapters are loaded once and switched via
     set_adapter() for faster evaluation.
   - Otherwise, the script falls back to reloading a fresh PEFT model per adapter.
4) For each example, it builds a Llama-3 chat-style prompt and generates SQL, optionally stopping early at the first
   semicolon. Predictions are always truncated to the first semicolon and ended with ';'. Empty outputs become <NO_SQL>.
5) It writes one JSONL results file per adapter with fields: input, gold, pred.

The script keeps the same inference behavior as the original version (padding side, batching, generation settings, and
optional semicolon EOS with a safe regeneration fallback).
"""

import os
import json
import argparse
from typing import List, Dict, Callable, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import PeftModel


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
    print(f"Loaded {len(data):,} examples from {path}")
    return data


def write_results_jsonl(path: str, examples: List[Dict[str, str]], preds: List[str]) -> None:
    assert len(examples) == len(preds)
    with open(path, "w", encoding="utf-8") as f:
        for ex, sql_pred in zip(examples, preds):
            rec = {"input": ex["input"], "gold": ex["output"], "pred": sql_pred}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Results written to {path}")


# Prompt (Llama-3)
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


def build_llama3_prompt(question: str) -> str:
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_MSG}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Generate the SQL query that answers the following question:\n{question}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
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
    device_map: str = "auto",
    attn_implementation: Optional[str] = None,
):
    torch_dtype = _select_dtype(dtype)

    print(f"\nLoading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs = dict(torch_dtype=torch_dtype, device_map=device_map)
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def load_lora_model_on_base(
    base_model,
    lora_dir: str,
    device_map: str = "auto",
    adapter_name: Optional[str] = None,
):
    print(f"Loading LoRA adapter from {lora_dir}")
    kwargs = dict(device_map=device_map)
    if adapter_name is not None:
        kwargs["adapter_name"] = adapter_name
    model = PeftModel.from_pretrained(base_model, lora_dir, **kwargs)
    model.eval()
    return model


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
        description="Text-to-SQL inference on test2.jsonl for multiple LoRA checkpoints (Llama-3-8B-Instruct)."
    )

    parser.add_argument("--test_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--stop_at_semicolon_eos", action="store_true")
    parser.add_argument("--semicolon_fallback_min_chars", type=int, default=20)

    parser.add_argument("--merge_lora", action="store_true")

    parser.add_argument("--attn_implementation", type=str, default=None)

    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto")

    parser.add_argument("--llama3_model_name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--lora_dirs", nargs="+", required=True)

    args = parser.parse_args()
    set_seed(args.seed)

    if args.attn_implementation is not None:
        allowed = {"flash_attention_2", "sdpa", "eager"}
        if args.attn_implementation not in allowed:
            raise ValueError(f"--attn_implementation must be one of {sorted(allowed)} or omitted.")

    os.makedirs(args.output_dir, exist_ok=True)

    test_examples = read_jsonl(args.test_jsonl)

    base_model, base_tok = load_base_model_and_tokenizer(
        model_name=args.llama3_model_name,
        dtype=args.dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
    )

    peft_model = None
    adapter_names: List[str] = []

    def _adapter_name_from_dir(path: str) -> str:
        p = os.path.abspath(path)
        return os.path.basename(os.path.normpath(p)) or "lora"

    try:
        first_dir = os.path.abspath(args.lora_dirs[0])
        first_name = _adapter_name_from_dir(first_dir)
        print(f"\nLoading first LoRA adapter: {first_name} from {first_dir}")

        peft_model = load_lora_model_on_base(
            base_model=base_model,
            lora_dir=first_dir,
            device_map=args.device_map,
            adapter_name=first_name,
        )
        adapter_names.append(first_name)

        for extra_dir in args.lora_dirs[1:]:
            extra_dir = os.path.abspath(extra_dir)
            extra_name = _adapter_name_from_dir(extra_dir)
            print(f"Loading additional LoRA adapter: {extra_name} from {extra_dir}")

            if hasattr(peft_model, "load_adapter"):
                peft_model.load_adapter(extra_dir, adapter_name=extra_name)
                adapter_names.append(extra_name)
            else:
                raise AttributeError("PeftModel has no load_adapter() in this version.")
    except Exception as e:
        print(f"\nMulti-adapter loading not available ({e}). Falling back to reloading per adapter.")
        peft_model = None
        adapter_names = []

    if peft_model is not None and len(adapter_names) > 0:
        for adapter_name in adapter_names:
            out_path = os.path.join(args.output_dir, f"results_lora_{adapter_name}.jsonl")
            print(f"\nLoRA generation (adapter={adapter_name})")

            if hasattr(peft_model, "set_adapter"):
                peft_model.set_adapter(adapter_name)

            if args.merge_lora:
                print("--merge_lora is ignored when multiple adapters are loaded in a single PeftModel.")

            preds = generate_sql_for_examples(
                model=peft_model,
                tokenizer=base_tok,
                examples=test_examples,
                prompt_fn=build_llama3_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                batch_size=args.batch_size,
                stop_at_semicolon_eos=args.stop_at_semicolon_eos,
                semicolon_fallback_min_chars=args.semicolon_fallback_min_chars,
            )
            write_results_jsonl(out_path, test_examples, preds)

        del peft_model
        torch.cuda.empty_cache()

    else:
        for lora_dir in args.lora_dirs:
            lora_dir = os.path.abspath(lora_dir)
            adapter_name = _adapter_name_from_dir(lora_dir)
            out_path = os.path.join(args.output_dir, f"results_lora_{adapter_name}.jsonl")

            model = load_lora_model_on_base(
                base_model=base_model,
                lora_dir=lora_dir,
                device_map=args.device_map,
                adapter_name=None,
            )

            if args.merge_lora and hasattr(model, "merge_and_unload"):
                print("Merging LoRA into base weights for generation (--merge_lora).")
                model = model.merge_and_unload()
                model.eval()

            print(f"\nLoRA generation: {adapter_name}")
            preds = generate_sql_for_examples(
                model=model,
                tokenizer=base_tok,
                examples=test_examples,
                prompt_fn=build_llama3_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                batch_size=args.batch_size,
                stop_at_semicolon_eos=args.stop_at_semicolon_eos,
                semicolon_fallback_min_chars=args.semicolon_fallback_min_chars,
            )
            write_results_jsonl(out_path, test_examples, preds)

            del model
            torch.cuda.empty_cache()

    del base_model
    torch.cuda.empty_cache()

    print("\nLoRA inference completed.")
    print("You can now run `postTest_final.py` on the generated files.")


if __name__ == "__main__":
    main()
