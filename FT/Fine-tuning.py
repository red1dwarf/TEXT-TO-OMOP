"""This script fine-tunes a LoRA adapter for Text-to-SQL using a Llama-3 Instruct base model.

How it works:
1) It reads training and validation datasets from JSONL files. Each line is expected to contain an example with
   "input" (the natural-language question) and "output" (the target SQL).
2) It builds prompts using the official Llama-3 chat/instruct token format, with a system message that enforces
   OMOP CDM constraints and requires SQL-only outputs.
3) It constructs a PyTorch Dataset that concatenates prompt tokens and answer tokens, then masks the loss on the
   prompt portion (labels = -100) so training optimizes only the SQL output tokens.
4) It loads the base model and applies a PEFT LoRA configuration to the specified target modules.
5) It trains with Hugging Face Trainer and saves the LoRA adapter and tokenizer to the output directory.
6) Optionally, it runs a small deterministic generation preview on validation samples after training.

Training settings (precision, batch sizes, scheduler, checkpointing) are controlled by command-line arguments.
The cleaned version preserves the same training and inference behavior as the original script; only emojis,
argparse help strings, and non-section comments are removed, and this module docstring is replaced.
"""

import os
import json
import math
import random
import argparse
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig, get_peft_model


                               
# Prompt builder for Llama‑3 Instruct
                               

def build_chat_prompt(question: str) -> str:
    system_msg = (
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
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system_msg}<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"Generate the SQL query that answers the following question:\n{question}\n"
        "<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )

                               
# Dataset
                               

def read_jsonl(path: str) -> list[dict[str, str]]:
    data = []
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

class Text2SQLDataset(Dataset):

    def __init__(self, examples: List[Dict[str, str]], tokenizer, max_length=4096):
        self.examples = examples
        self.tok = tokenizer
        self.max_len = max_length
        self.eot_id = self.tok.convert_tokens_to_ids("<|eot_id|>") or self.tok.eos_token_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        prompt = build_chat_prompt(ex['input'].rstrip())
        answer = ex['output'].rstrip()

        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        a_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        if self.eot_id is not None:
            a_ids += [self.eot_id]

        input_ids = p_ids + a_ids

        if len(input_ids) > self.max_len:
            overflow = len(input_ids) - self.max_len
            if overflow < len(p_ids):
                p_ids = p_ids[overflow:]
            else:
                a_ids = a_ids[overflow - len(p_ids):]
                p_ids = []
            input_ids = p_ids + a_ids

        labels = [-100]*len(p_ids) + a_ids
        attention_mask = [1]*len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }

class DataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tok = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, torch.Tensor]]):
        batch = {}
        keys = features[0].keys()
        for k in keys:
            seqs = [f[k] for f in features]
            if k == 'labels':
                pad_val = -100
            elif k == 'attention_mask':
                pad_val = 0
            else:
                pad_val = self.tok.pad_token_id
            batch[k] = torch.nn.utils.rnn.pad_sequence(
                seqs, batch_first=True, padding_value=pad_val
            )
            if self.pad_to_multiple_of is not None:
                seq = batch[k]
                pad_len = (self.pad_to_multiple_of - seq.size(1)%self.pad_to_multiple_of) % self.pad_to_multiple_of
                if pad_len:
                    extra = torch.full((seq.size(0), pad_len), pad_val, dtype=seq.dtype)
                    batch[k] = torch.cat([seq, extra], dim=1)
        return batch

                               
# Main
                               

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_jsonl', type=str, required=True)
    parser.add_argument('--valid_jsonl', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_length', type=int, default=4096)

    parser.add_argument('--lora_r', type=int, default=32)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--target_modules', type=str,
                        default='q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj')

    parser.add_argument('--num_train_epochs', type=float, default=3)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--eval_steps', type=int, default=250)
    parser.add_argument('--save_steps', type=int, default=250)
    parser.add_argument('--save_total_limit', type=int, default=3)

    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')

    parser.add_argument('--do_sample_preview', action='store_true')
    parser.add_argument('--preview_num', type=int, default=3)
    parser.add_argument('--preview_max_new_tokens', type=int, default=512)

    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    args = parser.parse_args()
    set_seed(args.seed)

    print('Loading tokenizer...')
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    tok.model_max_length = args.max_length

    print('Loading base model...')
    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map='auto'
    )

    target_modules = [m.strip() for m in args.target_modules.split(',') if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        task_type='CAUSAL_LM',
        bias='none'
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    model.config.use_cache = False 
    model.enable_input_require_grads()  

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print('Loading datasets...')
    train_examples = read_jsonl(args.train_jsonl)
    valid_examples = read_jsonl(args.valid_jsonl)
    random.shuffle(train_examples)

    ds_train = Text2SQLDataset(train_examples, tok, max_length=args.max_length)
    ds_valid = Text2SQLDataset(valid_examples, tok, max_length=args.max_length)

    collator = DataCollator(tok, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,   
        learning_rate=args.learning_rate,                            
        weight_decay=args.weight_decay,                               
        warmup_ratio=args.warmup_ratio,                             
        lr_scheduler_type=args.lr_scheduler_type,                      
        logging_steps=args.logging_steps,                               
        eval_strategy='steps',                                      
        eval_steps=args.eval_steps,                         
        save_strategy='steps',                                       
        save_steps=args.save_steps, 
        save_total_limit=args.save_total_limit,                        
        bf16=args.bf16,                                                
        fp16=args.fp16 and not args.bf16,                               
        gradient_checkpointing=args.gradient_checkpointing,            
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        report_to=['none']
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        data_collator=collator,
        tokenizer=tok
    )

    print('Start training...')
    trainer.train(resume_from_checkpoint=getattr(args, "resume_from_checkpoint", None))

    print('Evaluating...')
    metrics = trainer.evaluate()
    if 'eval_loss' in metrics:
        try:
            metrics['perplexity'] = math.exp(metrics['eval_loss'])
        except OverflowError:
            metrics['perplexity'] = float('inf')
    trainer.log_metrics('eval', metrics)
    trainer.save_metrics('eval', metrics)

    print('Saving adapter...')
    trainer.model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)

    if args.do_sample_preview:
        preview_samples(valid_examples, tok, model, args.preview_num, args.preview_max_new_tokens)

    print('Done.')

@torch.inference_mode()
def preview_samples(examples, tok, model, n=3, max_new_tokens=256):
    print('\n=== Preview generations ===')
    device = next(model.parameters()).device
    for ex in random.sample(examples, k=min(n, len(examples))):
        prompt = build_chat_prompt(ex['input'].rstrip())
        inputs = tok(prompt, return_tensors='pt').to(device)
        eot_id = tok.convert_tokens_to_ids("<|eot_id|>") or tok.eos_token_id
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eot_id,
            pad_token_id=tok.pad_token_id
        )
        gen = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        gen = gen.split(';', 1)[0] + ';' if ';' in gen else gen
        print(f"\n[Prompt]\n{ex['input']}")
        print(f"[Gold]\n{ex['output']}")
        print(f"[Gen ]\n{gen}")


if __name__ == '__main__':
    main()
