#!/usr/bin/env python3

from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/codereviewer")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/codereviewer")

PROMPT = "<add> import \"fmt\" <keep> foo"

t = tokenizer(PROMPT, truncation=True, padding=True, return_tensors="pt")
out = model.generate(t.input_ids, max_length=256)
print(tokenizer.decode(out[0][2:], skip_special_tokens=True))
