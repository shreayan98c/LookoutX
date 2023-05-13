#!/usr/bin/python

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")

model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")

model.save_pretrained('/mnt/d/LLaMa/model', from_pt=True)