from transformers import BartForConditionalGeneration, AutoTokenizer, Text2TextGenerationPipeline
import datasets
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use %s as device" % device)

tokenizer_pretrain_dir = "IDEA-CCNL/Randeng-BART-139M"

tokenizer=AutoTokenizer.from_pretrained(tokenizer_pretrain_dir, use_fast=True)
