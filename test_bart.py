from transformers import BartForConditionalGeneration, AutoTokenizer, Text2TextGenerationPipeline
import torch

tokenizer=AutoTokenizer.from_pretrained('IDEA-CCNL/Randeng-BART-139M', use_fast=false)
model=BartForConditionalGeneration.from_pretrained('IDEA-CCNL/Randeng-BART-139M')
text = '桂林市是世界闻名<mask> ，它有悠久的<mask>'
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
print(text2text_generator(text, max_length=50, do_sample=False))
