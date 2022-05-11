from transformers import BartForConditionalGeneration, AutoTokenizer, Text2TextGenerationPipeline
import datasets
import argparse
import torch
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument('--result_dir', type=str, default="")
    parser.add_argument('--input_length', type=int, default=128)
    parser.add_argument('--ouput_length', type=int, default=40)
    
    args = parser.parse_args()
    return args
    
args = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use %s as device" % device)

tokenizer_pretrain_dir = "IDEA-CCNL/Randeng-BART-139M"

tokenizer=AutoTokenizer.from_pretrained(tokenizer_pretrain_dir, use_fast=True)

test_frame = pd.read_csv(args.data_dir, sep="\t", header=0, error_bad_lines=False)
test_frame = test_frame.dropna()
test_dataset = datasets.Dataset.from_pandas(test_frame)

def kv2text_tokenize(element):
    input_encodings = tokenizer(
        element["input_str"],
        truncation = True,
        max_length=args.input_length
    )
    
    target_encodings = tokenizer(
        element["target"],
        truncation = True,
        max_length=args.out_length
    )
    
    labels = target_encodings['input_ids']
    
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': labels
    }
    
    return encodings

tokenized_datasets = test_dataset.map(
    kv2text_tokenize, batched=True, remove_columns=test_dataset.column_names
)

print(tokenized_datasets)

tokenized_datasets.set_format('torch')

model = BartForConditionalGeneration.from_pretrained(tokenizer_pretrain_dir)
model.to(device)

if device.type == 'cuda'
    device_id = 0
else:
    device_id = -1
    
text2text_generator = Text2TextGenerationPipeline(model, tokenizer, device=device_id)

def inference(element, max_length=args.out_length, do_sample=False):
    return {'bart_text': [k['generated_text'] for k in 
                         text2text_generator(element["input_str"], max_length=max_length, do_sample=do_sample]}

result_dataset = test_dataset.map(inference, batched=True, batch_size=32)

result_dataset[['item_id', 'input_str', 'target', 'bart_text']].to_csv(test_result_dir+f"generated_test.csv", sep='\t', index=False)
    

    
