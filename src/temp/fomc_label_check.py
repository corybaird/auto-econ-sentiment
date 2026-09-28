"""Check which FOMC-RoBERTa output index means hawkish, dovish and neutral using human-labelled sentences."""
import glob
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

files = glob.glob("/home/corybaird/.cache/huggingface/hub/datasets--gtfintechlab--all_annotated_sentences_25000/snapshots/*/*/test-*.parquet")
data = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(data.stance_label.value_counts())
sample = data.groupby("stance_label", group_keys=False).apply(lambda g: g.sample(min(len(g), 1000), random_state=0))
tok = AutoTokenizer.from_pretrained("gtfintechlab/FOMC-RoBERTa")
model = AutoModelForSequenceClassification.from_pretrained("gtfintechlab/FOMC-RoBERTa").to("cuda").eval()
preds = []
with torch.no_grad():
    for i in range(0, len(sample), 64):
        batch = tok(sample.sentences.iloc[i:i + 64].tolist(), truncation=True, max_length=256, padding=True, return_tensors="pt").to("cuda")
        preds += model(**batch).logits.argmax(-1).tolist()
sample["pred"] = [f"LABEL_{p}" for p in preds]
table = pd.crosstab(sample.stance_label, sample.pred)
print(table)
print((table.div(table.sum(axis=1), axis=0) * 100).round(1))
table.to_csv("src/temp/fomc_label_crosstab.csv")
