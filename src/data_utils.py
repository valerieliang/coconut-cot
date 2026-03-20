from datasets import load_dataset, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_prontoqa_example(ex):
    return {
        "context":  ex["facts"] + " " + ex["rules"],
        "query":    ex["question"],
        "chain":    ex["proof_steps"],   # adjust key after inspecting the dataset
        "n_hops":   len(ex["proof_steps"]),
        "answer":   ex["answer"] == "True",
    }

def load_and_split(save_path="data/prontoqa_split.csv"):
    raw = load_dataset("kaiokendev/ProntoQA")  # inspect keys before assuming
    df  = pd.DataFrame([parse_prontoqa_example(ex) for ex in raw["test"]])
    df  = df[df.n_hops.isin([2,3,4,5])].groupby("n_hops").sample(100, random_state=42)
    train, test = train_test_split(df, test_size=0.2, stratify=df.n_hops, random_state=42)
    train["split"] = "train"; test["split"] = "test"
    pd.concat([train, test]).to_csv(save_path, index=False)

def load_split(path, split=None):
    df = pd.read_csv(path)
    # chain was serialized as string — deserialize
    df["chain"] = df["chain"].apply(eval)
    return df if split is None else df[df.split == split]

def format_cot(df, tokenizer):
    # Format each row as: context + query + step1 + step2 + ... + answer
    # Returns a HuggingFace Dataset ready for the Trainer
    texts = []
    for _, row in df.iterrows():
        steps = " ".join(s["text"] for s in row["chain"])
        texts.append(row["context"] + " " + row["query"] + " " + steps + " " + str(row["answer"]))
    enc = tokenizer(texts, truncation=True, padding="max_length", max_length=256)
    enc["labels"] = enc["input_ids"].copy()
    return Dataset.from_dict(enc)