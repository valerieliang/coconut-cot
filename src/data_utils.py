import json
import pandas as pd
from sklearn.model_selection import train_test_split

def parse_example(ex):
    steps = ex["steps"]
    return {
        "question": ex["question"],
        "answer":   ex["answer"],
        "steps":    steps,
        "n_hops":   len(steps),
        "root":     ex["root"],
        "target":   ex["target"],
    }

def load_and_split(
    json_path="coconut/data/prosqa_test.json",
    save_path="data/prontoqa_split.csv"
):
    with open(json_path) as f:
        raw = json.load(f)

    rows = [parse_example(ex) for ex in raw]
    df   = pd.DataFrame(rows)

    print("Hop distribution in raw data:")
    print(df.n_hops.value_counts().sort_index())

    # No n_hops=2 in this file -- use 3,4,5 only
    df = df[df.n_hops.isin([3, 4, 5])]

    # Sample per hop level, keeping n_hops column intact
    sampled = pd.concat([
        grp.sample(min(len(grp), 100), random_state=42)
        for _, grp in df.groupby("n_hops")
    ]).reset_index(drop=True)

    print("\nAfter sampling:")
    print(sampled.n_hops.value_counts().sort_index())

    train, test = train_test_split(
        sampled, test_size=0.2,
        stratify=sampled["n_hops"],
        random_state=42
    )
    train = train.copy(); train["split"] = "train"
    test  = test.copy();  test["split"]  = "test"

    result = pd.concat([train, test]).reset_index(drop=True)
    result["steps"] = result["steps"].apply(json.dumps)

    import os; os.makedirs("data", exist_ok=True)
    result.to_csv(save_path, index=False)
    print(f"\nSaved {len(result)} rows -> {save_path}")
    print(result.groupby(["n_hops", "split"]).size())
    return result