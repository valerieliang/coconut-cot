# src/data_utils.py
"""
Data loading utilities for ProntoQA, ProsQA, and other reasoning datasets.
"""

import pandas as pd
import json
import os
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset
from sklearn.model_selection import train_test_split


def parse_example(ex: Dict) -> Dict:
    """Parse a raw example from ProsQA/ProntoQA JSON."""
    steps = ex.get("steps", [])
    return {
        "question": ex.get("question", ""),
        "answer": ex.get("answer", ""),
        "steps": steps,
        "n_hops": len(steps),
        "root": ex.get("root", ""),
        "target": ex.get("target", ""),
    }


def load_and_split_prosqa(
    json_path: str = "coconut/data/prosqa_test.json",
    save_path: str = "data/prontoqa_split.csv",
    hop_filter: List[int] = None,
    samples_per_hop: int = 100,
    test_size: float = 0.2,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load ProsQA JSON, sample balanced data, and save as CSV.
    
    Args:
        json_path: Path to ProsQA JSON file
        save_path: Path to save the split CSV
        hop_filter: List of hop counts to include (default: [3, 4, 5])
        samples_per_hop: Maximum samples per hop count
        test_size: Proportion of data for test set
        random_state: Random seed for reproducibility
    
    Returns:
        DataFrame with train/test split
    """
    if hop_filter is None:
        hop_filter = [3, 4, 5]
    
    with open(json_path) as f:
        raw = json.load(f)
    
    rows = [parse_example(ex) for ex in raw]
    df = pd.DataFrame(rows)
    
    print("Hop distribution in raw data:")
    print(df.n_hops.value_counts().sort_index())
    
    # Filter by hop count
    df = df[df.n_hops.isin(hop_filter)]
    
    # Sample per hop level
    sampled_dfs = []
    for n_hops, grp in df.groupby("n_hops"):
        n_samples = min(len(grp), samples_per_hop)
        sampled_dfs.append(grp.sample(n_samples, random_state=random_state))
    
    sampled = pd.concat(sampled_dfs).reset_index(drop=True)
    
    print("\nAfter sampling:")
    print(sampled.n_hops.value_counts().sort_index())
    
    # Split into train/test
    train, test = train_test_split(
        sampled, 
        test_size=test_size,
        stratify=sampled["n_hops"],
        random_state=random_state
    )
    train = train.copy()
    train["split"] = "train"
    test = test.copy()
    test["split"] = "test"
    
    result = pd.concat([train, test]).reset_index(drop=True)
    
    # Convert steps list to JSON string for CSV storage
    result["steps"] = result["steps"].apply(json.dumps)
    
    # Create directory and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result.to_csv(save_path, index=False)
    
    print(f"\nSaved {len(result)} rows -> {save_path}")
    print(result.groupby(["n_hops", "split"]).size())
    
    return result


def load_prontoqa(
    filepath: str, 
    split: Optional[str] = None
) -> pd.DataFrame:
    """
    Load ProntoQA dataset from CSV file.
    
    Args:
        filepath: Path to CSV file
        split: Which split to load ('train', 'valid', 'test', or None for all)
    
    Returns:
        DataFrame with columns: question, steps, answer, n_hops, split
    """
    df = pd.read_csv(filepath)
    
    # Parse steps from JSON string back to list
    if 'steps' in df.columns:
        df['steps'] = df['steps'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    # Add n_hops if not present
    if 'n_hops' not in df.columns:
        df['n_hops'] = df['steps'].apply(len)
    
    if split is not None and 'split' in df.columns:
        df = df[df['split'] == split].reset_index(drop=True)
    
    return df


def load_prosqa_raw(
    filepath: str, 
    split: Optional[str] = None
) -> pd.DataFrame:
    """
    Load ProsQA dataset directly from JSON.
    
    Args:
        filepath: Path to ProsQA JSON file
        split: Which split to load (ProsQA may have split in filename or data)
    
    Returns:
        DataFrame with parsed examples
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        rows = [parse_example(ex) for ex in data]
    elif isinstance(data, dict):
        if split and split in data:
            rows = [parse_example(ex) for ex in data[split]]
        else:
            # Flatten all splits
            rows = []
            for split_name, examples in data.items():
                for ex in examples:
                    parsed = parse_example(ex)
                    parsed['split'] = split_name
                    rows.append(parsed)
    else:
        raise ValueError(f"Unexpected JSON structure: {type(data)}")
    
    df = pd.DataFrame(rows)
    
    if split and 'split' in df.columns:
        df = df[df['split'] == split].reset_index(drop=True)
    
    return df


def extract_concept_from_step(step: str) -> Optional[str]:
    """
    Extract the derived concept from a reasoning step.
    
    Examples:
        "Alex is a reptile" -> "reptile"
        "Reptiles are cold-blooded" -> "cold-blooded"
        "Alex has scales" -> "scales"
    
    Args:
        step: Reasoning step text
    
    Returns:
        Extracted concept or None
    """
    import re
    
    # Patterns for concept extraction
    patterns = [
        r'is\s+(?:a\s+)?(\w+(?:\s+\w+)?)',  # "is a reptile", "is cold-blooded"
        r'are\s+(?:a\s+)?(\w+(?:\s+\w+)?)', # "are cold-blooded"
        r'has\s+(?:a\s+)?(\w+(?:\s+\w+)?)', # "has scales"
        r'have\s+(?:a\s+)?(\w+(?:\s+\w+)?)', # "have scales"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, step, re.IGNORECASE)
        if match:
            concept = match.group(1).strip().lower()
            # Clean up trailing punctuation
            concept = re.sub(r'[^\w\s]', '', concept)
            return concept
    
    return None


def extract_premise_and_negation(
    problem_row: Dict, 
    step_idx: int = 0
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract a premise and its logical negation from a problem step.
    
    Args:
        problem_row: Dictionary with 'steps' key
        step_idx: Index of step to extract from
    
    Returns:
        Tuple of (premise, negated_premise) or (None, None)
    """
    import re
    
    steps = problem_row.get("steps", [])
    if step_idx >= len(steps):
        return None, None
    
    step = steps[step_idx]
    
    # Pattern 1: "X is a Y" or "X is Y"
    match = re.search(r'(\w+)\s+is\s+(?:a\s+)?(\w+(?:\s+\w+)?)', step, re.IGNORECASE)
    if match:
        subject = match.group(1)
        predicate = match.group(2).rstrip('.')
        premise = f"{subject} is {predicate}."
        negated = f"{subject} is not {predicate}."
        return premise, negated
    
    # Pattern 2: "X are Y"
    match = re.search(r'(\w+)\s+are\s+(?:a\s+)?(\w+(?:\s+\w+)?)', step, re.IGNORECASE)
    if match:
        subject = match.group(1)
        predicate = match.group(2).rstrip('.')
        premise = f"{subject} are {predicate}."
        negated = f"{subject} are not {predicate}."
        return premise, negated
    
    # Pattern 3: "X has Y" or "X have Y"
    match = re.search(r'(\w+)\s+ha(?:s|ve)\s+(?:a\s+)?(\w+(?:\s+\w+)?)', step, re.IGNORECASE)
    if match:
        subject = match.group(1)
        predicate = match.group(2).rstrip('.')
        verb = "has" if match.group(0).startswith(f"{subject} has") else "have"
        premise = f"{subject} {verb} {predicate}."
        negated = f"{subject} does not have {predicate}."
        return premise, negated
    
    # Fallback: use the whole step and insert "not"
    premise = step.strip()
    if "not" in premise.lower():
        negated = premise.replace("not ", "").replace("n't ", "")
    else:
        # Simple negation by adding "not" after first verb
        words = premise.split()
        for i, word in enumerate(words):
            if word.lower() in ["is", "are", "has", "have", "was", "were"]:
                words.insert(i + 1, "not")
                break
        negated = " ".join(words)
    
    return premise, negated


def get_problem_statistics(df: pd.DataFrame) -> Dict:
    """
    Compute statistics about the dataset.
    
    Args:
        df: DataFrame with 'steps' and 'n_hops' columns
    
    Returns:
        Dictionary with dataset statistics
    """
    from collections import Counter
    
    stats = {
        'n_problems': len(df),
        'avg_hops': df['n_hops'].mean() if 'n_hops' in df.columns else None,
        'std_hops': df['n_hops'].std() if 'n_hops' in df.columns else None,
        'hop_distribution': df['n_hops'].value_counts().sort_index().to_dict() if 'n_hops' in df.columns else {},
    }
    
    # Split distribution if available
    if 'split' in df.columns:
        stats['split_distribution'] = df['split'].value_counts().to_dict()
        stats['hop_by_split'] = df.groupby(['split', 'n_hops']).size().unstack(fill_value=0).to_dict()
    
    # Extract all concepts
    if 'steps' in df.columns:
        all_concepts = []
        for steps in df['steps']:
            for step in steps:
                concept = extract_concept_from_step(step)
                if concept:
                    all_concepts.append(concept)
        
        concept_counter = Counter(all_concepts)
        stats['n_unique_concepts'] = len(concept_counter)
        stats['concept_frequencies'] = dict(concept_counter.most_common(20))
        stats['total_concept_occurrences'] = len(all_concepts)
    
    # Answer distribution
    if 'answer' in df.columns:
        stats['answer_distribution'] = df['answer'].value_counts().to_dict()
    
    return stats


def format_cot(
    df: pd.DataFrame, 
    tokenizer,
    max_length: int = 512,
    include_answer: bool = True
) -> Dataset:
    """
    Format DataFrame for verbal CoT training/evaluation.
    
    Format: question + "\n" + steps + "\n### " + answer + eos
    
    Args:
        df: DataFrame with 'question', 'steps', 'answer' columns
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        include_answer: Whether to include answer in the text
    
    Returns:
        HuggingFace Dataset with input_ids and labels
    """
    texts = []
    for _, row in df.iterrows():
        steps_text = "\n".join(row["steps"])
        if include_answer:
            text = row["question"] + "\n" + steps_text + "\n### " + row["answer"]
        else:
            text = row["question"] + "\n" + steps_text
        texts.append(text)
    
    enc = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
    
    return Dataset.from_dict(enc)


def format_coconut(
    df: pd.DataFrame,
    tokenizer,
    latent_id: int,
    start_id: int,
    end_id: int,
    max_length: int = 512
) -> Dataset:
    """
    Format DataFrame for Coconut training.
    
    Format: question + "\n" + <start> + <latent>*n_hops + <end>
    
    Args:
        df: DataFrame with 'question', 'n_hops' columns
        tokenizer: HuggingFace tokenizer
        latent_id: Token ID for <latent>
        start_id: Token ID for <start>
        end_id: Token ID for <end>
        max_length: Maximum sequence length
    
    Returns:
        HuggingFace Dataset with input_ids
    """
    input_ids_list = []
    
    for _, row in df.iterrows():
        question_ids = tokenizer.encode(row["question"] + "\n", add_special_tokens=True)
        n_hops = row["n_hops"]
        
        input_ids = question_ids + [start_id] + [latent_id] * n_hops + [end_id]
        
        # Pad or truncate
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        else:
            input_ids = input_ids + [tokenizer.pad_token_id] * (max_length - len(input_ids))
        
        input_ids_list.append(input_ids)
    
    return Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": [[1 if id != tokenizer.pad_token_id else 0 for id in ids] for ids in input_ids_list],
    })


def get_problems_by_hop(
    df: pd.DataFrame, 
    n_hops: int
) -> pd.DataFrame:
    """
    Filter DataFrame to problems with specific hop count.
    """
    return df[df['n_hops'] == n_hops].reset_index(drop=True)


def get_problem_by_idx(
    df: pd.DataFrame, 
    idx: int
) -> Dict:
    """
    Get a single problem as a dictionary by index.
    """
    if idx >= len(df):
        raise IndexError(f"Index {idx} out of range for DataFrame of length {len(df)}")
    
    row = df.iloc[idx]
    return row.to_dict()


def create_train_val_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    stratify_by: str = "n_hops",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/validation split from a DataFrame.
    
    Args:
        df: Input DataFrame
        val_size: Proportion for validation
        stratify_by: Column to stratify by
        random_state: Random seed
    
    Returns:
        Tuple of (train_df, val_df)
    """
    from sklearn.model_selection import train_test_split
    
    if stratify_by in df.columns:
        stratify = df[stratify_by]
    else:
        stratify = None
    
    train, val = train_test_split(
        df,
        test_size=val_size,
        stratify=stratify,
        random_state=random_state
    )
    
    return train.reset_index(drop=True), val.reset_index(drop=True)


# Legacy alias for backward compatibility
def load_split(path: str, split: Optional[str] = None) -> pd.DataFrame:
    """Legacy function - use load_prontoqa instead."""
    return load_prontoqa(path, split)