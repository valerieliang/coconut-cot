import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_utils import load_and_split

load_and_split(
    json_path="coconut/data/prosqa_test.json",
    save_path="data/prontoqa_split.csv"
)