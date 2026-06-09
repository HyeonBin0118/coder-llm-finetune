"""
기존 dataset_v2.json + dataset_evol.json 병합
→ train.jsonl / val.jsonl 재생성
"""
import json
import random
from pathlib import Path

V2_PATH = Path("data/dataset_v2.json")
EVOL_PATH = Path("data/dataset_evol.json")
TRAIN_PATH = Path("data/train.jsonl")
VAL_PATH = Path("data/val.jsonl")

SPLIT_RATIO = 0.9
SEED = 42

def main():
    v2 = json.loads(V2_PATH.read_text(encoding="utf-8"))
    evol = json.loads(EVOL_PATH.read_text(encoding="utf-8"))

    combined = v2 + evol
    print(f"v2:   {len(v2)}개")
    print(f"evol: {len(evol)}개")
    print(f"합계: {len(combined)}개")

    random.seed(SEED)
    random.shuffle(combined)

    split = int(len(combined) * SPLIT_RATIO)
    train = combined[:split]
    val = combined[split:]

    TRAIN_PATH.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in train),
        encoding="utf-8"
    )
    VAL_PATH.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in val),
        encoding="utf-8"
    )

    print(f"\ntrain: {len(train)}개 → {TRAIN_PATH}")
    print(f"val:   {len(val)}개 → {VAL_PATH}")

if __name__ == "__main__":
    main()