"""
dataset_v2.json → JSONL 변환
train 90% / val 10% 분리
출력: data/train.jsonl, data/val.jsonl
"""
import json
import random
from pathlib import Path

INPUT_PATH = Path("data/dataset_v2.json")
TRAIN_PATH = Path("data/train.jsonl")
VAL_PATH   = Path("data/val.jsonl")


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    print(f"총 {len(data)}개 샘플 로드")

    # messages 필드만 추출
    samples = [{"messages": item["messages"]} for item in data]

    random.seed(42)
    random.shuffle(samples)

    split = int(len(samples) * 0.9)
    train_samples = samples[:split]
    val_samples   = samples[split:]

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        for s in train_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(VAL_PATH, "w", encoding="utf-8") as f:
        for s in val_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"train: {len(train_samples)}개 → {TRAIN_PATH}")
    print(f"val:   {len(val_samples)}개 → {VAL_PATH}")


if __name__ == "__main__":
    main()