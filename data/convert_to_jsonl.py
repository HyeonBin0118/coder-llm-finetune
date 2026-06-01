"""
dataset_raw.json → 학습 포맷(jsonl) 변환
Qwen2.5-Coder Instruct 포맷으로 변환
결과: data/train.jsonl, data/val.jsonl
"""
import json
import random
from pathlib import Path

INPUT_PATH  = Path("data/dataset_raw.json")
TRAIN_PATH  = Path("data/train.jsonl")
VAL_PATH    = Path("data/val.jsonl")

SYSTEM_PROMPT = """당신은 프로그래머스 코딩 테스트 문제를 도와주는 어시스턴트입니다.
문제를 분석하고 힌트, 접근법, 정답 코드를 단계별로 제공합니다."""


def make_hint_sample(problem: dict) -> dict:
    user = f"""다음 프로그래머스 문제의 힌트를 알려주세요.

제목: {problem['title']}
난이도: Level {problem['level']}
문제 설명: {problem['description'][:500]}
제한사항: {problem.get('constraints', '')[:200]}"""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": problem["hint"]},
        ]
    }


def make_approach_sample(problem: dict) -> dict:
    user = f"""다음 프로그래머스 문제의 풀이 접근법을 알려주세요.

제목: {problem['title']}
난이도: Level {problem['level']}
문제 설명: {problem['description'][:500]}
제한사항: {problem.get('constraints', '')[:200]}"""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": problem["approach"]},
        ]
    }


def make_solution_sample(problem: dict) -> dict:
    user = f"""다음 프로그래머스 문제의 Python 정답 코드를 작성해주세요.

제목: {problem['title']}
난이도: Level {problem['level']}
문제 설명: {problem['description'][:500]}
제한사항: {problem.get('constraints', '')[:200]}"""

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": problem["solution"]},
        ]
    }


def main():
    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    print(f"총 {len(data)}개 문제 로드")

    samples = []
    for problem in data:
        samples.append(make_hint_sample(problem))
        samples.append(make_approach_sample(problem))
        samples.append(make_solution_sample(problem))

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
    print(f"총 샘플: {len(samples)}개 (문제당 3개 × {len(data)}문제)")


if __name__ == "__main__":
    main()