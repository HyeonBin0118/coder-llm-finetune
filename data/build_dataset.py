"""
B안 데이터셋 구성
- 힌트/접근법: dataset_raw.json (GPT 생성)
- 정답 코드: github_solutions.json (실제 통과 코드)
- 출력: data/dataset_v2.json
"""
import json
import re

RAW_PATH = "data/dataset_raw.json"
GITHUB_PATH = "data/github_solutions.json"
OUTPUT_PATH = "data/dataset_v2.json"

SYSTEM_PROMPT = "당신은 프로그래머스 코딩 테스트 문제를 도와주는 어시스턴트입니다. 문제를 분석하고 힌트, 접근법, 정답 코드를 단계별로 제공합니다."


def make_messages(user_content: str, assistant_content: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content}
    ]


def main():
    with open(RAW_PATH, encoding="utf-8") as f:
        raw_data = json.load(f)

    with open(GITHUB_PATH, encoding="utf-8") as f:
        github_data = json.load(f)

    github_map = {r["title"]: r["solutions"] for r in github_data}
    raw_map = {r["title"]: r for r in raw_data}

    dataset = []
    stats = {"hint": 0, "approach": 0, "solution": 0, "skipped": 0}

    for item in github_data:
        title = item["title"]
        level = item["level"]
        description = item["description"][:800]
        solutions = item["solutions"]
        raw = raw_map.get(title)

        # 힌트
        if raw and raw.get("hint"):
            dataset.append({
                "title": title,
                "level": level,
                "task": "hint",
                "messages": make_messages(
                    f"다음 프로그래머스 문제의 힌트를 알려주세요.\n\n제목: {title}\n난이도: Level {level}\n문제 설명:\n{description}",
                    raw["hint"]
                )
            })
            stats["hint"] += 1

        # 접근법
        if raw and raw.get("approach"):
            dataset.append({
                "title": title,
                "level": level,
                "task": "approach",
                "messages": make_messages(
                    f"다음 프로그래머스 문제의 접근법을 설명해주세요.\n\n제목: {title}\n난이도: Level {level}\n문제 설명:\n{description}",
                    raw["approach"]
                )
            })
            stats["approach"] += 1

        # 정답 코드 (GitHub 풀이 전부 활용)
        if solutions:
            for code in solutions:
                sig_match = re.search(r'def solution\([^)]*\)', code)
                sig = sig_match.group(0) if sig_match else "def solution(...)"

                dataset.append({
                    "title": title,
                    "level": level,
                    "task": "solution",
                    "messages": make_messages(
                        f"다음 프로그래머스 문제의 Python 정답 코드를 작성해주세요.\n\n제목: {title}\n난이도: Level {level}\n함수 시그니처: {sig}\n문제 설명:\n{description}",
                        code
                    )
                })
                stats["solution"] += 1
        else:
            stats["skipped"] += 1

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"데이터셋 생성 완료!")
    print(f"  힌트:     {stats['hint']}개")
    print(f"  접근법:   {stats['approach']}개")
    print(f"  정답코드: {stats['solution']}개")
    print(f"  풀이 없어 스킵: {stats['skipped']}개")
    print(f"  총합: {len(dataset)}개")
    print(f"  저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()