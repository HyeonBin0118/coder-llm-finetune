"""
GPT-4o-mini로 힌트/접근법/정답 생성
결과: data/dataset_raw.json
"""
import json
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

INPUT_PATH  = Path("data/problems_parsed.json")
OUTPUT_PATH = Path("data/dataset_raw.json")

HINT_PROMPT = """다음 프로그래머스 코딩 문제에 대해 핵심 힌트를 2~3줄로 알려줘.
- 알고리즘 이름은 직접 말하지 말고 접근 방향만 알려줘
- 코드 없이 텍스트로만

제목: {title}
설명: {description}
제한사항: {constraints}"""

APPROACH_PROMPT = """다음 프로그래머스 코딩 문제의 풀이 접근법을 단계별로 설명해줘.
- 번호 목록으로 4~6단계
- 시간복잡도 언급
- 코드 없이 텍스트로만

제목: {title}
설명: {description}
제한사항: {constraints}"""

SOLUTION_PROMPT = """다음 프로그래머스 코딩 문제의 Python 정답 코드를 작성해줘.
- solution 함수만
- 주석 없이 코드만
- ```python 코드블록 없이 순수 코드만

제목: {title}
설명: {description}
제한사항: {constraints}
입출력 예: {examples}"""


def call_gpt(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def generate_one(problem: dict) -> dict | None:
    title       = problem["title"]
    description = problem["description"]
    constraints = problem.get("constraints", "")
    examples    = problem.get("examples", [])

    try:
        hint = call_gpt(HINT_PROMPT.format(
            title=title, description=description, constraints=constraints))
        time.sleep(0.5)

        approach = call_gpt(APPROACH_PROMPT.format(
            title=title, description=description, constraints=constraints))
        time.sleep(0.5)

        solution = call_gpt(SOLUTION_PROMPT.format(
            title=title, description=description,
            constraints=constraints, examples=examples))
        time.sleep(0.5)

        return {
            "title":       title,
            "level":       problem["level"],
            "url":         problem["url"],
            "description": description,
            "constraints": constraints,
            "hint":        hint,
            "approach":    approach,
            "solution":    solution,
        }
    except Exception as e:
        print(f"  오류: {e}")
        return None


def main():
    problems = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    print(f"총 {len(problems)}개 문제 데이터 생성 시작\n")

    # 이미 생성된 결과 로드 (중간에 끊겨도 이어서 가능)
    if OUTPUT_PATH.exists():
        results = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        done_titles = {r["title"] for r in results}
        print(f"기존 결과 {len(results)}개 로드, 이어서 진행\n")
    else:
        results = []
        done_titles = set()
    ##for i, problem in enumerate(problems[:10]): # 테스트용으로 처음 10개만 생성
    for i, problem in enumerate(problems):# 전체 문제 생성
        title = problem["title"]
        if title in done_titles:
            print(f"[{i+1}/{len(problems)}] 스킵 (이미 생성됨): {title}")
            continue

        print(f"[{i+1}/{len(problems)}] {title}")
        result = generate_one(problem)

        if result:
            results.append(result)
            done_titles.add(title)
            print(f"  ✓ 완료")
        else:
            print(f"  ✗ 실패")

        # 10개마다 중간 저장
        if len(results) % 10 == 0:
            OUTPUT_PATH.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

        time.sleep(0.3)

    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n완료: {len(results)}개 생성 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()