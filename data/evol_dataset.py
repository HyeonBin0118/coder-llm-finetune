"""
Evol-Instruct 방식으로 기존 문제를 변형하여 데이터 확장
- 입력: data/dataset_raw.json (기존 GPT 생성 데이터)
- 출력: data/dataset_evol.json (변형 샘플 추가)
"""
import json
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

INPUT_PATH = Path("data/dataset_raw.json")
OUTPUT_PATH = Path("data/dataset_evol.json")

# 변형 타입 3가지
EVOL_PROMPTS = {
    "constraint": """다음 프로그래머스 문제에 새로운 제약 조건을 추가한 변형 문제를 만들어줘.
- 원본 문제에 "반복문 사용 금지" 또는 "내장 함수 사용 금지" 같은 제약 1개 추가
- 변형된 문제 설명과 그 풀이 힌트를 2~3줄로 작성
- 형식: 변형문제설명 / 힌트 (줄바꿈으로 구분)
- 코드 없이 텍스트만

원본 제목: {title}
원본 설명: {description}""",

    "scale": """다음 프로그래머스 문제의 입력 규모를 대폭 키운 변형 문제를 만들어줘.
- 입력 크기를 10배~100배로 키우고 시간/공간 복잡도 조건 추가
- 변형된 문제 설명과 최적화 힌트를 2~3줄로 작성
- 형식: 변형문제설명 / 힌트 (줄바꿈으로 구분)
- 코드 없이 텍스트만

원본 제목: {title}
원본 설명: {description}""",

    "recursive": """다음 프로그래머스 문제를 반드시 재귀(recursion)로 풀어야 하는 변형 문제를 만들어줘.
- "재귀 함수만 사용" 조건 추가
- 변형된 문제 설명과 재귀 접근 힌트를 2~3줄로 작성
- 형식: 변형문제설명 / 힌트 (줄바꿈으로 구분)
- 코드 없이 텍스트만

원본 제목: {title}
원본 설명: {description}""",
}

SOLUTION_PROMPT = """다음 변형된 코딩 문제의 Python 정답 코드를 작성해줘.
- solution 함수만
- 주석 없이 코드만
- ```python 코드블록 없이 순수 코드만

원본 제목: {title}
변형 조건: {evol_type}
변형 문제 설명: {evol_description}
원본 함수 시그니처: {signature}"""

EVOL_TYPE_KO = {
    "constraint": "제약 추가",
    "scale": "규모 확장",
    "recursive": "재귀 변환",
}

SYSTEM_PROMPT = "당신은 프로그래머스 코딩 테스트 문제를 도와주는 어시스턴트입니다. 문제를 분석하고 힌트, 접근법, 정답 코드를 단계별로 제공합니다."


def call_gpt(prompt: str, temperature: float = 0.7) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def extract_sig(solution_code: str) -> str:
    import re
    m = re.search(r'def solution\([^)]*\)', solution_code)
    return m.group(0) if m else "def solution(...)"


def make_messages(user_content: str, assistant_content: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def evolve_one(problem: dict) -> list[dict]:
    title = problem["title"]
    level = problem["level"]
    description = problem["description"][:600]
    solution_code = problem.get("solution", "")
    signature = extract_sig(solution_code)
    results = []

    for evol_type, prompt_template in EVOL_PROMPTS.items():
        try:
            # 1. 변형 문제 + 힌트 생성
            evol_raw = call_gpt(
                prompt_template.format(title=title, description=description),
                temperature=0.7
            )
            time.sleep(0.5)

            # 변형 설명 / 힌트 분리
            parts = evol_raw.split("/", 1)
            evol_description = parts[0].strip()
            evol_hint = parts[1].strip() if len(parts) > 1 else evol_raw.strip()

            # 2. 변형 문제 풀이 코드 생성
            evol_solution = call_gpt(
                SOLUTION_PROMPT.format(
                    title=title,
                    evol_type=EVOL_TYPE_KO[evol_type],
                    evol_description=evol_description,
                    signature=signature,
                ),
                temperature=0.3
            )
            time.sleep(0.5)

            evol_title = f"{title} [{EVOL_TYPE_KO[evol_type]}]"

            # hint 샘플
            results.append({
                "title": evol_title,
                "level": level,
                "task": "hint",
                "evol_type": evol_type,
                "messages": make_messages(
                    f"다음 프로그래머스 문제의 힌트를 알려주세요.\n\n제목: {evol_title}\n난이도: Level {level}\n문제 설명:\n{evol_description}",
                    evol_hint
                )
            })

            # solution 샘플
            results.append({
                "title": evol_title,
                "level": level,
                "task": "solution",
                "evol_type": evol_type,
                "messages": make_messages(
                    f"다음 프로그래머스 문제의 Python 정답 코드를 작성해주세요.\n\n제목: {evol_title}\n난이도: Level {level}\n함수 시그니처: {signature}\n문제 설명:\n{evol_description}",
                    evol_solution
                )
            })

            print(f"  ✓ {EVOL_TYPE_KO[evol_type]}")

        except Exception as e:
            print(f"  ✗ {evol_type} 실패: {e}")
            continue

    return results


def main():
    raw_data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    print(f"원본 문제 수: {len(raw_data)}개\n")

    # 이어서 실행 가능하도록
    if OUTPUT_PATH.exists():
        results = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        done_titles = {r["title"].split(" [")[0] for r in results}
        print(f"기존 결과 {len(results)}개 로드, 이어서 진행\n")
    else:
        results = []
        done_titles = set()

    for i, problem in enumerate(raw_data):
        title = problem["title"]

        if title in done_titles:
            print(f"[{i+1}/{len(raw_data)}] 스킵: {title}")
            continue

        print(f"[{i+1}/{len(raw_data)}] {title}")
        evol_samples = evolve_one(problem)
        results.extend(evol_samples)
        done_titles.add(title)

        # 10문제마다 중간 저장
        if (i + 1) % 10 == 0:
            OUTPUT_PATH.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"  → 중간 저장 ({len(results)}개)")

        time.sleep(0.3)

    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n완료: {len(results)}개 변형 샘플 생성 → {OUTPUT_PATH}")
    print(f"원본 681개 + 변형 {len(results)}개 = 총 {681 + len(results)}개")


if __name__ == "__main__":
    main()