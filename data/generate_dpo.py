"""
DPO 학습용 데이터 생성
- chosen: 깔끔하고 효율적인 풀이
- rejected: 동작은 하지만 비효율적인 풀이
- 입력: data/dataset_raw.json
- 출력: data/dataset_dpo.json
"""
import json
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

INPUT_PATH  = Path("data/dataset_raw.json")
OUTPUT_PATH = Path("data/dataset_dpo.json")

SYSTEM_PROMPT = "당신은 프로그래머스 코딩 테스트 문제를 도와주는 어시스턴트입니다."

CHOSEN_PROMPT = """다음 프로그래머스 문제의 Python 정답 코드를 작성해줘.
- 가장 깔끔하고 Pythonic한 풀이로
- 시간복잡도 최적화
- solution 함수만
- 주석 없이 코드만
- ```python 없이 순수 코드만

제목: {title}
설명: {description}
제한사항: {constraints}
함수 시그니처: {signature}"""

REJECTED_PROMPT = """다음 프로그래머스 문제의 Python 풀이 코드를 작성해줘.
- 동작은 하지만 비효율적인 방식으로 (반복문 중첩, 불필요한 변수, 비효율적 자료구조 등)
- 정답은 맞아야 함
- solution 함수만
- 주석 없이 코드만
- ```python 없이 순수 코드만

제목: {title}
설명: {description}
제한사항: {constraints}
함수 시그니처: {signature}"""


def call_gpt(prompt: str, temperature: float = 0.3) -> str:
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


def generate_one(problem: dict) -> dict | None:
    title       = problem["title"]
    level       = problem["level"]
    description = problem["description"][:600]
    constraints = problem.get("constraints", "")
    solution    = problem.get("solution", "")
    signature   = extract_sig(solution)

    user_content = (
        f"다음 프로그래머스 문제의 Python 정답 코드를 작성해주세요.\n\n"
        f"제목: {title}\n난이도: Level {level}\n"
        f"함수 시그니처: {signature}\n문제 설명:\n{description}"
    )

    try:
        chosen = call_gpt(CHOSEN_PROMPT.format(
            title=title, description=description,
            constraints=constraints, signature=signature
        ), temperature=0.3)
        time.sleep(0.5)

        rejected = call_gpt(REJECTED_PROMPT.format(
            title=title, description=description,
            constraints=constraints, signature=signature
        ), temperature=0.7)
        time.sleep(0.5)

        return {
            "title": title,
            "level": level,
            "chosen": make_messages(user_content, chosen),
            "rejected": make_messages(user_content, rejected),
        }

    except Exception as e:
        print(f"  오류: {e}")
        return None


def main():
    raw_data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    print(f"총 {len(raw_data)}개 문제 DPO 데이터 생성 시작\n")

    # 이어서 실행 가능
    if OUTPUT_PATH.exists():
        results = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        done_titles = {r["title"] for r in results}
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
    print(f"\n완료: {len(results)}개 DPO 쌍 생성 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()