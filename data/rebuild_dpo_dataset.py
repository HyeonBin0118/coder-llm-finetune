"""
DPO 데이터 재구성
chosen: GitHub 실제 통과 코드 (검증됨)
rejected: 기존 dataset_dpo.json의 rejected 코드 유지
solutions가 없는 문제는 제외
"""
import json
from pathlib import Path

DPO_PATH = "data/dataset_dpo.json"
GITHUB_PATH = "data/github_solutions.json"
OUTPUT_PATH = "data/dataset_dpo_v2.json"

SYSTEM_PROMPT = "당신은 프로그래머스 코딩 테스트 문제를 도와주는 어시스턴트입니다."


def main():
    dpo_data = json.loads(Path(DPO_PATH).read_text(encoding="utf-8"))
    github_data = json.loads(Path(GITHUB_PATH).read_text(encoding="utf-8"))

    # title 기준으로 GitHub solutions 매핑
    github_map = {d["title"]: d.get("solutions", []) for d in github_data}

    new_data = []
    skipped = []

    for item in dpo_data:
        title = item["title"]
        solutions = github_map.get(title, [])

        if not solutions:
            skipped.append(title)
            continue

        # user 프롬프트는 기존 chosen에서 그대로 가져옴
        user_content = next(m["content"] for m in item["chosen"] if m["role"] == "user")
        rejected_content = next(m["content"] for m in item["rejected"] if m["role"] == "assistant")

        # GitHub 검증된 코드를 chosen으로 사용 (가장 짧은 통과 코드 선택 = 보통 더 깔끔함)
        verified_code = min(solutions, key=len)

        new_item = {
            "title": title,
            "level": item["level"],
            "chosen": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": verified_code},
            ],
            "rejected": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": rejected_content},
            ],
        }
        new_data.append(new_item)

    print(f"기존 데이터: {len(dpo_data)}개")
    print(f"GitHub 검증 코드로 교체: {len(new_data)}개")
    print(f"제외(solutions 없음): {len(skipped)}개")

    Path(OUTPUT_PATH).write_text(
        json.dumps(new_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"저장 완료 → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()