"""
train_selected.jsonl 경량 정제
1. print()문 라인 제거 (코드 본체는 유지)
2. 같은 제목(title)의 solution이 중복되면 1개만 남김 (가장 먼저 등장한 것 유지)
- hint/approach는 그대로 유지 (중복 체크/정제 대상 아님)
결과: data/train_selected_v2_clean.jsonl
"""
import json
import re

INPUT_PATH = "data/train_selected.jsonl"
OUTPUT_PATH = "data/train_selected_v2_clean.jsonl"


def remove_print_lines(code: str) -> str:
    lines = code.split('\n')
    cleaned = [l for l in lines if not re.match(r'^\s*print\s*\(', l)]
    return '\n'.join(cleaned)


def main():
    data = []
    with open(INPUT_PATH, encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    print(f"원본 전체: {len(data)}개")

    seen_solution_titles = set()
    final_data = []
    print_cleaned_count = 0
    dup_removed_count = 0

    for item in data:
        task = item.get('task')

        if task != 'solution':
            # hint, approach는 그대로 유지
            final_data.append(item)
            continue

        title = item.get('title')

        # 같은 제목의 solution 중복 제거 (가장 먼저 등장한 것만 유지)
        if title in seen_solution_titles:
            dup_removed_count += 1
            continue
        seen_solution_titles.add(title)

        # print() 라인 제거
        for msg in item['messages']:
            if msg['role'] == 'assistant':
                original = msg['content']
                cleaned = remove_print_lines(original)
                if cleaned != original:
                    print_cleaned_count += 1
                msg['content'] = cleaned

        final_data.append(item)

    print(f"print() 라인 제거된 solution: {print_cleaned_count}개")
    print(f"중복 제목으로 제외된 solution: {dup_removed_count}개")
    print(f"최종 데이터: {len(final_data)}개")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"저장 완료 → {OUTPUT_PATH}")

    # 최종 task별 분포
    from collections import Counter
    tasks = Counter(d.get('task', 'unknown') for d in final_data)
    print(f"\ntask별 분포: {dict(tasks)}")


if __name__ == "__main__":
    main()