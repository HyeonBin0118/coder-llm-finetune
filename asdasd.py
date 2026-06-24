import json

data = []
with open('data/train_selected.jsonl', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

print(f"전체: {len(data)}개\n")

# task별 분포
from collections import Counter
tasks = Counter(d.get('task', 'unknown') for d in data)
print("task별 분포:", tasks)

# solution 타입만 모아서 별도 파일로 저장 (보기 편하게)
solutions = [d for d in data if d.get('task') == 'solution']
with open('solutions_preview.json', 'w', encoding='utf-8') as f:
    json.dump(solutions[:20], f, ensure_ascii=False, indent=2)

print(f"\nsolution 타입: {len(solutions)}개")
print("→ solutions_preview.json에 처음 20개 저장함 (VS Code나 메모장으로 열어서 확인)")