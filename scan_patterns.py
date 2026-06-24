import json

data = []
with open('data/train_selected.jsonl', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

solutions = [d for d in data if d.get('task') == 'solution']

# 패턴 체크
has_print = 0
has_repeated_comment = 0
too_short = 0
too_long = 0

for item in solutions:
    code = next(m['content'] for m in item['messages'] if m['role'] == 'assistant')
    lines = code.split('\n')

    if 'print(' in code:
        has_print += 1

    # 주석으로 시작하는 줄이 5개 이상이면 반복 패턴 의심
    comment_lines = sum(1 for l in lines if l.strip().startswith('#'))
    if comment_lines >= 5:
        has_repeated_comment += 1

    if len(lines) < 3:
        too_short += 1

    if len(lines) > 60:
        too_long += 1

print(f"전체 solution: {len(solutions)}개")
print(f"print() 포함: {has_print}개")
print(f"주석 5줄 이상(반복 의심): {has_repeated_comment}개")
print(f"3줄 미만(너무 짧음): {too_short}개")
print(f"60줄 초과(너무 길음, 반복 의심): {too_long}개")