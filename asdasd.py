import json

data = json.load(open('data/dataset_evol_clean.json', encoding='utf-8'))

# 제목별로 solution이 남아있는지 확인
titles_with_solution = {d['title'] for d in data if d.get('task') == 'solution'}

before = len(data)
final_data = []
orphan_hints = 0

for item in data:
    if item.get('evol_type') == 'constraint' and item.get('task') == 'hint':
        if item['title'] not in titles_with_solution:
            orphan_hints += 1
            continue
    final_data.append(item)

print(f"고아 hint 제외: {orphan_hints}개")
print(f"최종: {len(final_data)}개")

json.dump(final_data, open('data/dataset_evol_clean.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)