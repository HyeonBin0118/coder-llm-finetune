import json
data = [json.loads(l) for l in open('data/train_selected.jsonl', encoding='utf-8')]
print(f"현재 train_selected.jsonl: {len(data)}개")