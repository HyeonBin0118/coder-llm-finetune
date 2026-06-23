import json
data = [json.loads(l) for l in open('data/train_selected_v5_original.jsonl', encoding='utf-8')]
print(f"복원된 v5 원본 데이터: {len(data)}개")