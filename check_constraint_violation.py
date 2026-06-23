import json
import re

data = json.load(open('data/dataset_evol.json', encoding='utf-8'))
constraint_solutions = [d for d in data if d.get('evol_type') == 'constraint' and d.get('task') == 'solution']

violated = 0
for item in constraint_solutions:
    code = next(m['content'] for m in item['messages'] if m['role'] == 'assistant')
    user_msg = next(m['content'] for m in item['messages'] if m['role'] == 'user')
    
    # "반복문" 또는 "내장 함수" 금지 언급이 있는지 확인
    if '반복문' in user_msg and ('for ' in code or 'while ' in code):
        violated += 1

print(f"제약 위반(반복문 금지인데 반복문 사용): {violated}/{len(constraint_solutions)}")