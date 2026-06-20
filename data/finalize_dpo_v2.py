"""
DPO 데이터셋 v2 최종 정리
- import 누락된 38개: 필요한 import 자동 추가
- 진짜 깨진 51개: 제외
- 정상 100개: 그대로 유지
"""
import json
import ast

COMMON_IMPORTS = {
    'deque': 'from collections import deque',
    'defaultdict': 'from collections import defaultdict',
    'Counter': 'from collections import Counter',
    'combinations': 'from itertools import combinations',
    'permutations': 'from itertools import permutations',
    'product': 'from itertools import product',
    'bisect_left': 'from bisect import bisect_left',
    'bisect_right': 'from bisect import bisect_right',
    'heapq': 'import heapq',
    're': 'import re',
    'math': 'import math',
    'np': 'import numpy as np',
    'gcd': 'from math import gcd',
    'sqrt': 'from math import sqrt',
    'datetime': 'import datetime',
}


def find_undefined_names(code: str, func_name: str = "solution"):
    tree = ast.parse(code)
    func_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    if func_node is None:
        return None, None

    defined = set(a.arg for a in func_node.args.args)
    for node in ast.walk(func_node):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for n in ast.walk(target):
                    if isinstance(n, ast.Name):
                        defined.add(n.id)
        if isinstance(node, ast.For):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
        if isinstance(node, ast.comprehension):
            for n in ast.walk(node.target):
                if isinstance(n, ast.Name):
                    defined.add(n.id)
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
            for a in node.args.args:
                defined.add(a.arg)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split('.')[0])

    builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    used = set()
    for node in ast.walk(func_node):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)

    undefined = used - defined - builtins
    real_undefined = {n for n in undefined if not (len(n) == 1 and n.islower())}
    return real_undefined, func_node


def main():
    data = json.load(open('data/dataset_dpo_v2.json', encoding='utf-8'))

    final_data = []
    excluded = []
    fixed_count = 0

    for item in data:
        code = item['chosen'][2]['content']
        undefined, func_node = find_undefined_names(code)

        if undefined is None:
            excluded.append((item['title'], 'no_solution_func'))
            continue

        if not undefined:
            final_data.append(item)
            continue

        if undefined.issubset(COMMON_IMPORTS.keys()):
            # import 추가
            import_lines = "\n".join(COMMON_IMPORTS[n] for n in undefined)
            new_code = f"{import_lines}\n\n{code}"
            item['chosen'][2]['content'] = new_code
            final_data.append(item)
            fixed_count += 1
        else:
            excluded.append((item['title'], f'broken_deps: {undefined}'))

    print(f"원본: {len(data)}개")
    print(f"최종 사용: {len(final_data)}개")
    print(f"  - import 추가로 수정: {fixed_count}개")
    print(f"제외: {len(excluded)}개")

    with open('data/dataset_dpo_v2_final.json', 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
    print("저장 완료 → data/dataset_dpo_v2_final.json")


if __name__ == "__main__":
    main()