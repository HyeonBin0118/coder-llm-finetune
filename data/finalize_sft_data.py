"""
SFT 학습 데이터(train_selected.jsonl) 정제
- task == "solution": AST 검증 후 import 누락은 보정, 진짜 깨진 건 제외
- task == "hint" / "approach": 그대로 유지 (코드가 아니라 검증 대상 아님)
- 결과: data/train_selected_v2.jsonl
"""
import json
import ast

COMMON_IMPORTS = {
    'deque': 'from collections import deque',
    'defaultdict': 'from collections import defaultdict',
    'Counter': 'from collections import Counter',
    'combinations': 'from itertools import combinations',
    'combinations_with_replacement': 'from itertools import combinations_with_replacement',
    'permutations': 'from itertools import permutations',
    'product': 'from itertools import product',
    'cycle': 'from itertools import cycle',
    'bisect_left': 'from bisect import bisect_left',
    'bisect_right': 'from bisect import bisect_right',
    'heapq': 'import heapq',
    'heappop': 'from heapq import heappop',
    'heappush': 'from heapq import heappush',
    're': 'import re',
    'math': 'import math',
    'np': 'import numpy as np',
    'gcd': 'from math import gcd',
    'sqrt': 'from math import sqrt',
    'ceil': 'from math import ceil',
    'floor': 'from math import floor',
    'datetime': 'import datetime',
    'string': 'import string',
    'ascii_uppercase': 'from string import ascii_uppercase',
    'ascii_lowercase': 'from string import ascii_lowercase',
    'itertools': 'import itertools',
}


def find_undefined_names(code: str, func_name: str = "solution"):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None, "syntax_error"

    func_node = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == func_name), None)
    if func_node is None:
        return None, "no_solution_func"

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
    return real_undefined, None


def main():
    data = []
    with open('data/train_selected.jsonl', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    print(f"원본 전체: {len(data)}개")

    final_data = []
    excluded = []
    fixed_count = 0
    kept_non_solution = 0

    for item in data:
        if item.get('task') != 'solution':
            # hint, approach는 검증 없이 그대로 유지
            final_data.append(item)
            kept_non_solution += 1
            continue

        code = next(m["content"] for m in item["messages"] if m["role"] == "assistant")
        undefined, err = find_undefined_names(code)

        if err == "syntax_error" or err == "no_solution_func":
            excluded.append((item["title"], err))
            continue

        if not undefined:
            final_data.append(item)
            continue

        if undefined.issubset(COMMON_IMPORTS.keys()):
            import_lines = "\n".join(COMMON_IMPORTS[n] for n in undefined)
            new_code = f"{import_lines}\n\n{code}"
            # messages 안의 assistant content를 교체
            for m in item["messages"]:
                if m["role"] == "assistant":
                    m["content"] = new_code
            final_data.append(item)
            fixed_count += 1
        else:
            excluded.append((item["title"], f"broken_deps: {undefined}"))

    print(f"hint/approach (검증 없이 유지): {kept_non_solution}개")
    print(f"solution 중 정상/수정 후 사용: {len(final_data) - kept_non_solution}개")
    print(f"  - import 추가로 수정: {fixed_count}개")
    print(f"제외(진짜 깨진 코드): {len(excluded)}개")
    print(f"최종 데이터: {len(final_data)}개")

    with open('data/train_selected_v2.jsonl', 'w', encoding='utf-8') as f:
        for item in final_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print("저장 완료 → data/train_selected_v2.jsonl")


if __name__ == "__main__":
    main()