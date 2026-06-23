"""
GitHub 정답 코드(github_solutions.json) AST 검증
- 문법 오류 체크
- 미정의 변수(helper 함수 의존) 체크
- import 누락 체크
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
    data = json.load(open('data/github_solutions.json', encoding='utf-8'))

    total_solutions = 0
    clean_count = 0
    fixed_count = 0
    broken_count = 0
    syntax_error_count = 0
    no_func_count = 0

    for item in data:
        solutions = item.get('solutions', [])
        if not solutions:
            continue

        new_solutions = []
        for code in solutions:
            total_solutions += 1
            undefined, err = find_undefined_names(code)

            if err == "syntax_error":
                syntax_error_count += 1
                continue
            if err == "no_solution_func":
                no_func_count += 1
                continue

            if not undefined:
                new_solutions.append(code)
                clean_count += 1
                continue

            if undefined.issubset(COMMON_IMPORTS.keys()):
                import_lines = "\n".join(COMMON_IMPORTS[n] for n in undefined)
                new_code = f"{import_lines}\n\n{code}"
                new_solutions.append(new_code)
                fixed_count += 1
            else:
                broken_count += 1

        item['solutions'] = new_solutions

    # 검증 후 solutions가 빈 문제는 통계에서 확인
    empty_after = sum(1 for item in data if not item.get('solutions'))

    print(f"전체 풀이 코드: {total_solutions}개")
    print(f"정상: {clean_count}개")
    print(f"import 추가로 수정: {fixed_count}개")
    print(f"진짜 깨진 코드(제외): {broken_count}개")
    print(f"문법 오류(제외): {syntax_error_count}개")
    print(f"solution 함수 없음(제외): {no_func_count}개")
    print(f"\n검증 후 풀이가 0개가 된 문제: {empty_after}개")

    with open('data/github_solutions_v2.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("\n저장 완료 → data/github_solutions_v2.json")


if __name__ == "__main__":
    main()