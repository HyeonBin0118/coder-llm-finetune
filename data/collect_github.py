"""
GitHub 공개 레포에서 프로그래머스 풀이 수집
- 문제 제목으로 매칭
- solution 함수만 추출
- data/github_solutions.json 저장
"""
import os
import re
import json
import time
import base64
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

PROBLEMS_PATH = "data/problems_parsed.json"
OUTPUT_PATH = "data/github_solutions.json"

SEARCH_QUERIES = [
    "programmers python solution",
    "프로그래머스 파이썬 풀이",
    "programmers algorithm python",
    "프로그래머스 알고리즘 파이썬",
    "programmers coding test python",
    "프로그래머스 코딩테스트 파이썬",
    "programmers level python",
    "프로그래머스 python",
    "programmers solution level1 level2",
    "알고리즘 프로그래머스 풀이 python",
]


def search_repos(query: str, max_repos: int = 20) -> list:
    url = "https://api.github.com/search/repositories"
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": max_repos
    }
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code != 200:
        print(f"레포 검색 실패: {res.status_code}")
        return []
    return res.json().get("items", [])


def get_all_python_files(repo_full_name: str) -> list:
    url = f"https://api.github.com/repos/{repo_full_name}/git/trees/HEAD"
    params = {"recursive": "1"}
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code != 200:
        return []
    tree = res.json().get("tree", [])
    return [f for f in tree if f["path"].endswith(".py") and f["type"] == "blob"]


def get_file_content(repo_full_name: str, file_path: str) -> str:
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}"
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        return ""
    data = res.json()
    if isinstance(data, list):  # 디렉토리로 반환되는 경우
        return ""
    content = data.get("content", "")
    try:
        return base64.b64decode(content).decode("utf-8", errors="ignore")
    except:
        return ""


def extract_solution_func(code: str) -> str:
    match = re.search(r"(def solution\(.*?\n(?:(?:[ \t]+.+\n?)|(?:\n))*)", code)
    if match:
        return match.group(1).strip()
    return ""


def normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^a-z0-9가-힣]", "", title)
    return title


def match_problem(file_path: str, problems: list) -> dict | None:
    path_normalized = normalize_title(file_path)
    for problem in problems:
        title_normalized = normalize_title(problem["title"])
        if title_normalized in path_normalized:
            return problem
    return None


def main():
    with open(PROBLEMS_PATH, encoding="utf-8") as f:
        problems = json.load(f)

    print(f"문제 수: {len(problems)}개")

    # 기존 결과 로드 (있으면 이어서)
    if Path(OUTPUT_PATH).exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            existing = json.load(f)
        solutions = {r["title"]: r["solutions"] for r in existing}
        print(f"기존 결과 로드: {sum(len(v) for v in solutions.values())}개 풀이")
    else:
        solutions = {}

    # 레포 수집
    repos = []
    seen = set()
    for query in SEARCH_QUERIES:
        results = search_repos(query, max_repos=20)
        for r in results:
            if r["full_name"] not in seen:
                seen.add(r["full_name"])
                repos.append(r)
        time.sleep(1)

    print(f"수집된 레포: {len(repos)}개")

    for i, repo in enumerate(repos):
        full_name = repo["full_name"]
        print(f"\n[{i+1}/{len(repos)}] {full_name}")

        py_files = get_all_python_files(full_name)
        print(f"  Python 파일: {len(py_files)}개")

        matched = 0
        for file_info in py_files:
            path = file_info["path"]
            problem = match_problem(path, problems)
            if not problem:
                continue

            content = get_file_content(full_name, path)
            if not content:
                continue

            func = extract_solution_func(content)
            if not func:
                continue

            title = problem["title"]
            if title not in solutions:
                solutions[title] = []

            if func not in solutions[title]:
                solutions[title].append(func)
                matched += 1

            time.sleep(0.3)

        print(f"  매칭된 풀이: {matched}개")

        # 중간 저장 (레포 10개마다)
        if (i + 1) % 10 == 0:
            result = []
            for problem in problems:
                title = problem["title"]
                codes = solutions.get(title, [])
                result.append({
                    "title": title,
                    "level": problem["level"],
                    "url": problem["url"],
                    "description": problem["description"],
                    "solutions": codes
                })
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"  중간 저장 완료 (총 {sum(len(v) for v in solutions.values())}개)")

    # 최종 저장
    result = []
    for problem in problems:
        title = problem["title"]
        codes = solutions.get(title, [])
        result.append({
            "title": title,
            "level": problem["level"],
            "url": problem["url"],
            "description": problem["description"],
            "solutions": codes
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    total_solutions = sum(len(r["solutions"]) for r in result)
    covered = sum(1 for r in result if r["solutions"])
    print(f"\n완료!")
    print(f"  풀이 수집된 문제: {covered}/{len(problems)}개")
    print(f"  총 풀이 수: {total_solutions}개")


if __name__ == "__main__":
    main()