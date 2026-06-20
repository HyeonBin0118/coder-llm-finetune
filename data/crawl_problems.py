"""
프로그래머스 Level 1~2 문제 URL 수집
결과: data/problem_urls.json
"""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

OUTPUT_PATH = Path("data/problem_urls.json")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

LEVELS = [1, 2, 3]
BASE_API = "https://school.programmers.co.kr/api/v2/school/challenges/"


def fetch_level(page, level: int) -> list[dict]:
    results = []
    page_num = 1

    while True:
        url = f"{BASE_API}?perPage=50&order=recent&search=&page={page_num}&levels[]={level}"
        response = page.request.get(url)
        data = response.json()

        items = data.get("result", [])
        if not items:
            print(f"  → Level {level} 마지막 페이지 ({page_num})")
            break

        for c in items:
            title = c.get("title", "").strip()
            lesson_id = c.get("id")
            if title and lesson_id:
                results.append({
                    "title": title,
                    "level": level,
                    "url": f"https://school.programmers.co.kr/learn/courses/30/lessons/{lesson_id}",
                    "lesson_id": str(lesson_id),
                })
                print(f"  [{level}] {title}")

        total_pages = data.get("totalPages", 1)
        if page_num >= total_pages:
            break

        page_num += 1
        time.sleep(0.5)

    return results


def crawl_problem_urls() -> list[dict]:
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto("https://school.programmers.co.kr/learn/challenges",
                  wait_until="networkidle", timeout=15000)

        for level in LEVELS:
            print(f"\n[Level {level}] 수집 중...")
            results.extend(fetch_level(page, level))

        browser.close()
    return results


if __name__ == "__main__":
    urls = crawl_problem_urls()
    OUTPUT_PATH.write_text(
        json.dumps(urls, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n총 {len(urls)}개 문제 수집 완료 → {OUTPUT_PATH}")