"""
프로그래머스 문제 본문 파싱
- SQL 문제 제외 (파이썬 코딩 문제만)
- 결과: data/problems_parsed.json
"""
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

INPUT_PATH  = Path("data/problem_urls.json")
OUTPUT_PATH = Path("data/problems_parsed.json")

# SQL 관련 키워드 포함된 문제 제외
SQL_KEYWORDS = ["구하기", "조회하기", "목록 출력", "출력하기", "찾기", "개수 구하기"]

# SQL 문제 판별 (제목 기준 1차 필터 + 본문 기준 2차 필터)
SQL_TITLE_KEYWORDS = [
    "SELECT", "JOIN", "GROUP BY", "WHERE", "NULL", "DATETIME", "DATE",
    "입양", "물고기", "대장균", "아이템", "식품", "자동차 대여", "자동차 평균",
    "중고거래", "조건에 부합하는", "조건에 맞는", "재구매", "상품 별",
    "가격대 별", "카테고리 별", "진료과별", "월별", "연도별", "노선별",
    "연도 별", "분기별", "특정 물고기", "물고기 종류"
]


def is_sql_problem(title: str, description: str) -> bool:
    for kw in SQL_TITLE_KEYWORDS:
        if kw in title:
            return True
    sql_signs = ["SELECT", "FROM", "WHERE", "JOIN", "GROUP BY", "ORDER BY"]
    count = sum(1 for s in sql_signs if s in description.upper())
    return count >= 2


def parse_problem(page, url: str) -> dict | None:
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(3000)

        # 제목
        title = ""
        if page.locator("li.active").count() > 0:
            title = page.locator("li.active").first.inner_text().strip()

        # 본문 섹션들
        sections = page.query_selector_all("div.lesson-content div.markdown")
        description = ""
        constraints = ""
        examples = []

        for i, el in enumerate(sections):
            text = el.inner_text().strip()
            if i == 0:
                description = text
            elif "제한" in text:
                constraints = text
            elif "입출력" in text or "예시" in text:
                for row in el.query_selector_all("table tbody tr"):
                    cols = [c.inner_text().strip() for c in row.query_selector_all("td")]
                    if len(cols) >= 2:
                        examples.append({
                            "input": cols[0],
                            "output": cols[1],
                            "explanation": cols[2] if len(cols) > 2 else ""
                        })

        if not title or not description:
            return None

        if is_sql_problem(title, description):
            return None

        return {
            "title": title,
            "url": url,
            "description": description,
            "constraints": constraints,
            "examples": examples,
        }

    except Exception as e:
        print(f"  오류: {e}")
        return None


def main():
    urls = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    print(f"총 {len(urls)}개 문제 파싱 시작\n")

    results = []
    failed = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for i, item in enumerate(urls): # 전체 문제 파싱
        ##for i, item in enumerate(urls[:10]): # 테스트용으로 처음 10개만 파싱
            title = item["title"]
            url = item["url"]
            level = item["level"]
            print(f"[{i+1}/{len(urls)}] Level {level} | {title}")

            result = parse_problem(page, url)
            if result:
                result["level"] = level
                results.append(result)
                print(f"  ✓ 저장 ({len(result['description'])}자)")
            else:
                failed.append(url)
                print(f"  ✗ 스킵 (SQL 또는 파싱 실패)")

            time.sleep(1)

        browser.close()

    OUTPUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n완료: {len(results)}개 저장, {len(failed)}개 스킵")
    print(f"→ {OUTPUT_PATH}")


if __name__ == "__main__":
    main()