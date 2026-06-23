import json

data = json.load(open('data/github_solutions.json', encoding='utf-8'))

existing_10 = [
    "가장 많이 받은 선물", "[PCCP 기출문제] 1번 / 붕대 감기", "[PCCE 기출문제] 9번 / 이웃한 칸",
    "[PCCE 기출문제] 10번 / 데이터 분석", "달리기 경주", "서버 증설 횟수",
    "지게차와 크레인", "비밀 코드 해독", "[PCCP 기출문제] 2번 / 퍼즐 게임 챌린지", "도넛과 막대 그래프"
]

level1 = [d for d in data if d['level'] == 1 and d.get('solutions') and d['title'] not in existing_10]
level2 = [d for d in data if d['level'] == 2 and d.get('solutions') and d['title'] not in existing_10]

print(f"Level1 추가 후보: {len(level1)}개")
print(f"Level2 추가 후보: {len(level2)}개")

new_10_level1 = level1[:10]
new_10_level2 = level2[:10]

print("\n=== 추가할 Level1 10개 ===")
for d in new_10_level1:
    print(f"  - {d['title']}")
print("\n=== 추가할 Level2 10개 ===")
for d in new_10_level2:
    print(f"  - {d['title']}")