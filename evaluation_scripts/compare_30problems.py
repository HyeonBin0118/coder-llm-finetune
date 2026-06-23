"""
v4 vs v5 vs DPO vs v8 모델 비교 평가 (30문제 확장판)
- Level 1 문제 15개, Level 2 문제 15개 (총 30개)
- 결과: evaluation/compare_30problems.json
"""
import json
import time
import re
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ───────────── 설정 ─────────────
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
V4_DIR        = "output/qwen-coder-finetune-v4"
V5_DIR        = "output/qwen-coder-finetune-v5"
DPO_DIR       = "output/qwen-coder-finetune-dpo"
V8_DIR        = "output/qwen-coder-finetune-v8"
DATA_PATH     = "data/github_solutions.json"
OUTPUT_PATH   = "evaluation/compare_30problems.json"

SYSTEM_PROMPT = "당신은 프로그래머스 코딩 테스트 문제를 도와주는 어시스턴트입니다. 문제를 분석하고 힌트, 접근법, 정답 코드를 단계별로 제공합니다."

HINT_PROMPT = """다음 프로그래머스 문제의 힌트를 알려주세요.

제목: {title}
난이도: Level {level}
문제 설명:
{description}"""

SOLUTION_PROMPT = """다음 프로그래머스 문제의 Python 정답 코드를 작성해주세요.

제목: {title}
난이도: Level {level}
함수 시그니처: {sig}
문제 설명:
{description}"""

# 기존 10문제 + 신규 20문제 = 30문제 제목 고정
EXISTING_10 = [
    "가장 많이 받은 선물", "[PCCP 기출문제] 1번 / 붕대 감기", "[PCCE 기출문제] 9번 / 이웃한 칸",
    "[PCCE 기출문제] 10번 / 데이터 분석", "달리기 경주", "서버 증설 횟수",
    "지게차와 크레인", "비밀 코드 해독", "[PCCP 기출문제] 2번 / 퍼즐 게임 챌린지", "도넛과 막대 그래프"
]
NEW_LEVEL1_10 = [
    "추억 점수", "공원 산책", "바탕화면 정리", "덧칠하기", "대충 만든 자판",
    "카드 뭉치", "둘만의 암호", "개인정보 수집 유효기간", "크기가 작은 부분 문자열", "가장 가까운 같은 글자"
]
NEW_LEVEL2_10 = [
    "[PCCP 기출문제] 2번 / 석유 시추", "요격 시스템", "두 원 사이의 정수 쌍", "연속된 부분 수열의 합",
    "과제 진행하기", "광물 캐기", "리코쳇 로봇", "당구 연습", "혼자서 하는 틱택토", "미로 탈출"
]
ALL_30 = EXISTING_10 + NEW_LEVEL1_10 + NEW_LEVEL2_10


# ───────────── 모델 로드 ─────────────
def load_model(adapter_dir: str, label: str):
    print(f"{label} 모델 로드 중...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
    )

    if label == "dpo":
        # DPO는 v5 기반으로 학습됐으므로 base -> v5 -> dpo 순으로 얹는다
        model = PeftModel.from_pretrained(base_model, V5_DIR)
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, adapter_dir)
    else:
        model = PeftModel.from_pretrained(base_model, adapter_dir)

    model.eval()
    print(f"{label} 로드 완료")
    return model, tokenizer


def unload_model(model):
    del model
    torch.cuda.empty_cache()


# ───────────── 추론 ─────────────
def ask_model(model, tokenizer, prompt: str) -> tuple[str, int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    text = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed_ms = int((time.time() - start) * 1000)
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), elapsed_ms


# ───────────── 유틸 ─────────────
def extract_sig(solutions: list) -> str:
    if not solutions:
        return "def solution(...)"
    match = re.search(r'def solution\([^)]*\)', solutions[0])
    return match.group(0) if match else "def solution(...)"


def build_prompts(problem: dict) -> dict:
    title = problem["title"]
    level = problem["level"]
    desc  = problem["description"][:800]
    sig   = extract_sig(problem.get("solutions", []))
    return {
        "hint":     HINT_PROMPT.format(title=title, level=level, description=desc),
        "solution": SOLUTION_PROMPT.format(title=title, level=level, sig=sig, description=desc),
    }


def check_param_accuracy(response: str, sig: str) -> bool:
    expected_params = re.findall(r'\b(\w+)\b', sig.replace("def solution(", "").rstrip(")"))
    if not expected_params:
        return True
    match = re.search(r'def solution\([^)]*\)', response)
    if not match:
        return False
    actual_params = re.findall(r'\b(\w+)\b', match.group(0).replace("def solution(", "").rstrip(")"))
    return set(expected_params) == set(actual_params)


# ───────────── 메인 ─────────────
def main():
    data = json.loads(Path(DATA_PATH).read_text(encoding="utf-8"))
    title_map = {d["title"]: d for d in data}

    problems = []
    for title in ALL_30:
        if title in title_map and title_map[title].get("solutions"):
            problems.append(title_map[title])
        else:
            print(f"경고: '{title}' 문제를 찾을 수 없거나 solutions가 없음")

    print(f"평가 문제: {len(problems)}개\n")

    results = []
    versions = [("v4", V4_DIR), ("v5", V5_DIR), ("dpo", DPO_DIR), ("v8", V8_DIR)]

    for version, adapter_dir in versions:
        model, tokenizer = load_model(adapter_dir, version)

        for i, problem in enumerate(problems):
            title = problem["title"]
            level = problem["level"]
            sig   = extract_sig(problem.get("solutions", []))
            prompts = build_prompts(problem)

            print(f"[{version}] [{i+1}/{len(problems)}] {title}")

            entry = next((r for r in results if r["title"] == title), None)
            if entry is None:
                entry = {"title": title, "level": level, "sig": sig}
                for v, _ in versions:
                    entry[v] = {}
                results.append(entry)

            for task in ["hint", "solution"]:
                resp, ms = ask_model(model, tokenizer, prompts[task])
                entry[version][task] = resp
                entry[version][f"{task}_ms"] = ms

                if task == "solution":
                    entry[version]["param_ok"] = check_param_accuracy(resp, sig)
                    entry[version]["code_correct"] = None  # 수동 채점

                print(f"  {task}: {ms}ms")

        unload_model(model)
        print(f"\n{version} 평가 완료, 모델 언로드\n")

    Path("evaluation").mkdir(exist_ok=True)
    Path(OUTPUT_PATH).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"결과 저장 → {OUTPUT_PATH}")

    print("\n=== 30문제 비교 요약 ===")
    for task in ["hint", "solution"]:
        for v, _ in versions:
            avg = sum(r[v].get(f"{task}_ms", 0) for r in results) // len(results)
            print(f"  {task:10s} 응답 시간 [{v}]: {avg}ms")

    for v, _ in versions:
        param_ok = sum(1 for r in results if r[v].get("param_ok"))
        print(f"  파라미터 정확도 [{v}]: {param_ok}/{len(results)}")

    print("\n주의: code_correct 필드는 null로 저장됨. 직접 검토 후 채점 필요.")


if __name__ == "__main__":
    main()