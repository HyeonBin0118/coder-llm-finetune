"""
v4 vs v5 모델 비교 평가
- Level 1 문제 5개, Level 2 문제 5개 (총 10개)
- 각 문제마다 hint / solution 생성 및 비교
- 응답 시간(ms) + pass@1 측정
- 결과: evaluation/compare_v4_v5.json
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
DATA_PATH     = "data/github_solutions.json"
OUTPUT_PATH   = "evaluation/compare_v4_v5.json"

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
    """함수 시그니처 파라미터 일치 여부 확인"""
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

    level1 = [p for p in data if p["level"] == 1 and p.get("solutions")][:5]
    level2 = [p for p in data if p["level"] == 2 and p.get("solutions")][:5]
    problems = level1 + level2
    print(f"평가 문제: {len(problems)}개 (Level1: {len(level1)}, Level2: {len(level2)})\n")

    results = []

    for version, adapter_dir in [("v4", V4_DIR), ("v5", V5_DIR)]:
        model, tokenizer = load_model(adapter_dir, version)

        for i, problem in enumerate(problems):
            title = problem["title"]
            level = problem["level"]
            sig   = extract_sig(problem.get("solutions", []))
            prompts = build_prompts(problem)

            print(f"[{version}] [{i+1}/{len(problems)}] {title}")

            # 기존 entry 찾기 or 새로 생성
            entry = next((r for r in results if r["title"] == title), None)
            if entry is None:
                entry = {"title": title, "level": level, "sig": sig, "v4": {}, "v5": {}}
                results.append(entry)

            for task in ["hint", "solution"]:
                resp, ms = ask_model(model, tokenizer, prompts[task])
                entry[version][task] = resp
                entry[version][f"{task}_ms"] = ms

                if task == "solution":
                    entry[version]["param_ok"] = check_param_accuracy(resp, sig)

                print(f"  {task}: {ms}ms")

        unload_model(model)
        print(f"\n{version} 평가 완료, 모델 언로드\n")

    # 저장
    Path("evaluation").mkdir(exist_ok=True)
    Path(OUTPUT_PATH).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"결과 저장 → {OUTPUT_PATH}")

    # 요약
    print("\n=== v4 vs v5 비교 요약 ===")
    for task in ["hint", "solution"]:
        v4_avg = sum(r["v4"].get(f"{task}_ms", 0) for r in results) // len(results)
        v5_avg = sum(r["v5"].get(f"{task}_ms", 0) for r in results) // len(results)
        print(f"  {task:10s} 응답 시간: v4 {v4_avg}ms / v5 {v5_avg}ms")

    v4_param = sum(1 for r in results if r["v4"].get("param_ok")) 
    v5_param = sum(1 for r in results if r["v5"].get("param_ok"))
    print(f"  파라미터 정확도: v4 {v4_param}/{len(results)} / v5 {v5_param}/{len(results)}")


if __name__ == "__main__":
    main()