"""
v8 모델 repetition_penalty 적용 재평가
- 반복 생성 버그가 추론 설정 문제인지 학습 문제인지 검증
- 결과: evaluation/compare_v8_reppenalty.json
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
V8_DIR        = "output/qwen-coder-finetune-v8"
DATA_PATH     = "data/github_solutions.json"
OUTPUT_PATH   = "evaluation/compare_v8_reppenalty.json"

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


# ───────────── 추론 (repetition_penalty 추가) ─────────────
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
            repetition_penalty=1.3,   # ← 핵심 변경: 반복 억제
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

    level1 = [p for p in data if p["level"] == 1 and p.get("solutions")][:5]
    level2 = [p for p in data if p["level"] == 2 and p.get("solutions")][:5]
    problems = level1 + level2
    print(f"평가 문제: {len(problems)}개 (Level1: {len(level1)}, Level2: {len(level2)})\n")

    results = []
    model, tokenizer = load_model(V8_DIR, "v8")

    for i, problem in enumerate(problems):
        title = problem["title"]
        level = problem["level"]
        sig   = extract_sig(problem.get("solutions", []))
        prompts = build_prompts(problem)

        print(f"[v8+reppenalty] [{i+1}/{len(problems)}] {title}")

        entry = {"title": title, "level": level, "sig": sig}

        for task in ["hint", "solution"]:
            resp, ms = ask_model(model, tokenizer, prompts[task])
            entry[task] = resp
            entry[f"{task}_ms"] = ms

            if task == "solution":
                entry["param_ok"] = check_param_accuracy(resp, sig)
                entry["code_correct"] = None  # 수동 채점

            print(f"  {task}: {ms}ms")

        results.append(entry)

    unload_model(model)
    print(f"\nv8 (repetition_penalty=1.3) 평가 완료\n")

    Path("evaluation").mkdir(exist_ok=True)
    Path(OUTPUT_PATH).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"결과 저장 → {OUTPUT_PATH}")

    print("\n=== v8 (repetition_penalty=1.3) 요약 ===")
    for task in ["hint", "solution"]:
        avg = sum(r.get(f"{task}_ms", 0) for r in results) // len(results)
        print(f"  {task:10s} 응답 시간: {avg}ms")

    param_ok = sum(1 for r in results if r.get("param_ok"))
    print(f"  파라미터 정확도: {param_ok}/{len(results)}")

    print("\n주의: code_correct 필드는 null로 저장됨. 직접 검토 후 채점 필요.")
    print("v5 결과(이전 evaluation/compare_v5_v8.json)와 비교해서")
    print("응답 시간이 v5 수준(평균 40초대)으로 줄었는지, 반복 생성 버그가 사라졌는지 확인할 것.")


if __name__ == "__main__":
    main()