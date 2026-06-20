"""
GPT-4o-mini vs Qwen2.5-Coder-7B v4 비교 평가
- Level 1 문제 5개, Level 2 문제 5개 (총 10개)
- 각 문제마다 hint / approach / solution 생성
- 응답 시간(ms) 측정
- 결과: evaluation/compare_results.json
"""
import json
import time
import re
import torch
from pathlib import Path
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

# ───────────── 설정 ─────────────
BASE_MODEL_ID  = "Qwen/Qwen2.5-Coder-7B-Instruct"
FINETUNED_DIR  = "output/qwen-coder-finetune-v4"
DATA_PATH      = "data/github_solutions.json"
OUTPUT_PATH    = "evaluation/compare_results.json"

SYSTEM_PROMPT = "당신은 프로그래머스 코딩 테스트 문제를 도와주는 어시스턴트입니다. 문제를 분석하고 힌트, 접근법, 정답 코드를 단계별로 제공합니다."

HINT_PROMPT = """다음 프로그래머스 문제의 힌트를 알려주세요.

제목: {title}
난이도: Level {level}
문제 설명:
{description}"""

APPROACH_PROMPT = """다음 프로그래머스 문제의 접근법을 설명해주세요.

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

# ───────────── GPT 클라이언트 ─────────────
gpt_client = OpenAI()

def ask_gpt(prompt: str) -> tuple[str, int]:
    """GPT-4o-mini 호출, (응답, 소요ms) 반환"""
    start = time.time()
    response = gpt_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    elapsed_ms = int((time.time() - start) * 1000)
    return response.choices[0].message.content.strip(), elapsed_ms


# ───────────── 로컬 모델 로드 ─────────────
def load_local_model():
    print("로컬 모델 로드 중...")
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
    device_map="auto",
    trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, FINETUNED_DIR)
    model.eval()
    print("로컬 모델 로드 완료")
    return model, tokenizer


def ask_local(model, tokenizer, prompt: str) -> tuple[str, int]:
    """로컬 모델 추론, (응답, 소요ms) 반환"""
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
        "approach": APPROACH_PROMPT.format(title=title, level=level, description=desc),
        "solution": SOLUTION_PROMPT.format(title=title, level=level, sig=sig, description=desc),
    }


# ───────────── 메인 ─────────────
def main():
    data = json.loads(Path(DATA_PATH).read_text(encoding="utf-8"))

    # Level 1 5개, Level 2 5개 선택 (풀이 있는 것만)
    level1 = [p for p in data if p["level"] == 1 and p.get("solutions")][:5]
    level2 = [p for p in data if p["level"] == 2 and p.get("solutions")][:5]
    problems = level1 + level2
    print(f"평가 문제: {len(problems)}개 (Level1: {len(level1)}, Level2: {len(level2)})")

    model, tokenizer = load_local_model()

    results = []
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem['title']} (Level {problem['level']})")
        prompts = build_prompts(problem)

        entry = {
            "title": problem["title"],
            "level": problem["level"],
            "description": problem["description"][:800],
            "gpt":   {},
            "local": {},
        }

        for task in ["hint", "approach", "solution"]:
            prompt = prompts[task]

            print(f"  GPT {task} ...")
            gpt_resp, gpt_ms = ask_gpt(prompt)
            entry["gpt"][task]          = gpt_resp
            entry["gpt"][f"{task}_ms"]  = gpt_ms

            print(f"  Local {task} ...")
            local_resp, local_ms = ask_local(model, tokenizer, prompt)
            entry["local"][task]         = local_resp
            entry["local"][f"{task}_ms"] = local_ms

            print(f"    GPT: {gpt_ms}ms / Local: {local_ms}ms")

        results.append(entry)

    Path("evaluation").mkdir(exist_ok=True)
    Path(OUTPUT_PATH).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n결과 저장 → {OUTPUT_PATH}")

    # 응답 시간 요약
    print("\n=== 응답 시간 요약 (평균) ===")
    for task in ["hint", "approach", "solution"]:
        gpt_avg   = sum(r["gpt"][f"{task}_ms"]   for r in results) // len(results)
        local_avg = sum(r["local"][f"{task}_ms"] for r in results) // len(results)
        print(f"  {task:10s}: GPT {gpt_avg}ms / Local {local_avg}ms")


if __name__ == "__main__":
    main()
