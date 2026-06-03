"""
파인튜닝 모델 vs GPT-4o-mini 비교 평가
동일한 문제 10개로 힌트/정답 품질 비교
결과: evaluation/results.json
"""
import json
from pathlib import Path
from openai import OpenAI
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
FINETUNED_DIR = "output/qwen-coder-finetune"
DATA_PATH     = "data/dataset_raw.json"
OUTPUT_DIR    = Path("evaluation")
OUTPUT_DIR.mkdir(exist_ok=True)

SYSTEM_PROMPT = """당신은 프로그래머스 코딩 테스트 문제를 도와주는 어시스턴트입니다.
문제를 분석하고 힌트, 접근법, 정답 코드를 단계별로 제공합니다."""

client = OpenAI()


def make_user_prompt(problem: dict, task: str) -> str:
    desc = problem["description"][:500]
    constraints = problem.get("constraints", "")[:200]
    if task == "hint":
        return f"""다음 프로그래머스 문제의 힌트를 알려주세요.

제목: {problem['title']}
난이도: Level {problem['level']}
문제 설명: {desc}
제한사항: {constraints}"""
    else:
        return f"""다음 프로그래머스 문제의 Python 정답 코드를 작성해주세요.

제목: {problem['title']}
난이도: Level {problem['level']}
문제 설명: {desc}
제한사항: {constraints}"""


def ask_gpt(user_prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def load_finetuned_model():
    print("파인튜닝 모델 로드 중...")
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
    return model, tokenizer


def ask_finetuned(model, tokenizer, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"

    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    data = json.loads(Path(DATA_PATH).read_text(encoding="utf-8"))

    # Level 1, 2 각 5개씩 선택
    level1 = [p for p in data if p["level"] == 1][:5]
    level2 = [p for p in data if p["level"] == 2][:5]
    test_problems = level1 + level2
    print(f"평가 문제: {len(test_problems)}개")

    model, tokenizer = load_finetuned_model()

    results = []
    for i, problem in enumerate(test_problems):
        print(f"\n[{i+1}/{len(test_problems)}] {problem['title']} (Level {problem['level']})")

        for task in ["hint", "solution"]:
            user_prompt = make_user_prompt(problem, task)

            print(f"  GPT-4o-mini {task} 생성 중...")
            gpt_response = ask_gpt(user_prompt)

            print(f"  파인튜닝 모델 {task} 생성 중...")
            ft_response = ask_finetuned(model, tokenizer, user_prompt)

            results.append({
                "title": problem["title"],
                "level": problem["level"],
                "task": task,
                "reference": problem.get(task, ""),
                "gpt_response": gpt_response,
                "finetuned_response": ft_response,
            })

    output_path = OUTPUT_DIR / "results.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n결과 저장 → {output_path}")
    print("\n=== 완료 ===")
    print("evaluation/results.json")


if __name__ == "__main__":
    main()