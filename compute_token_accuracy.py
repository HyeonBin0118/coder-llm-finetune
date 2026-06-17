"""
v4 vs v5 token accuracy 재평가
동일한 val.jsonl 기준으로 두 모델의 다음 토큰 예측 정확도를 측정
결과: evaluation/token_accuracy.json
"""
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
VAL_PATH = "data/val.jsonl"
MAX_LENGTH = 512


def load_val_dataset(path: str):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def format_messages(messages: list) -> str:
    text = ""
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return text


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


@torch.no_grad()
def compute_token_accuracy(model, tokenizer, val_data: list) -> float:
    total_correct = 0
    total_tokens = 0

    for item in val_data:
        text = format_messages(item["messages"])
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=MAX_LENGTH).to("cuda")
        input_ids = inputs["input_ids"]

        outputs = model(**inputs, labels=input_ids)
        logits = outputs.logits

        # 다음 토큰 예측: logits[t]가 input_ids[t+1]을 맞추는지 확인
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        predictions = shift_logits.argmax(dim=-1)
        correct = (predictions == shift_labels).sum().item()
        total = shift_labels.numel()

        total_correct += correct
        total_tokens += total

    return total_correct / total_tokens if total_tokens > 0 else 0.0


def main():
    val_data = load_val_dataset(VAL_PATH)
    print(f"val 데이터: {len(val_data)}개")

    results = {}

    for version, adapter_dir in [
        ("v4", "output/qwen-coder-finetune-v4"),
        ("v5", "output/qwen-coder-finetune-v5"),
    ]:
        model, tokenizer = load_model(adapter_dir, version)
        acc = compute_token_accuracy(model, tokenizer, val_data)
        results[version] = round(acc * 100, 1)
        print(f"{version} token accuracy: {results[version]}%")
        unload_model(model)

    Path("evaluation").mkdir(exist_ok=True)
    Path("evaluation/token_accuracy.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n결과 저장 → evaluation/token_accuracy.json")
    print(f"\n=== 비교 ===")
    print(f"v4: {results['v4']}%")
    print(f"v5: {results['v5']}%")


if __name__ == "__main__":
    main()