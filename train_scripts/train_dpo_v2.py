"""
DPO 학습 스크립트 (v2)
모델: v5 기반 DPO 학습
데이터: data/dataset_dpo_v2_final.json (GitHub 검증 코드를 chosen으로 사용한 정제본)
"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

MODEL_ID   = "output/qwen-coder-finetune-v5"
OUTPUT_DIR = "output/qwen-coder-finetune-dpo-v2"
DPO_PATH   = "data/dataset_dpo_v2_final.json"
MAX_LENGTH = 512
BETA       = 0.1

SYSTEM_PROMPT = "당신은 프로그래머스 코딩 테스트 문제를 도와주는 어시스턴트입니다."


def load_dpo_dataset(path: str, tokenizer) -> Dataset:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    data = []
    for item in raw:
        try:
            chosen_user   = next(m["content"] for m in item["chosen"]   if m["role"] == "user")
            chosen_asst   = next(m["content"] for m in item["chosen"]   if m["role"] == "assistant")
            rejected_asst = next(m["content"] for m in item["rejected"] if m["role"] == "assistant")
        except StopIteration:
            continue

        if not chosen_user or not chosen_asst or not rejected_asst:
            continue

        # 미리 토크나이징해서 None 걸러내기
        prompt_tok = tokenizer(
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{chosen_user}<|im_end|>\n<|im_start|>assistant\n",
            truncation=True, max_length=MAX_LENGTH // 2
        )
        chosen_tok = tokenizer(
            chosen_asst + "<|im_end|>",
            truncation=True, max_length=MAX_LENGTH // 2
        )
        rejected_tok = tokenizer(
            rejected_asst + "<|im_end|>",
            truncation=True, max_length=MAX_LENGTH // 2
        )

        if (prompt_tok["input_ids"] is None or
            chosen_tok["input_ids"] is None or
            rejected_tok["input_ids"] is None):
            continue

        if (len(prompt_tok["input_ids"]) == 0 or
            len(chosen_tok["input_ids"]) == 0 or
            len(rejected_tok["input_ids"]) == 0):
            continue

        data.append({
            "prompt":   chosen_user,
            "chosen":   chosen_asst,
            "rejected": rejected_asst,
        })

    print(f"유효 데이터: {len(data)}개")
    return Dataset.from_list(data)


def main():
    print("=== DPO 학습 시작 (v2: GitHub 검증 코드 기반) ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"모델 로드 중: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = MAX_LENGTH

    dataset = load_dpo_dataset(DPO_PATH, tokenizer)

    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset   = split["test"]
    print(f"train: {len(train_dataset)}개, val: {len(val_dataset)}개")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        beta=BETA,
        fp16=False,
        bf16=False,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        report_to="none",
        remove_unused_columns=False,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_LENGTH // 2,
        max_target_length=MAX_LENGTH // 2,
        label_pad_token_id=-100,
        padding_value=0,
        truncation_mode="keep_start",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    print("DPO 학습 시작...")
    trainer.train()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"모델 저장 완료 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()