"""
DPO 학습 스크립트
모델: v5 기반 DPO 학습
데이터: data/dataset_dpo.json
"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

MODEL_ID     = "output/qwen-coder-finetune-v5"
OUTPUT_DIR   = "output/qwen-coder-finetune-dpo"
DPO_PATH     = "data/dataset_dpo.json"
MAX_LENGTH   = 512
BETA         = 0.1  # DPO temperature — 낮을수록 chosen/rejected 구분을 강하게 학습


def load_dpo_dataset(path: str) -> Dataset:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    data = []
    for item in raw:
        # chosen/rejected에서 user/assistant 메시지 추출
        chosen_user = next(m["content"] for m in item["chosen"] if m["role"] == "user")
        chosen_asst = next(m["content"] for m in item["chosen"] if m["role"] == "assistant")
        rejected_asst = next(m["content"] for m in item["rejected"] if m["role"] == "assistant")

        data.append({
            "prompt": chosen_user,
            "chosen": chosen_asst,
            "rejected": rejected_asst,
        })

    return Dataset.from_list(data)


def main():
    print("=== DPO 학습 시작 ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    dataset = load_dpo_dataset(DPO_PATH)
    print(f"DPO 데이터: {len(dataset)}개")

    # train/val 분리 (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    val_dataset   = split["test"]
    print(f"train: {len(train_dataset)}개, val: {len(val_dataset)}개")

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
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=5e-5,
        beta=BETA,
        fp16=False,
        bf16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=20,
        lr_scheduler_type="cosine",
        report_to="none",
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_LENGTH // 2,
    )

    trainer = DPOTrainer(
        model=model,
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