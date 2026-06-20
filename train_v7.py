"""
QLoRA 파인튜닝 스크립트 (v7)
모델: Qwen2.5-Coder-7B-Instruct
데이터: data/train_selected_v2.jsonl (SFT 데이터 AST 검증 후 정제본), data/val.jsonl
변경점: 학습 데이터만 정제(1,381→1,354개), rank는 v5와 동일하게 16 유지
"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer

MODEL_ID   = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "output/qwen-coder-finetune-v7"
TRAIN_PATH = "data/train_selected_v2.jsonl"
VAL_PATH   = "data/val.jsonl"
MAX_LENGTH = 512


def load_jsonl(path: str) -> Dataset:
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def format_messages(example):
    messages = example["messages"]
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    return {"text": text}


def main():
    print("=== QLoRA 파인튜닝 시작 (v7: 학습 데이터 AST 검증 정제) ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train_dataset = load_jsonl(TRAIN_PATH)
    val_dataset   = load_jsonl(VAL_PATH)
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

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset.map(format_messages),
        eval_dataset=val_dataset.map(format_messages),
        tokenizer=tokenizer,
        max_seq_length=MAX_LENGTH,
        dataset_text_field="text",
    )

    print("학습 시작...")
    trainer.train()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"모델 저장 완료 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()