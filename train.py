"""
QLoRA 파인튜닝 스크립트
모델: Qwen2.5-Coder-7B-Instruct
데이터: data/train.jsonl, data/val.jsonl
"""
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

MODEL_ID   = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "output/qwen-coder-finetune"
TRAIN_PATH = "data/train.jsonl"
VAL_PATH   = "data/val.jsonl"
MAX_LENGTH = 2048


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


def tokenize(example, tokenizer):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result


def main():
    print("=== QLoRA 파인튜닝 시작 ===")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    train_dataset = load_jsonl(TRAIN_PATH).map(format_messages)
    val_dataset   = load_jsonl(VAL_PATH).map(format_messages)
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
        dtype=torch.float16,
    )

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

    train_tokenized = train_dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=train_dataset.column_names
    )
    val_tokenized = val_dataset.map(
        lambda x: tokenize(x, tokenizer),
        remove_columns=val_dataset.column_names
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print("학습 시작...")
    trainer.train()

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"모델 저장 완료 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()