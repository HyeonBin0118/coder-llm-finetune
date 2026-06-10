"""
IFD + K-Means 데이터 선별 (논문: Data-efficient LLM Fine-tuning for Code Generation)
- 입력: data/train.jsonl
- 출력: data/train_selected.jsonl
- RTX 3060 12GB 환경: 4bit 양자화로 VRAM 최적화
"""
import json
import math
import random
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ── 경로 설정 ──────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_PATH = "output/qwen-coder-finetune-v4"
TRAIN_PATH   = Path("data/train.jsonl")
OUTPUT_PATH  = Path("data/train_selected.jsonl")

# ── 하이퍼파라미터 ────────────────────────────────────
N_CLUSTERS  = 10
SAMPLE_RATE = 0.40
SEED        = 42
MAX_LENGTH  = 512


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def messages_to_text(messages: list[dict]) -> tuple[str, str]:
    instruction, response = "", ""
    for m in messages:
        if m["role"] == "user":
            instruction = m["content"]
        elif m["role"] == "assistant":
            response = m["content"]
    return instruction, response


def compute_perplexity(model, tokenizer, text: str) -> float:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH
    )
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()

    return math.exp(loss)


def compute_ifd_scores(data: list[dict], model, tokenizer) -> list[float]:
    scores = []
    for item in tqdm(data, desc="IFD 계산 중"):
        instruction, response = messages_to_text(item["messages"])

        full_text = f"{instruction}\n{response}"
        ppl_with = compute_perplexity(model, tokenizer, full_text)
        ppl_without = compute_perplexity(model, tokenizer, response)

        ifd = ppl_with / (ppl_without + 1e-8)
        scores.append(ifd)

    return scores


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    print("디바이스: cuda (4bit 양자화 모드)")

    # ── 데이터 로드 ───────────────────────────────────
    data = load_jsonl(TRAIN_PATH)
    print(f"전체 데이터: {len(data)}개")

    # ── Step 1: 임베딩 + K-Means 클러스터링 ──────────
    print("\n[Step 1] 임베딩 및 클러스터링...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    instructions = [messages_to_text(d["messages"])[0] for d in data]
    embeddings = embed_model.encode(
        instructions, batch_size=64, show_progress_bar=True
    )

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    print(f"클러스터 {N_CLUSTERS}개 생성 완료")

    del embed_model
    torch.cuda.empty_cache()

    # ── Step 2: 4bit 양자화로 모델 로드 + IFD 계산 ───
    print("\n[Step 2] v4 모델 로드 (4bit 양자화) 및 IFD 계산...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True
    )
    base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()    

    ifd_scores = compute_ifd_scores(data, model, tokenizer)

    del model, base_model
    torch.cuda.empty_cache()

    # ── Step 3: 클러스터별 상위 40% 샘플링 ───────────
    print(f"\n[Step 3] 클러스터별 상위 {int(SAMPLE_RATE*100)}% 선별...")

    selected = []
    for c in range(N_CLUSTERS):
        indices = [i for i, label in enumerate(cluster_labels) if label == c]
        indices.sort(key=lambda i: ifd_scores[i], reverse=True)
        top_k = max(1, int(len(indices) * SAMPLE_RATE))
        selected.extend([data[i] for i in indices[:top_k]])

    print(f"선별 완료: {len(data)}개 → {len(selected)}개 ({len(selected)/len(data)*100:.1f}%)")

    # ── 저장 ─────────────────────────────────────────
    OUTPUT_PATH.write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in selected),
        encoding="utf-8"
    )
    print(f"저장 완료 → {OUTPUT_PATH}")

    # 태스크별 통계
    tasks = {}
    for d in selected:
        t = d.get("task", "unknown")
        tasks[t] = tasks.get(t, 0) + 1
    print("\n태스크별 분포:")
    for t, cnt in sorted(tasks.items()):
        print(f"  {t}: {cnt}개")


if __name__ == "__main__":
    main()