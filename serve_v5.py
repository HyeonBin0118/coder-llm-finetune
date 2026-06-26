"""
v5 모델을 OpenAI 호환 API(/v1/chat/completions)로 서빙하는 경량 FastAPI 서버.
vLLM 없이 transformers + bitsandbytes 4bit로 직접 구현 (Windows 호환).
ai-coding-test-assistant의 OpenAIClient와 동일한 인터페이스로 호출 가능.
"""
import asyncio
import json
import time
import uuid
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER_DIR = "output/qwen-coder-finetune-v5"
SERVED_MODEL_NAME = "qwen-coder-v5"
PORT = 8001

app = FastAPI()
model = None
tokenizer = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = SERVED_MODEL_NAME
    messages: list[ChatMessage]
    max_tokens: int = 1024
    temperature: float = 0.3
    stream: bool = False


def load_model():
    global model, tokenizer
    print(f"베이스 모델 로드 중: {BASE_MODEL_ID}")
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
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()
    print("모델 로드 완료, 서버 준비됨")


def build_prompt(messages: list[ChatMessage]) -> str:
    text = ""
    for msg in messages:
        if msg.role == "system":
            text += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
        elif msg.role == "user":
            text += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
        elif msg.role == "assistant":
            text += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


def generate_sync(prompt: str, max_tokens: int, temperature: float) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def stream_generate(prompt: str, max_tokens: int, temperature: float):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    kwargs = dict(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    for token_text in streamer:
        if token_text:
            yield token_text
    thread.join()


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    prompt = build_prompt(req.messages)
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    if req.stream:
        async def event_generator():
            loop = asyncio.get_event_loop()
            gen = await loop.run_in_executor(
                None, lambda: list(stream_generate(prompt, req.max_tokens, req.temperature))
            )
            for chunk in gen:
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": req.model,
                    "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            done_payload = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": req.model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    loop = asyncio.get_event_loop()
    content = await loop.run_in_executor(
        None, lambda: generate_sync(prompt, req.max_tokens, req.temperature)
    )

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)