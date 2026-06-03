# Coder LLM Finetune

GPT-4o-mini(유료 API)로 동작하는 [ai-coding-test-assistant](https://github.com/HyeonBin0118/ai-coding-test-assistant)를 도메인 특화 파인튜닝 모델로 교체하는 실험 프로젝트.
"작은 모델도 좁은 도메인에서는 대형 API를 대체할 수 있는가"를 데이터로 검증한다.

---

## 프로젝트 배경

ai-coding-test-assistant는 프로그래머스 문제를 자동 인식해 힌트/접근법/정답을 제공하는 데스크톱 위젯이다. GPT-4o-mini API를 사용하기 때문에 사용할수록 비용이 발생한다. 같은 도메인에 특화된 작은 모델로 교체했을 때 비슷한 품질이 나오는지 검증하고, 가능하다면 API 비용 없이 동일한 기능을 로컬에서 제공하는 것이 목표다.

---

## 기술 스택

| 분류 | 사용 기술 |
|---|---|
| 언어 | Python 3.11 |
| 베이스 모델 | Qwen2.5-Coder-7B-Instruct |
| 파인튜닝 기법 | QLoRA (4bit 양자화 + LoRA) |
| 학습 라이브러리 | transformers, peft, trl, accelerate, bitsandbytes |
| 데이터 수집 | Playwright (프로그래머스 API + DOM 파싱) |
| 데이터 생성 | GPT-4o-mini (Knowledge Distillation) |
| GPU | NVIDIA RTX 3060 (VRAM 12GB) |

---

## 전체 흐름

```
프로그래머스 문제 수집
    ↓
GPT-4o-mini로 힌트/접근법/정답 생성 (Knowledge Distillation)
    ↓
QLoRA 파인튜닝 (Qwen2.5-Coder-7B-Instruct)
    ↓
GPT-4o-mini vs 파인튜닝 모델 비교 평가
```

---

## Phase 1 — 데이터 수집

### 문제 URL 수집

프로그래머스 코딩 테스트 페이지를 처음엔 Playwright로 DOM을 직접 파싱하려 했으나, Level 필터가 JavaScript로 동작해 headless 환경에서 적용되지 않는 문제가 있었다. 브라우저 네트워크 탭을 분석해 내부 API(`/api/v2/school/challenges/`)를 발견하고, Playwright로 세션을 유지한 채 API를 직접 호출하는 방식으로 전환했다.

- 수집 대상: Level 1~2 코딩 문제
- 수집 결과: 274개

### 문제 본문 파싱

각 문제 URL을 Playwright로 접근해 문제 설명, 제한사항, 입출력 예시를 추출했다. SQL 문제는 파인튜닝 데이터로 적합하지 않아 제목 키워드 + 본문 패턴으로 필터링했다.

- 파싱 결과: 228개 (SQL 46개 제외)

### 학습 데이터 생성

GPT-4o-mini로 각 문제의 힌트/접근법/정답 코드를 생성했다. 이 방식은 GPT-4o-mini의 응답을 "교사"로 삼아 작은 모델을 학습시키는 Knowledge Distillation 구조다.

- 생성 결과: 227개 문제 × 3가지 (힌트/접근법/정답) = 681개 샘플
- train / val 분리: 612개 / 69개 (90/10)

---

## Phase 2 — QLoRA 파인튜닝

### 환경

| 항목 | 값 |
|---|---|
| GPU | RTX 3060 (VRAM 12GB) |
| 베이스 모델 | Qwen2.5-Coder-7B-Instruct (15GB) |
| 양자화 | 4bit NF4 |
| LoRA r | 16 |
| LoRA alpha | 32 |
| 학습 가능 파라미터 | 40,370,176 (전체의 0.53%) |

### 학습 설정

| 항목 | 값 |
|---|---|
| epochs | 3 |
| batch size | 2 |
| gradient accumulation | 4 (effective batch 8) |
| learning rate | 2e-4 |
| scheduler | cosine |
| max sequence length | 2048 |

### 학습 결과

| 지표 | 시작 | 종료 |
|---|---|---|
| train loss | 1.077 | 0.471 |
| val loss | 0.619 | 0.552 |
| token accuracy | 74.7% | 87.8% |
| 학습 시간 | - | 75분 |

loss가 지속적으로 감소했고, val loss도 함께 내려가 과적합 없이 학습이 잘 진행됐다.

---

## Phase 3 — 비교 평가

> 진행 중

동일한 문제 10개(Level 1: 5개, Level 2: 5개)에 대해 GPT-4o-mini와 파인튜닝 모델의 응답을 비교한다.

---

## 트러블슈팅

**Windows cp949 인코딩 에러**

trl 라이브러리 내부에서 jinja 템플릿 파일을 시스템 기본 인코딩(cp949)으로 읽으려다 실패하는 문제. `chat_template_utils.py`의 `.read_text()` 호출에 `encoding="utf-8"` 파라미터를 추가하는 패치 스크립트로 해결했다.

**BFloat16 미지원 에러**

RTX 3060은 BFloat16을 지원하지 않아 `_amp_foreach_non_finite_check_and_unscale_cuda` 에러 발생. `fp16=False, bf16=False`로 설정해 mixed precision을 비활성화하고 float32로 학습해 해결했다. 속도는 다소 느려졌으나 안정적으로 학습 완료.

---

## 프로젝트 구조

```
coder-llm-finetune/
├── data/
│   ├── crawl_problems.py      # 프로그래머스 문제 URL 수집
│   ├── fetch_problems.py      # 문제 본문 파싱 + SQL 필터링
│   ├── generate_dataset.py    # GPT-4o-mini로 학습 데이터 생성
│   ├── convert_to_jsonl.py    # Hugging Face 학습 포맷 변환
│   ├── problem_urls.json      # 수집된 문제 URL 274개
│   ├── problems_parsed.json   # 파싱된 문제 228개
│   ├── dataset_raw.json       # GPT 생성 데이터 227개
│   ├── train.jsonl            # 학습셋 612개
│   └── val.jsonl              # 검증셋 69개
├── output/
│   └── qwen-coder-finetune/   # 파인튜닝된 LoRA 가중치
├── evaluation/
│   └── results.json           # 비교 평가 결과
├── train.py                   # QLoRA 학습 스크립트
├── evaluate.py                # GPT vs 파인튜닝 모델 비교
├── patch.py                   # trl 인코딩 패치
└── PLAN.md
```

---

## 실행 방법

```cmd
conda create -n finetune_env python=3.11 -y
conda activate finetune_env
pip install playwright httpx openai python-dotenv tqdm
playwright install chromium
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft trl accelerate bitsandbytes datasets
```

`.env` 파일 생성:
```
OPENAI_API_KEY=sk-...
```

```cmd
# 1. 문제 수집
python data/crawl_problems.py

# 2. 문제 파싱
python data/fetch_problems.py

# 3. 학습 데이터 생성
python data/generate_dataset.py

# 4. jsonl 변환
python data/convert_to_jsonl.py

# 5. trl 패치 (Windows 필수)
python patch.py

# 6. 파인튜닝
python train.py

# 7. 비교 평가
python evaluate.py
```

---

## 동작 환경

- Windows 10 / 11
- NVIDIA GPU (VRAM 12GB 이상 권장)
- CUDA 11.8
- Python 3.11

## 라이선스

MIT