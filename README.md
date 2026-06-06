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
| 데이터 수집 | Playwright (프로그래머스 API + DOM 파싱), GitHub REST API |
| 데이터 생성 | GPT-4o-mini (힌트/접근법) + GitHub 공개 풀이 (정답 코드) |
| GPU | NVIDIA RTX 3060 (VRAM 12GB) |

---

## 전체 흐름

```
프로그래머스 문제 수집
    ↓
힌트/접근법: GPT-4o-mini 생성 (Knowledge Distillation)
정답 코드: GitHub 공개 레포에서 실제 통과 코드 수집
    ↓
QLoRA 파인튜닝 (Qwen2.5-Coder-7B-Instruct)
    ↓
데이터 품질/프롬프트 설계별 비교 평가 (v1 ~ v4)
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

### 학습 데이터 생성 (v1)

GPT-4o-mini로 각 문제의 힌트/접근법/정답 코드를 생성했다. GPT-4o-mini의 응답을 "교사"로 삼아 작은 모델을 학습시키는 Knowledge Distillation 구조다.

- 생성 결과: 227개 문제 × 3가지 (힌트/접근법/정답) = 681개 샘플
- train / val 분리: 612개 / 69개 (90/10)

---

## Phase 2 — 1차 파인튜닝 (v1)

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

---

## Phase 3 — v1 평가와 한계 발견

동일한 문제 10개(Level 1·2)에 대해 GPT-4o-mini와 v1 파인튜닝 모델의 응답을 비교했다.

### 결과

| 태스크 | v1 모델 품질 |
|---|---|
| 힌트 | 방향은 맞으나 구체성 부족 (GPT 대비 60~70% 수준) |
| 접근법 | 힌트와 유사. 사용 가능한 수준 |
| 정답 코드 | **실패** — 로직 오류, 파라미터 불일치, 엉뚱한 문제 풀이, `return -1` 회피 다수 |

### 원인 분석

1. **데이터 품질 오염** — 정답 코드를 GPT-4o-mini가 생성했는데, 이 코드 자체가 틀린 경우가 있었다.
2. **데이터 양 부족** — solution 태스크 학습 샘플이 227개뿐이라 코드 생성 일반화에 절대적으로 부족했다.
3. **에포크 부족 가능성** — 3 에포크에서 일반화가 덜 됐을 여지.

> **핵심 교훈:** train/val loss가 좋아도 실제 태스크 품질을 보장하지 않는다.

---

## Phase 4 — 데이터 재구성 및 반복 실험 (v2~v4)

v1 실패를 분석하며 네 차례의 개선 실험을 진행했다. 각 실험은 변수를 하나씩만 바꿔 원인을 분리했다.

### 실험 요약

| 실험 | 변경 내용 | 데이터 | val loss | acc | 평가 결과 |
|---|---|---|---|---|---|
| v1 | 기준 (GPT 생성 데이터) | 681 | 0.552 | 87.8% | 파라미터·로직 모두 실패 |
| v2 | 정답 코드를 GitHub 통과 코드로 교체 | 2,783 | 0.338 | 92.0% | 파라미터·로직 모두 실패 |
| v3 | 정규식 버그 수정 후 재수집 | 2,796 | 0.263 | 93.7% | 파라미터 실패, 로직 실패 |
| v4 | 프롬프트에 함수 시그니처 추가 | 2,796 | 0.203 | 95.1% | **파라미터 해결**, 로직 부분 실패 |

### v2 — GitHub 통과 코드로 교체

힌트·접근법은 GPT-4o-mini 생성물을 유지하고, 정답 코드만 GitHub 레포에서 수집한 실제 통과 코드로 교체했다.

- 검색 쿼리: 한국어/영어 10종
- 탐색 레포: 132개
- 수집 결과: 190/228개 문제 커버, 총 2,327개 정답 코드
- val loss 개선(0.552→0.338)에도 실제 출력 품질 변화 없음 — 추가 원인 발견

### v3 — 정규식 버그 수정

v2 실패 원인을 분석하던 중, `extract_solution_func` 정규식이 빈 줄을 만나면 함수 중간에서 끊기는 버그를 발견했다. 수정 후 재수집(2,340개)하고 재학습했다.

- val loss 추가 개선(0.338→0.263)
- 그러나 파라미터 오류 지속 — 또 다른 원인 발견

### v4 — 프롬프트에 함수 시그니처 추가

v1~v3에서 모델이 `solution(signals)` 대신 `solution(n, k)` 같은 엉뚱한 파라미터를 반복적으로 생성하는 패턴을 발견했다. 학습 프롬프트에 파라미터 정보가 없어 모델이 파라미터를 추론해야 했던 것이 원인이었다.

학습 데이터의 solution 프롬프트에 `함수 시그니처: def solution(signals):` 형태로 파라미터를 명시했다.

**결과:** 파라미터 문제 완전 해결. val loss도 추가 개선(0.263→0.203).

---

## 현재 한계

파라미터는 정확하게 생성하나, 함수 내부 로직이 틀린 경우가 남아있다. 이는 두 가지 원인으로 추정된다.

1. **GitHub 수집 코드의 품질** — 레포에 올라온 코드가 실제로 통과한 코드인지 검증하지 않고 수집했다.
2. **7B 모델 크기의 한계** — 복잡한 알고리즘 로직을 일반화하기에 파라미터 수가 부족할 수 있다.

---

## 트러블슈팅

**Windows cp949 인코딩 에러**

trl 라이브러리 내부에서 jinja 템플릿 파일을 시스템 기본 인코딩(cp949)으로 읽으려다 실패하는 문제. `chat_template_utils.py`의 `.read_text()` 호출에 `encoding="utf-8"` 파라미터를 추가하는 패치 스크립트(`patch.py`)로 해결했다.

**BFloat16 미지원 에러**

RTX 3060은 BFloat16을 지원하지 않아 `_amp_foreach_non_finite_check_and_unscale_cuda` 에러 발생. `fp16=False, bf16=False`로 설정해 mixed precision을 비활성화하고 float32로 학습해 해결했다.

**GitHub API: 파일 대신 디렉토리 응답**

`contents` API가 경로에 따라 단일 파일(dict)이 아닌 디렉토리 목록(list)을 반환하는 경우가 있어 `'list' object has no attribute 'get'` 에러 발생. 응답이 list면 건너뛰도록 가드를 추가했다. 레포 10개마다 중간 저장하고, 재실행 시 기존 결과를 이어받도록 했다.

**정규식이 함수 중간에서 끊기는 버그**

`extract_solution_func`의 정규식이 빈 줄을 만나면 함수 추출을 중단하는 버그. 들여쓰기 기반으로 함수 범위를 판단하는 방식으로 교체했다. 이 버그로 인해 v2 학습 데이터의 상당수가 불완전한 코드였음을 나중에 확인했다.

**파라미터 오류 — 프롬프트 설계 문제**

v1~v3에서 모델이 일관되게 엉뚱한 파라미터명을 사용하는 패턴이 발견됐다. 학습 프롬프트에 파라미터 정보가 없어 모델이 스스로 추론해야 했기 때문이다. solution 프롬프트에 `함수 시그니처: def solution(...):` 형태로 명시한 v4에서 해결됐다.

---

## 프로젝트 구조

```
coder-llm-finetune/
├── data/
│   ├── crawl_problems.py        # 프로그래머스 문제 URL 수집
│   ├── fetch_problems.py        # 문제 본문 파싱 + SQL 필터링
│   ├── generate_dataset.py      # GPT-4o-mini로 힌트/접근법/정답 생성 (v1)
│   ├── collect_github.py        # GitHub 공개 레포에서 정답 코드 수집
│   ├── build_dataset.py         # GPT 힌트/접근법 + GitHub 정답 결합
│   ├── convert_to_jsonl.py      # Hugging Face 학습 포맷 변환
│   ├── problem_urls.json        # 수집된 문제 URL 274개
│   ├── problems_parsed.json     # 파싱된 문제 228개
│   ├── dataset_raw.json         # GPT 생성 데이터 (v1)
│   ├── github_solutions.json    # GitHub 수집 정답 코드 2,340개
│   ├── dataset_v2.json          # 결합 데이터셋 2,796개
│   ├── train.jsonl              # 학습셋 2,516개
│   └── val.jsonl                # 검증셋 280개
├── output/
│   ├── qwen-coder-finetune/     # v1 LoRA 가중치
│   ├── qwen-coder-finetune-v2/  # v2 LoRA 가중치
│   ├── qwen-coder-finetune-v3/  # v3 LoRA 가중치
│   └── qwen-coder-finetune-v4/  # v4 LoRA 가중치
├── evaluation/
│   ├── results_epoch3.json      # v1 비교 평가 결과
│   ├── results.json             # v2/v3 비교 평가 결과
│   └── results_v4.json          # v4 비교 평가 결과
├── train.py                     # QLoRA 학습 스크립트
├── evaluate.py                  # GPT vs 파인튜닝 모델 비교
├── patch.py                     # trl 인코딩 패치
└── PLAN.md
```

---

## 실행 방법

```cmd
conda create -n finetune_env python=3.11 -y
conda activate finetune_env
pip install playwright httpx openai python-dotenv tqdm requests
playwright install chromium
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers peft trl accelerate bitsandbytes datasets
```

`.env` 파일 생성:
```
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp_...
```

```cmd
# 1. 문제 수집
python data/crawl_problems.py

# 2. 문제 파싱
python data/fetch_problems.py

# 3. 힌트/접근법/정답(v1) 생성
python data/generate_dataset.py

# 4. GitHub 정답 코드 수집
python data/collect_github.py

# 5. 데이터셋 구성 (GPT 힌트/접근법 + GitHub 정답 + 함수 시그니처)
python data/build_dataset.py

# 6. jsonl 변환
python data/convert_to_jsonl.py

# 7. trl 패치 (Windows 필수)
python patch.py

# 8. 파인튜닝 (train.py 내 OUTPUT_DIR 조정)
python train.py

# 9. 비교 평가
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