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

## Phase 2 — 파인튜닝 환경

| 항목 | 값 |
|---|---|
| GPU | RTX 3060 (VRAM 12GB) |
| 베이스 모델 | Qwen2.5-Coder-7B-Instruct (15GB) |
| 양자화 | 4bit NF4 |
| LoRA r | 16 |
| LoRA alpha | 32 |
| 학습 가능 파라미터 | 40,370,176 (전체의 0.53%) |
| epochs | 3 |
| batch size | 2 (gradient accumulation 4, effective 8) |
| learning rate | 2e-4 (cosine scheduler) |
| max sequence length | 2048 |

이 설정은 v1~v4 동일하게 유지하고, **데이터와 프롬프트만 바꿔서** 변수의 영향을 분리해 측정했다.

---

## Phase 3 — 비교 실험 (v1 ~ v4)

각 실험은 변수를 하나씩만 바꿔 원인을 분리했다. 학습 지표(val loss, token accuracy)와 실제 코드 생성 품질을 함께 측정했다.

### 정량 결과 요약

| 실험 | 데이터 출처 | 샘플 수 | val loss | token acc | 파라미터 정확도 | 로직 정확도 |
|---|---|---|---|---|---|---|
| v1 | GPT 생성 | 681 | 0.552 | 87.8% | ❌ 실패 | ❌ 실패 |
| v2 | GitHub (정규식 버그) | 2,783 | 0.338 | 92.0% | ❌ 실패 | ❌ 실패 |
| v3 | GitHub (정규식 수정) | 2,796 | 0.263 | 93.7% | ❌ 실패 | ❌ 실패 |
| v4 | GitHub + 함수 시그니처 프롬프트 | 2,796 | **0.203** | **95.1%** | ✅ 해결 | ⚠️ 부분 실패 |

- val loss는 v1 → v4까지 **63% 감소** (0.552 → 0.203)
- token accuracy는 **+7.3%p 향상** (87.8% → 95.1%)

### v1 — 베이스라인

GPT-4o-mini가 생성한 힌트/접근법/정답 코드를 그대로 학습했다.

**결과:**
- 힌트/접근법: GPT 대비 60~70% 수준으로 사용 가능
- 정답 코드: **완전 실패** — 로직 오류, 파라미터 불일치, 엉뚱한 문제 풀이, `return -1` 회피 다수

**원인 분석:**
1. GPT가 생성한 정답 코드 자체가 틀린 경우가 있어 오염된 교사 신호로 학습됨
2. solution 태스크 학습 샘플이 227개로 절대적 부족

**교훈:** train/val loss가 좋아도 실제 태스크 품질을 보장하지 않는다.

### v2 — GitHub 통과 코드로 교체

힌트·접근법은 GPT-4o-mini 생성물을 유지하고, 정답 코드만 GitHub 레포에서 수집한 실제 통과 코드로 교체했다.

- GitHub REST API로 132개 레포 탐색
- 190/228개 문제 커버, 총 2,327개 정답 코드
- 데이터 양 **4배 증가** (681 → 2,783)

**결과:** val loss 0.552 → 0.338 (39% 감소). 그러나 실제 코드 생성 품질은 v1과 거의 차이 없음.

**원인 분석:** `extract_solution_func` 정규식이 빈 줄을 만나면 함수 추출을 중단하는 버그를 발견. 학습 데이터의 상당수가 불완전한 코드였다.

### v3 — 정규식 버그 수정

함수 추출 로직을 들여쓰기 기반으로 교체하고 재수집했다.

- 수집 결과: 2,340개 (이전 2,327개 대비 코드 길이도 정상화)

**결과:** val loss 0.338 → 0.263 (22% 추가 감소), token acc 92.0% → 93.7%. 그러나 여전히 파라미터 오류 지속.

**원인 분석:** 평가 결과를 분석하니 모델이 일관되게 엉뚱한 파라미터명을 사용했다. 예를 들어 `solution(signals)` 대신 `solution(n, k)` 같은 패턴. 학습 프롬프트에 파라미터 정보가 없어 모델이 추론해야 했던 것이 원인.

### v4 — 함수 시그니처 프롬프트 추가

학습 데이터의 solution 프롬프트에 함수 시그니처를 명시했다.

**프롬프트 변경:**
```
[v3] 다음 프로그래머스 문제의 Python 정답 코드를 작성해주세요.
     제목: 노란불 신호등
     문제 설명: ...

[v4] 다음 프로그래머스 문제의 Python 정답 코드를 작성해주세요.
     제목: 노란불 신호등
     함수 시그니처: def solution(signals)
     문제 설명: ...
```

**결과:** val loss 0.263 → 0.203 (23% 추가 감소), token acc 93.7% → 95.1%. **파라미터 오류 완전 해결.**

평가 케이스 비교:

| 문제 | v3 (파라미터) | v4 (파라미터) |
|---|---|---|
| 가장 많이 받은 선물 | ❌ 틀림 | ✅ `(friends, gifts)` |
| 붕대 감기 | ❌ 틀림 | ✅ `(bandage, health, attacks)` |
| 이웃한 칸 | ❌ 틀림 | ✅ `(board, h, w)` |
| 데이터 분석 | ❌ 틀림 | ✅ `(data, ext, val_ext, sort_by)` |
| 달리기 경주 | ❌ 틀림 | ✅ `(players, callings)` |
| 서버 증설 횟수 | ❌ 틀림 | ✅ `(players, m, k)` |

### v4의 잔존 한계

파라미터는 정확하나 함수 내부 로직이 틀린 케이스가 남아있다:

- 미정의 변수 사용 (`dh, dw, color` 등을 정의 없이 참조)
- 문제 조건의 일부를 누락한 로직 (예: 붕대 감기에서 시간 누적 처리 누락)
- 반환값 타입 불일치 (이름 대신 숫자를 반환해야 하는데 반대)

---

## Phase 4 — 추가 개선 시도 및 한계

v4에서 파라미터 문제는 해결했으나 로직 오류가 남아있어, 두 가지 개선을 시도했다.

### 시도한 개선

**1. 학습 데이터 품질 필터링**
- 5줄 미만의 너무 짧은 코드 제거
- `return` 문이 없는 미완성 코드 제거
- `return -1`만 있는 포기 패턴 제거
- 미정의 변수를 사용하는 코드 제거
- 결과: 2,340개 → 2,030개 (310개, 13% 필터링)

**2. 문제 설명(description) 전체 사용**
- 기존: `description[:800]` (800자로 잘라서 사용)
- 개선: 전체 description 사용 (평균 1,500~3,000자)
- 의도: 모델이 문제 조건을 더 충분히 학습

### 중단 이유 — RTX 3060 12GB VRAM 한계

description 길이를 늘리자 시퀀스 길이가 증가해 메모리 사용량이 급증했다.

| 설정 | 결과 |
|---|---|
| description 전체 + batch=2 | **OUT OF MEMORY** (42GB 시도, 가용 12GB) |
| description 1500자 + batch=2 | **OUT OF MEMORY** (42GB 시도) |
| description 800자 + batch=1, grad_accum=8 | 학습 가능하나 **약 100시간 소요 예상** (1스텝당 약 7분) |

**판단:** batch size를 줄여 메모리를 확보할 수는 있으나, 학습 시간이 비현실적으로 길어진다. RTX 3060 12GB는 7B 모델 + 긴 시퀀스 학습에 명확한 하드웨어 상한이 있다.

**결론:** v4를 최종 모델로 확정. description 길이 확장으로 얻을 수 있는 추가 개선보다 하드웨어 비용/시간이 더 크다고 판단했다.

---

## 핵심 발견

1. **train/val loss는 실제 품질을 보장하지 않는다.** v1은 val loss 0.552로 학습이 잘 된 것처럼 보였으나 실제 출력은 사용 불가 수준이었다.

2. **데이터 품질 > 데이터 양.** v1(681개) → v2(2,783개)에서 양은 4배 늘렸으나 품질 문제(코드 잘림)로 실제 개선은 거의 없었다.

3. **프롬프트 설계가 학습 효율에 크게 기여한다.** v3 → v4에서 한 줄(함수 시그니처)만 추가했는데 val loss가 23% 감소하고 파라미터 오류가 완전히 사라졌다.

4. **7B 모델 + 12GB VRAM 조합의 실용적 상한.** 시퀀스 길이를 늘리려면 batch size를 희생해야 하고, 그러면 학습 시간이 비현실적이 된다.

---

## 트러블슈팅

**Windows cp949 인코딩 에러**

trl 라이브러리가 jinja 템플릿 파일을 시스템 기본 인코딩(cp949)으로 읽으려다 실패. `chat_template_utils.py`의 `.read_text()` 호출에 `encoding="utf-8"` 파라미터를 추가하는 패치 스크립트(`patch.py`)로 해결.

**BFloat16 미지원 에러**

RTX 3060은 BFloat16을 지원하지 않아 `_amp_foreach_non_finite_check_and_unscale_cuda` 에러 발생. `fp16=False, bf16=False`로 mixed precision 비활성화하고 float32로 학습.

**GitHub API: 파일 대신 디렉토리 응답**

`contents` API가 단일 파일(dict) 대신 디렉토리 목록(list)을 반환하는 경우가 있어 가드 추가. 레포 10개마다 중간 저장, 재실행 시 이어받기 구현.

**정규식이 함수 중간에서 끊기는 버그 (v2 실패의 원인)**

`extract_solution_func`의 정규식이 빈 줄을 만나면 함수 추출을 중단. 들여쓰기 기반으로 함수 범위를 판단하는 방식으로 교체.

**파라미터 오류 (v3 실패의 원인)**

모델이 일관되게 엉뚱한 파라미터명을 사용. 학습 프롬프트에 파라미터 정보가 없어 모델이 스스로 추론해야 했기 때문. solution 프롬프트에 `함수 시그니처: def solution(...)` 명시한 v4에서 해결.

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
│   ├── train.jsonl              # 학습셋
│   └── val.jsonl                # 검증셋
├── output/
│   ├── qwen-coder-finetune/     # v1 LoRA 가중치
│   ├── qwen-coder-finetune-v2/  # v2 LoRA 가중치
│   ├── qwen-coder-finetune-v3/  # v3 LoRA 가중치
│   └── qwen-coder-finetune-v4/  # v4 LoRA 가중치 (최종 모델)
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

# 8. 파인튜닝
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