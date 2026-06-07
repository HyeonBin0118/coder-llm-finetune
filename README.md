# Coder LLM Finetune

GPT-4o-mini(유료 API)로 동작하는 [ai-coding-test-assistant](https://github.com/HyeonBin0118/ai-coding-test-assistant)를 도메인 특화 파인튜닝 모델로 교체할 수 있는지 검증한 실험 프로젝트.

**핵심 질문:** "작은 모델(7B)도 좁은 도메인에서 대형 API(GPT-4o-mini)를 대체할 수 있는가?"

**결과 요약:** 4차례의 반복 실험으로 val loss 63% 감소(0.552→0.203), token accuracy 7.3%p 향상(87.8%→95.1%) 달성. 파라미터 정확도 문제는 완전히 해결. 로직 정확도는 부분 성공. RTX 3060 12GB VRAM 한계로 v4를 최종 모델로 확정. GPT-4o-mini와의 정량 비교 결과, 힌트 태스크에서는 15% 빠른 응답과 유사한 방향성을 보였으나 코드 생성에서는 완전 대체에 실패.

---

## 프로젝트 배경

ai-coding-test-assistant는 프로그래머스 문제를 자동 인식해 힌트/접근법/정답을 제공하는 데스크톱 위젯이다. GPT-4o-mini API에 의존하기 때문에 사용할수록 비용이 발생한다. 같은 도메인에 특화된 작은 모델로 교체했을 때 비슷한 품질이 나오는지를 데이터로 검증하는 것이 이 프로젝트의 목표다.

---

## 기술 스택과 선택 이유

| 분류 | 기술 | 선택 이유 |
|---|---|---|
| 언어 | Python 3.11 | 머신러닝 생태계 표준 |
| 베이스 모델 | Qwen2.5-Coder-7B-Instruct | 코드 특화 + RTX 3060 12GB에서 QLoRA로 학습 가능한 최대 크기 + Instruct로 채팅 포맷 그대로 사용 |
| 파인튜닝 기법 | QLoRA (4bit 양자화 + LoRA) | 풀 파인튜닝은 RTX 3060으로 불가능. LoRA만으로도 메모리 부족. 4bit 양자화로 메모리 1/4 절감 |
| 학습 라이브러리 | transformers, peft, trl, accelerate, bitsandbytes | HuggingFace 표준 스택 |
| 데이터 수집 | Playwright + GitHub REST API | Playwright는 JS 렌더링 문제 해결 / GitHub API는 실제 통과 코드 수집 |
| 데이터 생성 | GPT-4o-mini (Knowledge Distillation) | 교사 모델로 활용해 학습 데이터 자동 생성 |
| GPU | RTX 3060 (12GB) | 로컬 학습 환경 |

---

## 전체 흐름

```
프로그래머스 문제 수집 (Level 1~2, 228개)
    ↓
힌트/접근법: GPT-4o-mini 생성 (Knowledge Distillation)
정답 코드: GitHub 공개 레포에서 실제 통과 코드 수집 (2,340개)
    ↓
QLoRA 파인튜닝 (Qwen2.5-Coder-7B-Instruct)
    ↓
v1 → v2 → v3 → v4: 한 번에 한 변수씩 변경하며 비교 평가
    ↓
GPT-4o-mini vs v4 정량 비교 (응답 시간 + 품질)
```

---

## Phase 1 — 데이터 수집

### 문제 URL 수집

초기 시도는 Playwright로 프로그래머스 페이지의 DOM을 직접 파싱하는 방식이었으나, Level 필터가 JavaScript로 동작해 headless 환경에서 적용되지 않았다. 브라우저 네트워크 탭을 분석해 내부 API(`/api/v2/school/challenges/`)를 발견하고, Playwright로 세션을 유지한 채 API를 직접 호출하는 방식으로 전환했다.

- 수집 대상: Level 1~2 코딩 문제
- 수집 결과: **274개**

### 문제 본문 파싱

각 문제 URL을 Playwright로 접근해 문제 설명, 제한사항, 입출력 예시를 추출했다. SQL 문제는 파인튜닝 데이터로 적합하지 않아 제목 키워드 + 본문 패턴으로 필터링했다.

- 파싱 결과: **228개** (SQL 46개 제외)

### 학습 데이터 생성 (v1용)

GPT-4o-mini로 각 문제의 힌트/접근법/정답 코드를 생성했다. GPT-4o-mini의 응답을 "교사"로 삼아 작은 모델을 학습시키는 Knowledge Distillation 구조다.

- 생성 결과: 227개 문제 × 3가지(힌트/접근법/정답) = **681개 샘플**
- train / val 분리: 612개 / 69개 (90/10)

---

## Phase 2 — 파인튜닝 환경

모든 실험에서 학습 설정은 동일하게 유지하고, **데이터와 프롬프트만 변경**해 변수의 영향을 분리해 측정했다.

| 항목 | 값 |
|---|---|
| GPU | RTX 3060 (VRAM 12GB) |
| 베이스 모델 | Qwen2.5-Coder-7B-Instruct (15GB) |
| 양자화 | 4bit NF4 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| 학습 가능 파라미터 | 40,370,176 (전체 7B의 0.53%) |
| epochs | 3 |
| batch size | 2 (gradient accumulation 4, effective 8) |
| learning rate | 2e-4 (cosine scheduler) |
| max sequence length | 2048 |
| 학습 시간 (v4 기준) | 약 5시간 45분 |

---

## Phase 3 — 비교 실험 (v1 ~ v4)

### 정량 결과 종합

| 실험 | 변경 내용 | 샘플 수 | val loss | token acc | 파라미터 정확도 | 로직 정확도 |
|---|---|---|---|---|---|---|
| v1 | 기준 (GPT 생성 데이터) | 681 | 0.552 | 87.8% | ❌ 실패 | ❌ 실패 |
| v2 | 정답을 GitHub 통과 코드로 교체 | 2,783 | 0.338 (-39%) | 92.0% (+4.2%p) | ❌ 실패 | ❌ 실패 |
| v3 | 정규식 버그 수정 후 재수집 | 2,796 | 0.263 (-22%) | 93.7% (+1.7%p) | ❌ 실패 | ❌ 실패 |
| v4 | 프롬프트에 함수 시그니처 추가 | 2,796 | **0.203 (-23%)** | **95.1% (+1.4%p)** | **✅ 해결** | ⚠️ 부분 실패 |

**총 변화 (v1 → v4):**
- val loss: 0.552 → 0.203 (**63% 감소**)
- token accuracy: 87.8% → 95.1% (**+7.3%p**)
- 파라미터 정확도: 실패 → 100% 해결

---

### v1 — 베이스라인

GPT-4o-mini가 생성한 힌트/접근법/정답 코드를 그대로 학습했다.

**결과:**
- 힌트/접근법: GPT 대비 60~70% 수준으로 사용 가능
- 정답 코드: **완전 실패** — 로직 오류, 파라미터 불일치, 엉뚱한 문제 풀이, `return -1` 회피 다수

**원인 분석:**
1. GPT가 생성한 정답 코드 자체가 틀린 경우가 있어 오염된 교사 신호로 학습됨
2. solution 태스크 학습 샘플이 227개로 절대적 부족

**교훈:** train/val loss가 좋아도 실제 태스크 품질을 보장하지 않는다. 학습 지표와 정성 평가는 별개로 봐야 한다.

---

### v2 — GitHub 통과 코드로 교체

힌트·접근법은 GPT-4o-mini 생성물을 유지하고, 정답 코드만 GitHub 레포에서 수집한 실제 통과 코드로 교체했다.

**데이터 수집:**
- GitHub REST API로 한국어/영어 검색 쿼리 10종 사용
- 132개 레포 탐색
- 190/228개 문제 커버 (83%)
- 정답 코드 2,327개 (문제당 평균 약 12개의 서로 다른 풀이)
- 데이터 양: 681 → 2,783 (**4배 증가**)

**결과:**
- val loss: 0.552 → 0.338 (39% 감소)
- token accuracy: 87.8% → 92.0% (+4.2%p)
- 그러나 **실제 코드 생성 품질은 v1과 거의 차이 없음**

**원인 분석:**
평가 결과를 정밀 검토하니, 생성된 코드들이 중간에 끊겨있었다. `extract_solution_func` 정규식이 빈 줄을 만나면 함수 추출을 중단하는 버그를 발견. 학습 데이터의 상당수가 불완전한 코드였던 것이다.

**교훈:** 데이터 양만 늘려서는 부족하다. **데이터 품질이 더 중요**하다.

---

### v3 — 정규식 버그 수정

함수 추출 로직을 들여쓰기 기반으로 교체하고 재수집했다.

- 수집 결과: 2,340개 (코드 길이도 정상화)

**결과:**
- val loss: 0.338 → 0.263 (22% 추가 감소)
- token accuracy: 92.0% → 93.7% (+1.7%p)
- 그러나 **파라미터 오류는 여전히 지속**

**원인 분석:**
평가 결과를 보니 모델이 일관되게 잘못된 파라미터명을 사용했다.

```
정답: def solution(signals): ...
출력: def solution(n, k): ...
```

학습 프롬프트를 다시 확인해보니 함수 시그니처 정보가 없었다. 모델이 문제 설명만 보고 파라미터를 추론해야 했던 것이 원인이다.

**교훈:** 모델이 학습할 정보가 프롬프트에 충분히 포함되어 있는지 점검해야 한다.

---

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

**결과:**
- val loss: 0.263 → 0.203 (23% 추가 감소)
- token accuracy: 93.7% → 95.1% (+1.4%p)
- **파라미터 오류 완전 해결**

**파라미터 정확도 비교:**

| 문제 | v3 (파라미터) | v4 (파라미터) |
|---|---|---|
| 가장 많이 받은 선물 | ❌ 틀림 | ✅ `(friends, gifts)` |
| 붕대 감기 | ❌ 틀림 | ✅ `(bandage, health, attacks)` |
| 이웃한 칸 | ❌ 틀림 | ✅ `(board, h, w)` |
| 데이터 분석 | ❌ 틀림 | ✅ `(data, ext, val_ext, sort_by)` |
| 달리기 경주 | ❌ 틀림 | ✅ `(players, callings)` |
| 서버 증설 횟수 | ❌ 틀림 | ✅ `(players, m, k)` |

**교훈:** 한 줄(함수 시그니처)만 프롬프트에 추가했는데 val loss가 추가로 23% 감소하고 파라미터 오류가 완전히 해결됐다. **프롬프트 설계가 학습 효율에 결정적으로 기여**한다.

---

### v4의 잔존 한계 (해결 못한 부분)

파라미터는 정확하나 함수 내부 로직이 틀린 케이스가 남아있다:

| 문제 | 잔존 오류 패턴 |
|---|---|
| 가장 많이 받은 선물 | 반환값 타입 불일치 (숫자 대신 이름 반환) |
| 붕대 감기 | 시간 누적 처리 로직 누락 |
| 이웃한 칸 | 미정의 변수(`dh, dw, color`) 사용 |
| 데이터 분석 | 인덱스 계산 오류 |
| 퍼즐 게임 챌린지 | 미정의 변수(`level`) 사용 |
| 도넛과 막대 그래프 | 로직 완전히 잘못됨 |

추정 원인:
1. GitHub 수집 코드 자체의 품질 한계 (실제 통과 여부 미검증)
2. 7B 모델 크기의 일반화 한계

---

## Phase 4 — 추가 개선 시도 및 한계 확인

v4의 잔존 로직 오류를 개선하기 위해 두 가지를 시도했다.

### 시도 1 — 학습 데이터 품질 필터링

GitHub 수집 코드 중 품질이 낮은 코드를 제거했다.

**필터링 기준:**
- 5줄 미만의 너무 짧은 코드
- `return` 문이 없는 미완성 코드
- `return -1`만 있는 포기 패턴
- 미정의 변수(`dh, dw, color, INF` 등)를 함수 시작 부분에서 사용하는 코드

**결과:** 2,340개 → 2,030개 (**310개, 13% 필터링**)

### 시도 2 — description 전체 사용

기존 `description[:800]`(800자 자르기) → 전체 description 사용 (평균 1,500~3,000자).

의도: 모델이 문제 조건을 더 충분히 학습.

### 중단 이유 — RTX 3060 12GB VRAM 한계

description 길이를 늘리자 시퀀스 길이가 증가해 메모리 사용량이 급증했다. 다양한 설정을 시도했으나 모두 한계에 부딪혔다.

| 설정 | 결과 |
|---|---|
| description 전체 + batch=2 | **OOM** (42GB 필요, 가용 12GB) |
| description 1500자 + batch=2 | **OOM** (42GB 필요) |
| description 800자 + batch=1, gradient_accumulation=8 | 학습 가능하나 **약 100시간 소요 예상** (1스텝당 약 7분) |

**판단:** batch size를 줄여 메모리는 확보할 수 있으나, 학습 시간이 비현실적으로 길어진다. RTX 3060 12GB는 7B 모델 + 긴 시퀀스 학습에 명확한 하드웨어 상한이 있다.

**결론:** v4를 최종 모델로 확정. description 길이 확장으로 얻을 수 있는 추가 개선보다 하드웨어 비용/시간이 더 크다고 판단했다.

---

## Phase 5 — GPT-4o-mini vs 로컬 v4 모델 비교 평가

v4 파인튜닝 모델을 실제 서비스에 연결하기 전에, GPT-4o-mini와 동일한 문제 10개(Level 1 5개, Level 2 5개)로 정량 비교를 진행했다.

평가 스크립트: `compare_eval.py` / 결과: `evaluation/compare_results.json`

### 응답 시간 비교

| 태스크 | GPT-4o-mini | 로컬 v4 | 비율 |
|---|---|---|---|
| 힌트 | 9,839ms | 8,399ms | 로컬 **15% 빠름** |
| 접근법 | 9,619ms | 32,837ms | GPT **3.4배 빠름** |
| 정답 코드 | 8,206ms | 11,703ms | GPT **43% 빠름** |

힌트 태스크에서는 로컬 모델이 더 빠르다. 접근법은 로컬 모델이 `max_new_tokens=512` 제한까지 길게 생성하는 경향이 있어 느리게 측정됐다.

### 품질 비교

| 항목 | GPT-4o-mini | 로컬 v4 |
|---|---|---|
| 힌트 품질 (정성 평가) | 3.9 / 5 | 2.9 / 5 |
| 파라미터 정확도 | 100% | 80% |
| 코드 완전 정답 | 3 / 10 | 0 / 10 |
| 코드 방향 맞음 | 6 / 10 | 2 / 10 |
| API 비용 | 유료 | **0원** |

**힌트 태스크:** 방향성은 맞으나 구체성이 부족하다. GPT 대비 약 74% 수준.

**정답 코드 태스크:** 파라미터는 대부분 맞추나(80%) 함수 내부 로직이 틀린 케이스가 많다. 완전 정답 0개로 GPT 완전 대체는 불가능하다.

### 결론

> 힌트 기능에서는 GPT 대비 15% 빠른 응답과 유사한 방향성을 보였으나, 코드 생성에서는 7B 모델의 한계로 완전 대체에 실패했다. 데이터 품질과 모델 크기가 코드 생성 품질의 핵심 변수임을 실험으로 확인했다.

GPT-4o-mini를 완전히 대체하려면 더 큰 모델(13B+) 또는 코드 실행 기반 데이터 검증 파이프라인이 필요하다. 현재 RTX 3060 12GB 환경에서는 한계가 있다. 다만 힌트 태스크에 한정하면, 응답 속도와 0원의 비용 측면에서 로컬 모델의 실용 가능성을 확인했다.

---

## 핵심 발견 (이 프로젝트에서 배운 것)

1. **train/val loss는 실제 품질을 보장하지 않는다.** v1은 val loss 0.552로 학습이 잘 된 것처럼 보였으나 실제 출력은 사용 불가 수준이었다. 학습 지표와 정성 평가는 항상 별개로 봐야 한다.

2. **데이터 품질 > 데이터 양.** v1(681개) → v2(2,783개)에서 양은 4배 늘렸으나 품질 문제(정규식 버그로 코드 잘림)로 실제 개선은 거의 없었다. 정규식 한 줄을 고친 v3에서 비로소 개선이 나타났다.

3. **프롬프트 설계가 학습 효율에 결정적이다.** v3 → v4에서 함수 시그니처 한 줄만 추가했는데 val loss가 23% 감소하고 파라미터 오류가 완전히 사라졌다. 모델이 학습할 정보가 프롬프트에 충분히 포함되었는지 점검하는 것이 중요하다.

4. **하드웨어 한계는 명확히 인식하고 적절한 시점에 종료해야 한다.** v5 시도는 가설은 합리적이었으나 RTX 3060 12GB로는 학습이 비현실적이었다. 더 큰 GPU 없이는 추가 개선의 비용/효과 비율이 떨어진다고 판단해 v4에서 종료했다.

5. **태스크별로 실용성을 분리해 판단해야 한다.** 코드 생성은 GPT를 대체하지 못했지만, 힌트 태스크는 속도·비용 측면에서 실용 가능성이 있었다. "전부 대체" 또는 "전부 실패"가 아니라 태스크 단위로 나눠 보면 작은 모델의 활용 지점이 보인다.

6. **"한 번에 한 변수만 변경" 원칙은 실제로 효과가 있다.** 각 버전에서 정확히 무엇을 바꿨는지 분리했기 때문에 어떤 변경이 어떤 효과를 만드는지 정확히 측정할 수 있었다.

---

## 트러블슈팅 요약

| 문제 | 해결 |
|---|---|
| Windows cp949 인코딩 에러 (trl jinja 로드 실패) | `.read_text()`에 `encoding="utf-8"` 강제 패치 (`patch.py`) |
| RTX 3060 BFloat16 미지원 | `fp16=False, bf16=False`로 mixed precision 비활성화, float32 학습 |
| GitHub API 응답 타입 불일치 (file dict vs dir list) | 응답이 list면 건너뛰는 가드 추가 |
| 정규식 함수 추출 버그 (v2 실패의 원인) | 들여쓰기 기반 함수 범위 판단 방식으로 교체 |
| 파라미터 오류 (v3 실패의 원인) | 학습 프롬프트에 함수 시그니처 명시 |
| 추론 시 vLLM Windows 미지원 | transformers 직접 로드 방식으로 전환 (eval_env 분리) |
| peft 버전 불일치 (`alora_invocation_tokens` 등) | `adapter_config.json`에서 신버전 전용 키 제거 |

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
│   └── convert_to_jsonl.py      # Hugging Face 학습 포맷 변환
├── output/
│   ├── qwen-coder-finetune/     # v1 LoRA 가중치
│   ├── qwen-coder-finetune-v2/  # v2 LoRA 가중치
│   ├── qwen-coder-finetune-v3/  # v3 LoRA 가중치
│   └── qwen-coder-finetune-v4/  # v4 LoRA 가중치 (최종 모델)
├── evaluation/
│   ├── results_epoch3_baseline.json  # v1 비교 평가 결과
│   ├── results.json                  # v2/v3 비교 평가 결과
│   └── compare_results.json          # v4 vs GPT-4o-mini 비교 평가 결과
├── train.py                     # QLoRA 학습 스크립트
├── evaluate.py                  # GPT vs 파인튜닝 모델 비교 (학습 단계용)
├── compare_eval.py              # GPT vs v4 응답 시간/품질 비교 (Phase 5)
├── patch.py                     # trl 인코딩 패치
├── requirements.txt             # 의존성 (버전 고정)
└── PLAN.md
```

---

## 실행 방법

```cmd
conda create -n finetune_env python=3.11 -y
conda activate finetune_env
pip install -r requirements.txt
playwright install chromium
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

# 9. 비교 평가 (학습 단계)
python evaluate.py

# 10. GPT vs v4 정량 비교 (Phase 5)
python compare_eval.py
```

---

## 동작 환경

- Windows 10 / 11
- NVIDIA GPU (VRAM 12GB 이상 권장)
- CUDA 11.8
- Python 3.11

## 라이선스

MIT