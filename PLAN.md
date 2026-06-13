# 🗺️ Coder LLM Finetune — 개발 계획 및 회고

이 문서는 실험 설계 의도, 각 버전의 변경 이유, 실패 분석, 회고를 기록한다.
정량 결과와 사용법은 [README.md](./README.md) 참조.

## 프로젝트 목표

GPT-4o-mini(유료 API)로 동작하는 ai-coding-test-assistant를 도메인 특화 파인튜닝 모델로 교체할 수 있는지 검증한다.

**핵심 질문:** "작은 모델(7B)도 좁은 도메인에서 대형 API를 대체할 수 있는가?"

---

## 진행 상황

| Phase | 내용 | 상태 |
|---|---|---|
| 1 | 데이터 수집 및 전처리 | ✅ |
| 2 | 파인튜닝 환경 구성 | ✅ |
| 3 | 비교 실험 (v1~v4) | ✅ |
| 4 | 추가 개선 시도 및 한계 확인 | ✅ |
| 5 | GPT-4o-mini vs v4 비교 평가 | ✅ |
| 6 | Evol-Instruct 데이터 확장 | ✅ |
| 7 | IFD + K-Means 선별 및 v5 학습 | ✅ |
| 8 | DPO 데이터 생성 및 학습 | 🔄 |

---

## Phase 1 — 데이터 수집

### 문제 URL 수집
초기 시도는 Playwright로 프로그래머스 페이지의 DOM을 직접 파싱하는 방식이었으나, Level 필터가 JavaScript로 동작해 headless 환경에서 적용되지 않았다. 브라우저 네트워크 탭을 분석해 내부 API(`/api/v2/school/challenges/`)를 발견하고, Playwright로 세션을 유지한 채 API를 직접 호출하는 방식으로 전환했다.
- 수집 결과: 274개

### 문제 본문 파싱
각 문제 URL을 Playwright로 접근해 설명·제한사항·입출력 예시를 추출. SQL 문제는 파인튜닝에 부적합해 필터링.
- 파싱 결과: 228개 (SQL 46개 제외)

### 학습 데이터 생성 (v1용)
GPT-4o-mini로 힌트/접근법/정답 코드를 생성. GPT의 응답을 "교사"로 삼는 Knowledge Distillation 구조.
- 227개 문제 × 3가지 = 681개 샘플

---

## Phase 2 — 파인튜닝 환경

모든 실험에서 학습 설정은 동일하게 유지하고, **데이터와 프롬프트만 변경**해 변수의 영향을 분리 측정했다.

| 항목 | 값 |
|---|---|
| GPU | RTX 3060 (12GB) |
| 베이스 모델 | Qwen2.5-Coder-7B-Instruct |
| 양자화 | 4bit NF4 |
| LoRA r / alpha / dropout | 16 / 32 / 0.05 |
| 학습 가능 파라미터 | 40,370,176 (0.53%) |
| epochs | 3 |
| learning rate | 2e-4 (cosine) |

---

## Phase 3 — 비교 실험 (v1 ~ v4)

### v1 — 베이스라인
GPT-4o-mini 생성 데이터를 그대로 학습.
- 힌트/접근법: GPT 대비 60~70% 수준 사용 가능
- 정답 코드: **완전 실패** (로직 오류, 파라미터 불일치, `return -1` 회피 다수)

**원인:** (1) GPT 생성 정답 코드 자체가 틀린 경우가 있어 오염된 교사 신호로 학습, (2) solution 샘플 227개로 절대 부족.

**교훈:** train/val loss가 좋아도 실제 태스크 품질을 보장하지 않는다.

### v2 — GitHub 통과 코드로 교체
정답 코드만 GitHub 실제 통과 코드로 교체 (681 → 2,783, 4배).
- val loss 39% 감소했으나 **실제 코드 품질은 v1과 차이 없음**

**원인:** `extract_solution_func` 정규식이 빈 줄을 만나면 함수 추출을 중단하는 버그. 학습 데이터 상당수가 잘린 코드였음.

**교훈:** 데이터 양만 늘려서는 부족하다. 품질이 더 중요하다.

### v3 — 정규식 버그 수정
함수 추출 로직을 들여쓰기 기반으로 교체 후 재수집 (2,340개).
- val loss 추가 22% 감소했으나 **파라미터 오류 지속**

**원인:** 모델이 일관되게 잘못된 파라미터명 사용 (`def solution(signals)` → `def solution(n, k)`). 학습 프롬프트에 함수 시그니처 정보가 없어 문제 설명만으로 추론해야 했음.

**교훈:** 모델이 학습할 정보가 프롬프트에 충분히 포함됐는지 점검해야 한다.

### v4 — 함수 시그니처 프롬프트 추가
solution 프롬프트에 함수 시그니처 한 줄 추가.
- val loss 추가 23% 감소, **파라미터 오류 완전 해결**

| 문제 | v3 | v4 |
|---|---|---|
| 가장 많이 받은 선물 | ❌ | ✅ `(friends, gifts)` |
| 붕대 감기 | ❌ | ✅ `(bandage, health, attacks)` |
| 이웃한 칸 | ❌ | ✅ `(board, h, w)` |
| 데이터 분석 | ❌ | ✅ `(data, ext, val_ext, sort_by)` |

**교훈:** 프롬프트 설계 한 줄이 학습 효율에 결정적으로 기여한다.

### v4의 잔존 한계
파라미터는 정확하나 함수 내부 로직이 틀린 케이스가 남음:
- 반환값 타입 불일치 (숫자 대신 이름 반환)
- 시간 누적 처리 로직 누락
- 미정의 변수(`dh, dw, color, level`) 사용
- 로직 완전 오류

**추정 원인:** (1) GitHub 수집 코드 자체의 품질 한계(실제 통과 미검증), (2) 7B 모델 일반화 한계.

---

## Phase 4 — 추가 개선 시도 및 한계 확인

### 시도 1 — 학습 데이터 품질 필터링
5줄 미만 / `return` 없음 / `return -1`만 / 미정의 변수 사용 코드 제거.
- 2,340개 → 2,030개 (13% 필터링)

### 시도 2 — description 전체 사용
800자 자르기 → 전체(평균 1,500~3,000자) 사용 시도.

### 중단 이유 — RTX 3060 12GB VRAM 한계
| 설정 | 결과 |
|---|---|
| description 전체 + batch=2 | OOM (42GB 필요) |
| description 1500자 + batch=2 | OOM |
| description 800자 + batch=1, grad_accum=8 | 학습 가능하나 약 100시간 소요 |

**판단:** 무작정 데이터를 늘리는 대신 논문 기반 데이터 선별 전략(Phase 6~7)으로 방향 전환.

---

## Phase 5 — GPT-4o-mini vs v4 비교 평가

동일 문제 10개로 정량 비교. (결과 수치는 README 참조)

**결론:** 힌트 태스크는 응답 속도·비용 측면에서 실용 가능. 코드 생성은 7B 모델 한계로 완전 대체 실패. GPT 완전 대체에는 더 큰 모델(13B+) 또는 코드 실행 기반 데이터 검증 파이프라인 필요.

---

## Phase 6 — Evol-Instruct 데이터 확장

### 근거 논문
**Data-efficient LLM Fine-tuning for Code Generation** (arXiv:2504.12687, 2025)

핵심 인사이트: 대량 합성 데이터엔 저품질 샘플이 40~60% 섞여있어, 전체를 학습시키기보다 고품질 40%만 선별하는 게 더 효율적. (풀 데이터 66.1% → 40% 선별 66.9%)

### Evol-Instruct 방식
기존 227개 문제를 GPT-4o-mini로 3가지 변형:
- 제약 추가 ("반복문 없이 풀어라")
- 규모 확장 (입력 크기 100배)
- 재귀 변환 ("재귀 함수만 사용")

**결과:** 681개 + 변형 1,362개 = 총 3,848개 (train 3,463 / val 385)

---

## Phase 7 — IFD + K-Means 선별 및 v5 학습

### 논문 3단계 방법론
1. **K-Means 클러스터링(k=10):** sentence-transformer 임베딩으로 10개 군집 분류, 유형별 분포 유지
2. **IFD 점수 계산:** `IFD(C|I) = PPL(C|I) / PPL(C)`, 높을수록 고난도·고가치 샘플
3. **상위 40% 선별:** 각 클러스터에서 IFD 높은 순 40% 선별

**선별 결과:** 3,463개 → 1,381개 (39.9%) / approach 195, hint 213, solution 973

### 환경 적응
논문은 A100 4장(320GB) 기준. RTX 3060 12GB에서는 IFD 계산 시 4bit 양자화 적용. 양자화는 perplexity 비율 계산에 영향 없어 방법론 핵심은 동일 유지. IFD 계산 약 41분 소요.

### v5 학습 설정 변경
| 항목 | v4 | v5 |
|---|---|---|
| 학습 데이터 | 2,796 | 1,381 (IFD 선별) |
| max seq length | 2048 | 512 |
| batch size | 2 | 1 |
| gradient accumulation | 4 | 8 |
| gradient checkpointing | ❌ | ✅ |

### v5 학습 결과
- train loss: 1.23 → **0.13** (89% 감소)
- val loss: 0.451 → **0.222** (51% 감소)
- 학습 시간: **34시간 18분** (RTX 3060 기준)

**핵심 가설 E:** IFD 선별 40%로 학습한 v5가 전체 데이터 v4보다 코드 생성 pass@1이 높다.
→ **검증 예정 (compare_v4_v5.py)**

---

## Phase 8 — DPO 학습

### 배경
SFT는 "정답이 무엇인가"는 학습하지만 "정답 중 어느 쪽이 더 좋은가"는 구분 못함. DPO(Direct Preference Optimization)가 이 간극을 채움.

### 데이터 생성
GPT-4o-mini로 227개 chosen/rejected 쌍 생성:
- chosen: 깔끔하고 Pythonic한 풀이 (시간복잡도 최적화)
- rejected: 동작하지만 비효율적인 풀이 (반복문 중첩, 불필요한 변수)

**핵심 가설 F:** SFT(v5) 대비 DPO 적용 모델이 코드 효율성·가독성 평가에서 높은 점수를 받는다. → 미검증

---

## 핵심 발견 (회고)

1. **train/val loss는 실제 품질을 보장하지 않는다.** v1은 loss 0.552로 잘 학습된 듯 보였으나 출력은 사용 불가 수준이었다.

2. **데이터 품질 > 데이터 양.** v1→v2에서 양을 4배 늘렸으나 품질 문제로 개선 없었고, 정규식 한 줄 고친 v3에서 비로소 개선됐다.

3. **프롬프트 설계가 학습 효율에 결정적이다.** v3→v4에서 함수 시그니처 한 줄로 val loss 23% 감소 + 파라미터 오류 완전 해결.

4. **하드웨어 한계를 명확히 인식하고 적절한 시점에 전환한다.** 무작정 데이터를 늘리는 대신 논문 기반 선별 전략으로 방향 전환.

5. **태스크별로 실용성을 분리해 판단한다.** 코드 생성은 GPT 대체 실패했으나 힌트 태스크는 속도·비용 면에서 실용 가능했다.

6. **"한 번에 한 변수만 변경" 원칙은 실제로 효과가 있다.** 각 변경의 효과를 분리 측정할 수 있었다.

7. **논문 방법론을 제한된 환경에 맞게 적용하는 능력.** A100 기준 논문을 RTX 3060에서 재현하기 위해 4bit 양자화, gradient checkpointing 등 환경 최적화를 직접 해결.

---

## 트러블슈팅 요약

| 문제 | 해결 |
|---|---|
| Windows cp949 인코딩 에러 (trl jinja 로드 실패) | `.read_text()`에 `encoding="utf-8"` 강제 패치 (`patch.py`) |
| RTX 3060 BFloat16 미지원 | `fp16=False, bf16=False`로 mixed precision 비활성화 |
| GitHub API 응답 타입 불일치 | 응답이 list면 건너뛰는 가드 추가 |
| 정규식 함수 추출 버그 (v2 실패 원인) | 들여쓰기 기반 함수 범위 판단으로 교체 |
| 파라미터 오류 (v3 실패 원인) | 학습 프롬프트에 함수 시그니처 명시 |
| 추론 시 vLLM Windows 미지원 | transformers 직접 로드로 전환 |
| peft 버전 불일치 (`alora_invocation_tokens` 등) | `adapter_config.json`에서 신버전 전용 키 제거 |
| sentence-transformers 버전 충돌 (transformers 5.x) | sentence-transformers==2.7.0 + transformers==4.40.0 다운그레이드 |
| 4bit 모델 OOM (IFD 계산 시) | accelerate==0.27.2 다운그레이드 |
| gradient checkpointing + 4bit 충돌 | `prepare_model_for_kbit_training()` 적용 |

---

## 향후 가능한 방향 (이 프로젝트 범위 외)

- ai-coding-test-assistant에 v5/DPO 모델 연결 (FastAPI 로컬 서빙)
- GPT-4o-mini vs v5 응답속도/비용 비교 측정
- 모델 가중치 허깅페이스 Hub 업로드
- 코드 실행 기반 데이터 검증 파이프라인 구축
- 더 큰 모델/GPU 환경에서 description 전체 학습 재시도

---

## 참고 논문

- Data-efficient LLM Fine-tuning for Code Generation — arXiv:2504.12687 — https://github.com/Kyle-Lyu/data-efficient-finetuning
- Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in RAG — arXiv:2505.10792 — https://github.com/Pints-AI/Finetune-Bench-RAG

---

## 프로젝트 운영 원칙

- 각 Phase 완료 후 정량 측정 결과 기록
- 실패 실험과 원인 분석도 그대로 문서화
- "초기 가설 → 한계 → 대안 → 결과" 구조 유지
- 비교 실험은 한 번에 하나의 변수만 변경
- 하드웨어 한계 명확히 인식, 비용/효과 비율 떨어지면 전환