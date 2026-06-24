# Coder LLM Finetune

GPT-4o-mini(유료 API)로 동작하는 [ai-coding-test-assistant](https://github.com/HyeonBin0118/ai-coding-test-assistant)를 도메인 특화 파인튜닝 모델로 교체할 수 있는지 검증하는 실험 프로젝트.

**핵심 질문:** "작은 모델(7B)도 좁은 도메인에서 대형 API(GPT-4o-mini)를 대체할 수 있는가?"

RTX 3060 12GB 단일 GPU 환경에서 Qwen2.5-Coder-7B-Instruct를 QLoRA로 파인튜닝하고, 데이터·프롬프트를 한 번에 한 변수씩 변경하며 성능 변화를 정량 측정했다. 논문 기반 데이터 선별 전략(IFD + K-Means)으로 v5를 만든 뒤, DPO·LoRA rank 증대·데이터 정제·데이터 규모 확장·중복 제거(v9)까지 총 6가지 추가 개선을 시도했다. **30문제로 평가셋을 확장한 결과 v5, v8, v9는 통계적으로 동급(43~57%) 성능**이었고, 이는 현재 조건(7B 모델, 1,300~1,700개 데이터)에서 데이터 조정만으로는 일정 구간을 벗어나기 어렵다는 것을 시사한다. 이에 **v5를 최종 모델로 확정**했다.

> 실험 설계 의도, 실패 분석, 회고 등 상세 기록은 [PLAN.md](./PLAN.md) 참조.

---

## 결과 요약

![버전별 성능 추이](evaluation/version_growth.png)

**10문제 평가 (학습 단계의 1차 검증용)**

| 항목 | v1 | v4 | v5 | DPO v1/v2 | v6 (rank32) | v7 (데이터정제) | v8 (데이터확장) | v9 (중복제거) |
|---|---|---|---|---|---|---|---|---|
| val/eval loss | 0.552 | 0.203 | 0.222 | - | 0.2096 | 0.2298 | 0.2343 | 0.262 |
| token accuracy | 87.8% | 91.7%* | **94.1%** | - | - | - | - | - |
| 파라미터 정확도 | 실패 | 8/10 | **10/10** | 9~10/10 | 10/10 | - | 9/10 | - |
| 코드 완전 정답 (10문제) | 0/10 | 0/10 | **3/10** | 2/10 | 2/10 | 평가 중단 | 2/10 | - |

*v4 token accuracy는 동일 `val.jsonl` 기준으로 재측정한 값(91.7%)이다. 과거 기록된 95.1%는 측정 방식이 명확하지 않아, `compute_token_accuracy.py`로 재현 가능한 값으로 대체했다.

**30문제 확장 평가 (최종 판단 기준 — 10문제 평가의 통계적 한계를 보완)**

| 모델 | 1차 평가 | 2차 평가(v9 포함) |
|---|---|---|
| v4 | 9/30 (30%) | - |
| **v5** | 17/30 (**57%**) | 14/30 (**47%**) |
| dpo | 13/30 (43%) | - |
| v8 | 16/30 (53%) | 13/30 (43%) |
| v9 | - | 14/30 (47%) |

*같은 모델(v5, v8)을 두 번 평가했는데 실행마다 약 10%p씩 다르게 나왔다. temperature 샘플링과 채점 기준의 미세한 차이 때문으로 추정되며, 절대값보다 모델 간 상대적 순위로 판단해야 한다.

- **4차례 반복 실험(v1→v4)** 으로 val loss 63% 감소, 파라미터 정확도 문제 완전 해결
- **GPT-4o-mini 정량 비교:** 힌트 태스크에서 응답 속도 15% 우위, 코드 생성에서는 완전 대체 실패
- **논문 기반 개선(v5):** Evol-Instruct 데이터 확장(681→3,848) + IFD+K-Means 40% 선별 → 코드 정답 0→3개(10문제 기준), 30문제 기준 가장 높은 정답률, 파라미터 정확도 100%, token accuracy 94.1% 달성
- **DPO/v6/v7/v8/v9 (10문제 기준):** 대부분 v5(3/10)에 못 미치는 2/10 또는 평가 중단. 응답 시간이 v5 대비 2.5~6배 급증하는 공통 부작용도 관찰
- **v5 재현 실험(v5-repro):** git에서 v5 원본 데이터를 복원해 동일 설정으로 재학습한 결과 eval loss가 0.222 → 0.2229로 거의 일치, v5의 학습 결과가 재현 가능함을 직접 확인
- **30문제로 확장 재평가한 결과, v5/v8/v9는 모두 43~57% 구간에서 겹쳐 통계적으로 거의 구분되지 않는다.** "추가 시도가 v5보다 못하다"는 10문제 기준의 결론은 표본 크기의 한계에서 비롯된 과대해석이었을 가능성이 높다. 다만 v4(30%)와의 차이는 일관되게 유지되어, 논문 기반 개선(v5)의 효과 자체는 노이즈가 아닌 것으로 판단된다
- **최종 결론:** 현재 조건(7B + QLoRA + 1,300~1,700개 데이터)에서는 데이터 양·품질을 조정하는 정도로는 40~55% 구간을 벗어나기 어렵다는 것을 확인했고, **v5를 최종 모델로 확정**했다

---

## 기술 스택

| 분류 | 기술 |
|---|---|
| 언어 | Python 3.11 |
| 베이스 모델 | Qwen2.5-Coder-7B-Instruct |
| 파인튜닝 | QLoRA (4bit NF4 + LoRA r=16) |
| 학습 | transformers, peft, trl, accelerate, bitsandbytes |
| 데이터 수집 | Playwright, GitHub REST API |
| 데이터 생성 | GPT-4o-mini (Knowledge Distillation) |
| 데이터 선별 | sentence-transformers, scikit-learn (IFD + K-Means) |
| GPU | RTX 3060 (12GB) |

---

## 파이프라인

```
프로그래머스 문제 수집 (Level 1~2, 228개)
    ↓
힌트/접근법: GPT-4o-mini 생성 (Knowledge Distillation)
정답 코드: GitHub 공개 레포 실제 통과 코드
    ↓
QLoRA 파인튜닝 → v1 → v2 → v3 → v4 (한 번에 한 변수씩 변경)
    ↓
GPT-4o-mini vs v4 정량 비교
    ↓
Evol-Instruct 데이터 확장 → IFD+K-Means 선별 → v5 (최고 성능)
    ↓
DPO 학습 (v1, v2) → 기각 (코드 정답률 하락)
    ↓
LoRA rank 증대 (v6) → 기각 (코드 정답률 하락, 응답시간 급증)
    ↓
학습 데이터 검증·정제 (v7) → 기각 (eval loss 악화, 응답시간 급증)
    ↓
학습 데이터 규모 확장 (v8, Level 3 추가 + GitHub 코드 AST 검증) → 진행 중
```

---

## 실험 결과

### 비교 실험 (v1 ~ v4)

학습 설정은 고정하고 데이터·프롬프트만 변경해 각 변수의 영향을 분리 측정했다.

| 실험 | 변경 내용 | 샘플 수 | val loss | token acc | 파라미터 | 로직 |
|---|---|---|---|---|---|---|
| v1 | 기준 (GPT 생성 데이터) | 681 | 0.552 | 87.8% | ❌ | ❌ |
| v2 | 정답을 GitHub 통과 코드로 교체 | 2,783 | 0.338 (-39%) | 92.0% | ❌ | ❌ |
| v3 | 정규식 버그 수정 후 재수집 | 2,796 | 0.263 (-22%) | 93.7% | ❌ | ❌ |
| v4 | 프롬프트에 함수 시그니처 추가 | 2,796 | **0.203 (-23%)** | **91.7%*** | **✅** | ⚠️ |

*v4 token accuracy 91.7%는 `compute_token_accuracy.py`로 동일 `val.jsonl` 기준 재측정한 값. v1~v3는 동일 방식으로 재측정하지 않은 과거 기록 값.

각 버전의 변경 의도와 실패 원인 분석은 [PLAN.md](./PLAN.md) 참조.

### GPT-4o-mini vs 로컬 v4

동일 문제 10개(Level 1×5, Level 2×5) 기준. 스크립트: `evaluation_scripts/compare_eval.py`

**응답 시간**

| 태스크 | GPT-4o-mini | 로컬 v4 | 비교 |
|---|---|---|---|
| 힌트 | 9,839ms | 8,399ms | 로컬 15% 빠름 |
| 접근법 | 9,619ms | 32,837ms | GPT 3.4배 빠름 |
| 정답 코드 | 8,206ms | 11,703ms | GPT 43% 빠름 |

**품질**

| 항목 | GPT-4o-mini | 로컬 v4 |
|---|---|---|
| 힌트 품질 (정성 5점) | 3.9 | 2.9 |
| 파라미터 정확도 | 100% | 80% |
| 코드 완전 정답 | 3 / 10 | 0 / 10 |
| API 비용 | 유료 | 0원 |

> 힌트 태스크는 응답 속도·비용 측면에서 실용 가능. 코드 생성은 7B 모델 한계로 완전 대체 실패.

### 논문 기반 개선 (v5, 최고 성능 모델)

**근거 논문:** [Data-efficient LLM Fine-tuning for Code Generation](https://arxiv.org/abs/2504.12687) (arXiv:2504.12687)

| 단계 | 방법 | 결과 |
|---|---|---|
| 데이터 확장 | Evol-Instruct (제약/규모/재귀 변형) | 681 → 3,848개 |
| 데이터 선별 | IFD 점수 + K-Means(k=10) 상위 40% | 3,463 → 1,381개 (39.9%) |
| v5 학습 | 선별 데이터로 QLoRA (34시간) | train loss 0.13, val loss 0.222, token accuracy 94.1% ✅ |

**v4 vs v5 비교 평가 결과** (동일 문제 10개 기준, `evaluation_scripts/compare_v4_v5.py`)

| 항목 | v4 | v5 |
|---|---|---|
| 파라미터 정확도 | 8/10 | **10/10** |
| 코드 완전 정답 | 0/10 | **3/10** |
| hint 응답 시간 | 8,153ms | 177,516ms* |
| solution 응답 시간 | 13,767ms | 119,721ms* |

*응답 시간 증가 원인: MAX_LENGTH=512 학습으로 EOS 생성 타이밍 미학습. 실서비스 적용 시 max_new_tokens=256으로 제한하면 해결 가능.

**v4 vs v5 token accuracy 재평가** (`evaluation_scripts/compute_token_accuracy.py`, 동일 `val.jsonl` 385개 기준)

| 항목 | v4 | v5 |
|---|---|---|
| token accuracy | 91.7% | **94.1%** |

기존 README/PLAN.md에 기록된 v1~v4 token accuracy는 계산 코드가 프로젝트에 남아있지 않아 재현이 불가능했다. 이에 `compute_token_accuracy.py`를 새로 작성해 v4, v5를 동일 기준(다음 토큰 예측 정확도, 4bit 양자화 forward pass)으로 재측정했다.

### DPO 학습 (v1, v2 — 기각)

v5 기반으로 chosen/rejected 데이터로 DPO 학습 진행. v1은 GPT 생성 chosen, v2는 chosen을 GitHub 검증 코드로 교체.

| 항목 | DPO v1 | DPO v2 |
|---|---|---|
| 학습 데이터 | chosen/rejected 204쌍 | chosen/rejected 124쌍 |
| rewards/accuracies | **0.925** | 0.7875 |
| 코드 완전 정답 (vs v5 3/10) | 2/10 | 2/10 |
| 응답 시간 | 4배 증가 | 4배 증가 |

**결론:** 선호도 학습(rewards/accuracies)과 실제 태스크 성능(pass@1)은 별개의 지표다. chosen 데이터를 GPT 생성 코드에서 GitHub 검증 코드로 바꿔도 결과가 동일하게 나와, 데이터 품질이 원인이 아니라 **베이스 모델(v5)의 코드 생성 능력 자체가 DPO를 적용하기엔 부족했다**는 결론에 도달했다. 상세 분석은 [PLAN.md](./PLAN.md) Phase 8~9 참조.

### LoRA rank 증대 시도 (v6 — 기각)

DPO 기각 후, 베이스 모델(SFT)의 코드 생성 능력 자체를 끌어올리기 위해 LoRA rank를 16→32로 증대(나머지 설정은 v5와 동일).

| 항목 | v5 | v6 |
|---|---|---|
| 학습 가능 파라미터 | 40,370,176 (0.53%) | 80,740,352 (1.05%) |
| eval loss | 0.222 | **0.2096** (개선) |
| 코드 완전 정답 | **3/10** | 2/10 (하락) |
| solution 응답 시간 | 26,378ms | **159,721ms** (6.1배 증가) |

eval loss는 개선됐지만 코드 정답률은 하락했고, 일부 문제에서 동일 코드 블록을 6~7번 반복 생성하는 버그도 발견됐다. **기각.**

### 학습 데이터 검증 및 정제 시도 (v7 — 기각)

DPO 데이터에서 발견했던 "GitHub 코드 27%가 helper 함수 의존으로 실행 불가" 문제를 SFT 학습 데이터(1,381개)에도 검증. AST 분석 결과 2.8%(27개)가 동일 문제로 깨져 있어 제외, 1,354개로 정제 후 재학습(LoRA rank는 v5와 동일하게 16 유지).

| 항목 | v5 | v7 |
|---|---|---|
| 학습 데이터 | 1,381 | 1,354 (27개 제외) |
| eval loss | 0.222 | **0.2298** (오히려 악화) |
| solution 응답 시간 | 26,378ms | **4~6배 증가** (v6과 동일 패턴) |

데이터 정제는 eval loss를 오히려 악화시켰고, v6과 동일한 응답 시간 급증 현상이 재현되어 코드 정답률 확인 전 평가를 중단했다. **기각.**

### 세 가지 독립 시도의 공통 패턴 → 데이터 규모 확장(v8)으로

DPO(선호도 학습), v6(LoRA rank), v7(데이터 품질)이라는 서로 다른 변수를 바꿨음에도 매번 같은 부작용(응답 시간 4~6배 급증, 코드 정답률 하락·정체)이 나타났다. 남은 미시도 변수인 **학습 데이터 규모 확장**을 다음 단계(v8)로 진행했다.

### 학습 데이터 규모 확장 (v8)

프로그래머스 Level 3까지 수집 범위를 넓혀(228→320개 문제) 데이터 규모를 키우고, GitHub 코드에 AST 검증을 새로 추가했다. IFD 선별 결과 1,381개(v5) → 1,685개(v8, 동일 비율 39.9%).

| 항목 | v5 | v8 |
|---|---|---|
| 학습 데이터 | 1,381 | 1,685 (+22%) |
| eval loss | 0.222 | **0.2343** (악화) |
| 코드 완전 정답 (10문제) | **3/10** | 2/10 (하락) |
| solution 응답 시간 | 26,378ms | 108,776ms (2.5배 증가) |

10문제 기준으로는 v6, v7과 동일한 패턴(eval loss 악화, 코드 정답률 하락, 응답 시간 급증)이 또 나타났다.

**repetition_penalty 검증:** v6~v8 공통으로 관찰된 "응답이 길어지고 반복 생성하는" 증상이 추론 설정 문제인지 확인하기 위해 v8에 `repetition_penalty=1.3`을 적용해 재평가했다. 응답 시간은 v5 수준(평균 30초대)으로 빨라졌지만, 코드 자체가 와해되는 현상(미완성 함수, 함수 시그니처 불일치)이 나타나 코드 정답률은 **0/10으로 오히려 악화**됐다. 추론 설정 조정만으로는 해결되지 않는다는 결론에 도달했다.

### v5 재현 실험과 30문제 확장 평가 — 결론을 뒤집은 검증

다섯 차례의 시도(DPO, v6, v7, v8, repetition_penalty)가 모두 v5를 못 넘는 상황에서, "v5가 우연히 좋게 나온 결과는 아닌가?"를 직접 검증했다.

**v5-repro:** `train_selected.jsonl`, `val.jsonl`이 v8 작업 중 이미 덮어써진 상태였으나, git 커밋(`fd2e98f`)에 v5 원본이 보존되어 있어 `git show`로 복원, 동일 설정으로 재학습했다.

| 항목 | v5 (원본) | v5-repro |
|---|---|---|
| eval loss | 0.222 | **0.2229** (거의 일치) |
| 학습 시간 | 34시간 18분 | 7시간 56분 |

eval loss가 거의 일치해, **v5의 학습 결과는 재현 가능함을 확인했다.**

**30문제 확장 평가:** 동시에 "10문제라는 평가셋 자체가 너무 작아 통계적 노이즈를 진짜 성능 차이로 오인했을 가능성"을 검증하기 위해, 평가셋을 30문제로 확장해 v4, v5, dpo, v8을 재평가했다.

| 모델 | 코드 완전 정답 (30문제) | 비율 |
|---|---|---|
| v4 | 9/30 | 30% |
| v5 | 17/30 | **57%** |
| dpo | 13/30 | 43% |
| v8 | 16/30 | **53%** |

10문제 기준으로 10%p 차이였던 v5 vs v8이, 30문제 기준으로는 4%p 차이로 좁혀져 **통계적으로 거의 구분되지 않는 수준**이 됐다. **"v8이 v5보다 못하다"는 기존 결론은 표본 크기의 한계에서 비롯된 과대해석이었을 가능성이 높다.** 다만 v4(30%)와의 차이는 30문제에서도 유지되어, v1→v5 구간의 개선(논문 기반 데이터 선별)은 노이즈가 아닌 실재하는 효과로 판단된다. v6, v7은 시간 제약상 30문제로 재검증하지 못해 동일한 재평가가 필요한 한계로 남는다.

### 데이터 중복 제거 및 경량 정제 (v9)

v8 데이터 파이프라인을 재실행하는 과정에서 1,601개의 IFD 선별 결과를 직접 검토했고, 두 가지 문제를 발견했다: (1) 같은 문제의 GitHub 풀이가 여러 개 중복 포함됨, (2) 일부 코드에 `print()` 디버깅 라인이 남아있음.

| 항목 | 정제 전(1,601개) | 정제 후(1,278개) |
|---|---|---|
| 중복 제목으로 제외된 solution | - | 323개 (30%) |
| print() 라인 제거 | - | 11개 |

중복 제거 비중이 압도적으로 컸다 — 인기 있는 쉬운 문제일수록 GitHub 풀이가 여러 개 중복 포함되어, 데이터의 "양"과 실제 "고유 문제 다양성"이 달랐다는 것을 확인했다.

| 항목 | v5 | v8 | v9 |
|---|---|---|---|
| 학습 데이터 | 1,381 | 1,685 | 1,278 (중복제거+정제) |
| eval loss | 0.222 | 0.2343 | **0.262** (가장 악화) |
| 학습 시간 | 34시간 18분 | 8시간 32분 | 6시간 11분 |

### v5 vs v8 vs v9 — 최종 30문제 동시 비교

| 항목 | v5 | v8 | v9 |
|---|---|---|---|
| solution 평균 응답시간 | 44,129ms | 58,064ms | 63,768ms |
| 코드 완전 정답 (30문제) | 14/30 (47%) | 13/30 (43%) | 14/30 (47%) |

응답 시간은 v5 < v8 < v9 순으로 점점 느려졌고(v9는 30문제 중 11개가 max_new_tokens를 거의 다 채움), eval loss가 가장 나빴던 v9가 응답 시간도 가장 느려 두 지표 간 일관성이 있었다. 그러나 **코드 완전 정답률은 v5·v9가 47%로 동률, v8이 43%로 셋 다 통계적으로 구분되지 않는 수준**이었다. (참고: 같은 모델을 두 번 평가했을 때도 10%p 정도 편차가 있어, 이 수치들은 절대값보다 상대적 위치로 해석해야 한다.)

### 최종 결론

DPO, v6, v7, v8, v9, repetition_penalty까지 6가지 추가 개선을 시도한 결과, **현재 조건(Qwen2.5-Coder-7B + QLoRA r=16 + 4bit 양자화 + 1,300~1,700개 데이터 + RTX 3060)에서는 데이터 양·품질을 조정하는 정도로는 코드 생성 정답률이 40~55% 구간을 벗어나지 못한다**는 결론에 도달했다. 이 구간을 넘으려면 모델 크기, 데이터 규모의 차수(10배 이상), 또는 코드 실행 기반 검증 같은 더 근본적인 변화가 필요하다고 판단해, **v5를 최종 모델로 확정**하고 추가 학습 시도는 이 프로젝트 범위에서 마무리했다.

상세 분석은 [PLAN.md](./PLAN.md) Phase 10~14 참조.

---

## 프로젝트 구조

```
coder-llm-finetune/
├── data/
│   ├── crawl_problems.py           # 프로그래머스 문제 URL 수집 (Level 1~3)
│   ├── fetch_problems.py           # 문제 본문 파싱 + SQL 필터링
│   ├── generate_dataset.py         # GPT-4o-mini 힌트/접근법/정답 생성
│   ├── collect_github.py           # GitHub 정답 코드 수집
│   ├── validate_github_solutions.py # GitHub 코드 AST 검증 (v8 신규)
│   ├── build_dataset.py            # GPT 힌트/접근법 + GitHub 정답 결합
│   ├── convert_to_jsonl.py         # 학습 포맷 변환
│   ├── evol_dataset.py             # Evol-Instruct 데이터 확장
│   ├── merge_dataset.py            # 데이터셋 병합
│   ├── ifd_select.py               # IFD + K-Means 선별
│   ├── generate_dpo.py             # DPO chosen/rejected 쌍 생성 (v1)
│   ├── rebuild_dpo_dataset.py      # DPO 데이터 재구성 (v2)
│   ├── finalize_dpo_v2.py          # DPO v2 데이터 정제
│   └── finalize_sft_data.py        # SFT 학습 데이터 정제 (v7)
├── output/                         # LoRA 가중치 (v1~v9, dpo)
├── evaluation/                     # 비교 평가 결과 JSON + 성능 그래프
├── evaluation_scripts/             # 비교 평가 스크립트 모음
│   ├── compare_eval.py             # GPT vs 로컬 정량 비교
│   ├── compare_v4_v5.py            # v4 vs v5 직접 비교
│   ├── compare_v4_v5_dpo.py        # v4 vs v5 vs DPO v1 비교
│   ├── compare_v4_v5_dpo_v2.py     # v5 vs DPO v2 비교
│   ├── compare_v5_v6.py            # v5 vs v6 비교
│   ├── compare_v5_v7.py            # v5 vs v7 비교
│   ├── compare_v5_v8.py            # v5 vs v8 비교 (10문제)
│   ├── compare_v8_reppenalty.py    # v8 repetition_penalty 검증
│   ├── compare_30problems.py       # v4/v5/dpo/v8 30문제 비교
│   ├── compare_v5_v8_v9_30problems.py # v5/v8/v9 30문제 최종 비교
│   └── compute_token_accuracy.py   # token accuracy 동일 기준 재평가
├── train_scripts/                  # 학습 스크립트 모음
│   ├── train.py                    # QLoRA 학습 (v1~v5)
│   ├── train_dpo.py                # DPO 학습 (v1)
│   ├── train_dpo_v2.py             # DPO 학습 (v2)
│   ├── train_v6.py                 # v6 학습 (rank 32)
│   ├── train_v7.py                 # v7 학습 (데이터 정제)
│   ├── train_v8.py                 # v8 학습 (데이터 규모 확장)
│   ├── train_v9.py                 # v9 학습 (중복 제거 + 경량 정제)
│   └── train_v5_repro.py           # v5 재현 실험
├── select_30_problems.py           # 30문제 평가셋 선정
├── patch.py                        # trl 인코딩 패치 (Windows)
├── requirements.txt
└── PLAN.md                         # 실험 설계·회고 상세 기록
```

---

## 실행 방법

```cmd
conda create -n finetune_env python=3.11 -y
conda activate finetune_env
pip install -r requirements.txt
playwright install chromium
```

`.env` 생성:
```
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp_...
```

```cmd
# v1~v4 파이프라인
python data/crawl_problems.py
python data/fetch_problems.py
python data/generate_dataset.py
python data/collect_github.py
python data/build_dataset.py
python data/convert_to_jsonl.py
python patch.py
python train_scripts/train.py
python evaluation_scripts/compare_eval.py

# v5 (논문 기반 개선, 최고 성능)
python data/evol_dataset.py
python data/merge_dataset.py
python data/ifd_select.py
python train_scripts/train.py
python evaluation_scripts/compare_v4_v5.py
python evaluation_scripts/compute_token_accuracy.py

# DPO (별도 환경, 기각된 시도)
conda create -n dpo_env python=3.11 -y
conda activate dpo_env
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.40.0 peft==0.10.0 trl==0.11.4 accelerate==0.27.2 bitsandbytes==0.43.0 datasets rich python-dotenv
python data/generate_dpo.py
python train_scripts/train_dpo.py
conda activate finetune_env
python evaluation_scripts/compare_v4_v5_dpo.py

# v6, v7 (기각된 시도)
python train_scripts/train_v6.py
python evaluation_scripts/compare_v5_v6.py
python data/finalize_sft_data.py
python train_scripts/train_v7.py
python evaluation_scripts/compare_v5_v7.py

# v8 (데이터 규모 확장)
python data/crawl_problems.py      # LEVELS=[1,2,3]로 수정 후 실행
python data/fetch_problems.py
python data/collect_github.py
python data/validate_github_solutions.py
python data/generate_dataset.py
python data/build_dataset.py       # GITHUB_PATH를 github_solutions_v2.json으로 수정
python data/convert_to_jsonl.py
python data/evol_dataset.py
python data/merge_dataset.py       # V2_PATH를 dataset_v3.json으로 수정
python data/ifd_select.py
python train_scripts/train_v8.py
python evaluation_scripts/compare_v5_v8.py
python evaluation_scripts/compare_v8_reppenalty.py

# v5 재현 실험 (v5 원본 데이터를 git에서 복원 후 재학습)
git show <v5_data_commit>:data/train_selected.jsonl > data/train_selected_v5_original.jsonl
git show <v5_data_commit>:data/val.jsonl > data/val_v5_original.jsonl
python train_scripts/train_v5_repro.py

# 30문제 확장 평가 (v4, v5, dpo, v8 일괄 비교)
python select_30_problems.py
python evaluation_scripts/compare_30problems.py

# v9 (데이터 중복 제거 + 경량 정제)
python data/clean_train_selected.py   # train_selected.jsonl 중복 제거 + print() 정리
python train_scripts/train_v9.py

# v5 vs v8 vs v9 최종 30문제 비교
python evaluation_scripts/compare_v5_v8_v9_30problems.py
```

---

## 동작 환경

- Windows 10 / 11
- NVIDIA GPU (VRAM 12GB 이상 권장)
- CUDA 11.8
- Python 3.11

---

## 참고 논문

- **Data-efficient LLM Fine-tuning for Code Generation** — Weijie Lv et al., [arXiv:2504.12687](https://arxiv.org/abs/2504.12687) ([code](https://github.com/Kyle-Lyu/data-efficient-finetuning))
- **Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in RAG** — Zhan Peng Lee et al., [arXiv:2505.10792](https://arxiv.org/abs/2505.10792) ([code](https://github.com/Pints-AI/Finetune-Bench-RAG))

---

## 라이선스

MIT