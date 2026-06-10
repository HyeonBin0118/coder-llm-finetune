# 🗺️ Coder LLM Finetune — 개발 계획

## 프로젝트 목표

GPT-4o-mini(유료 API)로 동작하는 ai-coding-test-assistant를 도메인 특화 파인튜닝 모델로 교체할 수 있는지 검증한다.

**핵심 질문:** "작은 모델(7B)도 좁은 도메인에서 대형 API를 대체할 수 있는가?"

---

## 프로젝트 상태: 진행 중 🔄

**현재 모델:** v5 학습 중 (`output/qwen-coder-finetune-v5`)

**v4 결과 요약:**
- 4차례 반복 실험으로 val loss 63% 감소 (0.552 → 0.203)
- token accuracy 7.3%p 향상 (87.8% → 95.1%)
- **파라미터 정확도 완전 해결**
- 로직 정확도는 부분 성공 — RTX 3060 12GB VRAM 한계로 추가 개선 중

---

## 완료된 작업

### Phase 1 — 데이터 수집 및 전처리 ✅

- [x] 프로그래머스 문제 URL 수집 (274개, 내부 API 활용)
- [x] 문제 본문 파싱 + SQL 필터링 (228개)
- [x] GPT-4o-mini로 v1용 학습 데이터 생성 (681개)
- [x] GitHub REST API로 정답 코드 수집 (132개 레포, 2,340개)
- [x] 데이터셋 결합 및 jsonl 변환

### Phase 2 — 파인튜닝 환경 ✅

- [x] Qwen2.5-Coder-7B-Instruct 다운로드
- [x] CUDA 11.8 + transformers + peft + trl 환경 구성
- [x] QLoRA 설정 (r=16, alpha=32, dropout=0.05)
- [x] Windows 환경 이슈 해결 (cp949 인코딩 패치, BFloat16 미지원 대응)

### Phase 3 — 비교 실험 (v1~v4) ✅

| 실험 | 변경 내용 | 데이터 | val loss | acc | 평가 결과 |
|---|---|---|---|---|---|
| v1 | 기준 (GPT 생성 데이터) | 681 | 0.552 | 87.8% | 파라미터·로직 모두 실패 |
| v2 | 정답을 GitHub 통과 코드로 교체 | 2,783 | 0.338 | 92.0% | 정규식 버그로 코드 잘림 |
| v3 | 정규식 버그 수정 후 재수집 | 2,796 | 0.263 | 93.7% | 파라미터 오류 지속 |
| v4 | 프롬프트에 함수 시그니처 추가 | 2,796 | **0.203** | **95.1%** | **파라미터 해결**, 로직 부분 실패 |

- [x] v1 학습 및 평가 → 실패 원인 분석 (데이터 오염 + 양 부족)
- [x] v2 학습 및 평가 → 정규식 버그 발견
- [x] v3 학습 및 평가 → 파라미터 오류 패턴 발견
- [x] v4 학습 및 평가 → 파라미터 문제 해결 확인

### Phase 4 — 추가 개선 시도 및 한계 확인 ✅

v4 잔존 로직 오류 개선을 위해 두 가지 시도:

**시도 1 — 학습 데이터 품질 필터링**
- 5줄 미만 / `return` 없음 / `return -1`만 있는 패턴 / 미정의 변수 사용 코드 제거
- 결과: 2,340개 → 2,030개 (13% 필터링)

**시도 2 — description 전체 사용 (800자 → 전체)**
- 결과: **RTX 3060 12GB VRAM 한계 초과**

| 설정 | 결과 |
|---|---|
| description 전체 + batch=2 | OOM (42GB 필요) |
| description 1500자 + batch=2 | OOM (42GB 필요) |
| description 800자 + batch=1, grad_accum=8 | 학습 가능, 약 100시간 소요 |

**판단:** 하드웨어 한계로 추가 개선의 비용/효과 비율이 떨어진다고 판단. 논문 기반 데이터 선별 전략(Phase 6~7)으로 방향 전환.

### Phase 5 — 문서화 ✅

- [x] README 최종 정리 (정량 지표 + 가설/결과 분석 + 멈춘 이유)
- [x] PLAN.md 최종 정리
- [x] requirements.txt 의존성 버전 고정
- [x] .gitignore 정리 (output 폴더, 데이터 파일 제외)

### Phase 6 — Evol-Instruct 데이터 확장 ✅

**목표:** RTX 3060 환경에서 데이터 양을 늘리기 위해 기존 문제를 변형하여 학습 데이터 확장

**근거 논문:** Data-efficient LLM Fine-tuning for Code Generation (arXiv:2504.12687, 2025)
- 핵심 인사이트: 대량 합성 데이터엔 저품질 샘플이 40~60% 섞여있어 선별이 효과적

**논문 핵심 방법:**
- K-Means 클러스터링으로 데이터 분포 유지
- IFD(Instruction Following Difficulty) 점수로 복잡도 측정
- 전체 40% 선별로 풀 데이터 이상 성능 달성 (66.1% → 66.9%)
- Dynamic Pack 토크나이징으로 패딩 비율 36% → 15% 감소

**방법 (Evol-Instruct):**
- 기존 227개 문제 × 변형 3종(제약 추가 / 규모 확장 / 재귀 변환) × 샘플 2개(hint + solution)

**결과:**
- 기존 681개 + 변형 1,362개 = 총 3,848개 (dataset_v2.json + dataset_evol.json 병합)

**작업 목록:**
- [x] `data/evol_dataset.py` 작성
- [x] evol_dataset.py 실행 완료 → `data/dataset_evol.json` 생성 (1,362개)
- [x] 기존 dataset_v2.json + dataset_evol.json 병합 (`data/merge_dataset.py`)
- [x] train.jsonl / val.jsonl 재생성 (총 3,848개)

---

## 진행 중인 작업

### Phase 7 — IFD + K-Means 데이터 선별 및 v5 학습 🔄

**목표:** 논문 방식대로 고품질 샘플만 선별하여 v5 학습, 코드 생성 성능 향상 검증

**방법:**
1. sentence-transformer로 전체 데이터 임베딩
2. K-Means(k=10)로 클러스터링
3. 각 클러스터 내 IFD 점수 계산 (PPL(C|I) / PPL(C))
4. 클러스터별 상위 40% 샘플 선별
5. 선별된 데이터로 v5 QLoRA 학습

**핵심 가설 E:** 전체 데이터보다 IFD 기반 선별 40%로 학습한 v5가 v4보다 코드 생성 pass@1이 높다.

**작업 목록:**
- [x] `data/ifd_select.py` 작성 (IFD 계산 + K-Means 선별)
- [x] IFD 선별 실행 → `data/train_selected.jsonl` 생성 (3,463개 → 1,381개, 39.9%)
- [ ] v5 학습 실행 (진행 중 🔄)
- [ ] v4 vs v5 비교 평가 (compare_eval.py)

### Phase 8 — DPO 데이터 생성 및 학습 ⏳

**목표:** SFT만으로는 구분 못하는 "좋은 코드 vs 나쁜 코드" 선호도 학습으로 코드 품질 향상

**방법:**
- 각 문제마다 (chosen: 깔끔한 풀이 / rejected: 비효율적인 풀이) 쌍 생성
- GPT-4o-mini로 자동 생성
- v5 기반으로 DPO 학습 (trl DPOTrainer 활용)

**핵심 가설 F:** SFT(v5) 대비 DPO 적용 모델이 코드 효율성 및 가독성 평가에서 높은 점수를 받는다.

**작업 목록:**
- [x] `data/generate_dpo.py` 작성 (chosen/rejected 쌍 생성)
- [x] DPO 데이터 생성 실행 → `data/dataset_dpo.json` (227개)
- [ ] `train_dpo.py` 작성
- [ ] DPO 학습 실행
- [ ] v5 vs DPO 모델 비교 평가

---

## 핵심 가설과 검증 결과

**가설 A (v1→v2):** 정답 데이터를 실제 통과 코드로 바꾸고 양을 4배로 늘리면 코드 생성 품질이 크게 개선된다.
**결과:** ❌ **기각.** val loss는 개선됐으나(0.552→0.338) 정규식 버그로 코드가 잘려 실제 품질 변화 없음.

**가설 B (v2→v3):** 정규식 버그 수정 후 재수집하면 코드 품질이 개선된다.
**결과:** ⚠️ **부분 확인.** val loss 추가 개선(0.338→0.263). 그러나 파라미터 오류는 별도 원인으로 지속.

**가설 C (v3→v4):** 프롬프트에 함수 시그니처를 명시하면 파라미터 오류가 해결된다.
**결과:** ✅ **확인.** val loss 0.263→0.203, acc 93.7%→95.1%. **파라미터 문제 완전 해결.**

**가설 D (v4 이후):** 문제 설명 전체와 품질 필터링으로 로직 오류도 개선된다.
**결과:** ❓ **검증 불가.** RTX 3060 12GB VRAM 한계로 학습 시간이 비현실적(100시간+)이 되어 중단.

**가설 E (v5):** IFD 기반 선별 데이터(40%)로 학습한 v5가 v4보다 코드 생성 pass@1이 높다.
**결과:** ⏳ **검증 중.**

**가설 F (DPO):** SFT(v5) 대비 DPO 적용 모델이 코드 효율성 및 가독성 평가에서 높은 점수를 받는다.
**결과:** ⏳ **미검증.**

---

## v4 잔존 한계

파라미터는 정확하나 다음 패턴의 로직 오류가 남아있음:
- 미정의 변수 사용 (`dh, dw, color` 등)
- 문제 조건 일부 누락
- 반환값 타입 불일치 (이름 vs 숫자 등)

추정 원인:
1. GitHub 수집 코드 자체의 품질 한계 (실제 통과 여부 미검증)
2. 7B 모델 크기의 일반화 한계

---

## 핵심 발견 (프로젝트 회고)

1. **train/val loss가 좋아도 실제 태스크 품질은 보장되지 않는다.** v1에서 직접 경험. 학습 지표와 정성 평가는 반드시 별개로 봐야 한다.

2. **데이터 품질 > 데이터 양.** v1(681개) → v2(2,783개)에서 양은 4배 늘렸으나 품질 문제로 개선 없음. 정규식 한 줄 고친 v3에서 비로소 개선.

3. **프롬프트 설계 한 줄이 학습 효율에 결정적이다.** v3 → v4에서 함수 시그니처 한 줄 추가로 val loss 23% 감소 + 파라미터 오류 완전 해결.

4. **하드웨어 한계를 명확히 인식하고 적절한 시점에 종료하는 판단력.** 무작정 데이터를 늘리는 대신 논문 기반 데이터 선별 전략으로 방향 전환.

5. **"한 번에 한 변수만 변경" 원칙이 실제로 효과가 있다.** 각 변경의 효과를 분리해서 측정할 수 있었기에 어떤 변경이 어떤 효과를 만드는지 정확히 알 수 있었다.

6. **논문 방법론을 제한된 환경에 맞게 적용하는 능력.** A100 기준 논문을 RTX 3060에서 재현하기 위해 4bit 양자화, gradient checkpointing 등 환경 최적화를 직접 해결.

---

## 향후 가능한 방향 (이 프로젝트 범위 외)

- [ ] ai-coding-test-assistant에 v5/DPO 모델 연결 (FastAPI 로컬 모델 서빙)
- [ ] GPT-4o-mini vs v5 응답속도 / 비용 비교 측정
- [ ] 모델 가중치 허깅페이스 Hub 업로드
- [ ] 코드 실행 기반 데이터 검증 파이프라인 구축
- [ ] 더 큰 모델/GPU 환경에서 description 전체 학습 재시도

---

## 참고 논문 및 출처

- **Data-efficient LLM Fine-tuning for Code Generation**
  Weijie Lv et al., arXiv:2504.12687 (2025)
  https://arxiv.org/abs/2504.12687
  https://github.com/Kyle-Lyu/data-efficient-finetuning

- **Finetune-RAG: Fine-Tuning Language Models to Resist Hallucination in RAG**
  Zhan Peng Lee et al., arXiv:2505.10792 (2025)
  https://arxiv.org/abs/2505.10792
  https://github.com/Pints-AI/Finetune-Bench-RAG

---

## 프로젝트 운영 원칙

- 각 Phase 완료 후 정량 측정 결과를 README에 인라인 기록
- 실패 실험과 원인 분석도 그대로 문서화 (v1/v2/v3 모두)
- "초기 가설 → 한계 → 대안 → 결과" 구조 유지
- 비교 실험은 한 번에 하나의 변수만 변경
- 하드웨어 한계 명확히 인식, 비용/효과 비율 떨어지면 종료