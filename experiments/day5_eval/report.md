# Day 5 — 파인튜닝 평가 · 오류 분석 · ablation

- **날짜**: 2026-07-22 · **평가 대상**: Day 4 QLoRA 어댑터([experiments/day4_qlora/adapter](../day4_qlora/)) on Qwen3-VL-8B
- **채점기**: [scripts/eval.py](../../scripts/eval.py) (Day 3와 동일) · **추론**: [run_zeroshot.py](../../scripts/run_zeroshot.py) `--adapter`
- **목적**: (1) val 564 정식 before/after 확정, (2) Day 1 진단셋으로 **분포 밖 일반화** 검증, (3) 잔여 오류 분석, (4) **epoch ablation**

## 1. 정식 평가 — val 564문항: 70.2% → **96.6%** (+26.4%p)

동일 val 564문항·동일 하네스. baseline = Day 3 zero-shot([day3_baseline](../day3_baseline/report.md)).

| 질문유형 | baseline | **adapter** | Δ | |
|---|---|---|---|---|
| **unit_convert** | 0% (0/73) | **96%** (70/73) | **+96%p** | 조→억 환산 |
| **rank_kth** | 57% (52/92) | **95%** (87/92) | **+38%p** | 조밀 순위 |
| narrow_compare | 79% (90/114) | 96% (109/114) | +17%p | 근소 비교 |
| cross_series | 83% | 97% | +14%p | |
| sum | 88% | 100% | +12%p | |
| dual_trap | 89% | 100% | +11%p | |
| value_read | 92% | 100% | +8%p | |
| argmax | 92% | 96% | +4%p | |
| diff | 83% | 100% | +17%p | |
| **전체** | **70.2%** (396/564) | **96.6%** (545/564) | **+26.4%p** | |

라벨 유무 무관하게 향상: 무라벨 71%→**96%**(281/292), 라벨 69%→**97%**(264/272). Day 3 최대 약점 3종(단위환산·조밀순위·근소비교)이 모두 90%대로 진입. 상세: [val_compare.json](val_compare.json) · [val_adapter/metrics.json](val_adapter/metrics.json)

## 2. 분포 밖 일반화 — Day 1 진단셋 16문항: 75% → 81%

Day 1 수기 차트(합성 생성기와 렌더링·값이 다른 OOD)에 어댑터 적용. baseline = [day1_zeroshot](../day1_zeroshot/report.md).

| id | 유형 | 정답 | baseline → adapter | |
|---|---|---|---|---|
| 08-q2 | rank_kth | 부산 | ❌ → ✅ | 조밀 순위 **일반화** |
| 08-q1 | value_read(무라벨) | 약 110만 | ❌ → ✅ | 판독 **일반화** |
| **07-q1** | **unit_convert** | **8,500억** | ❌ 850억 → ❌ **85,000억** | 오류 방향만 반전 |
| 07-q2 | sum | 2.98조 | ✅ 3.03조 → ❌ 2.68조 | **퇴행** |
| 02-q2 | narrow_compare | 2분기 | ❌ → ❌ | 잔존 |
| 나머지 11 | — | — | ✅ → ✅ | 유지(퇴행 없음) |

**핵심 발견 — 합성 SFT의 렌더링 분포 과적합:**
- val의 unit_convert 96%가 **OOD로 완전히 전이되지 않는다.** 07-q1에서 base는 10배 *과소*(850억), adapter는 10배 *과대*(85,000억). **×10,000 환산 산술은 학습됐으나, 시각적으로 다른 차트에서 값의 자릿수(0.85조를 8.5조로) 판독이 합성 분포에 민감**하다.
- 조밀 순위(08-q2)·무라벨 판독(08-q1)은 OOD에서도 개선 → 이 능력들은 **진짜 일반화**.
- 07-q2 합산이 통과→실패(경미한 분포 이동). 그 외 11문항은 퇴행 없음 → **차트 과제 내 catastrophic forgetting은 제한적**(아래 §4).

상세 16행: [day1_scored.json](day1_scored.json)

## 3. 잔여 오류 분석 (val 19/564)

| 군집 | 건수 | 예 | 성격 |
|---|---|---|---|
| 조밀/누적 차트 **근소 비교**(narrow_compare) | 6 | 가전 vs 철강, 전북 vs 세종 | near-tie 시지각 |
| 조밀 차트 **순위**(rank_kth) | 5 | 3위 석유화학→디스플레이 | near-tie 시지각 |
| 누적 차트 **세그먼트 비교**(cross_series/argmax) | 4 | 국내→해외, 최대연도 오독 | near-tie 시지각 |
| **unit_convert** 판독 | 3 | 46,000→46,500억(축 1스텝) | 환산 아닌 판독 |
| 라벨 오독(argmax pie) | 1 | G물산→"G울산" | 글자 오독 |

**잔여 오답의 성격이 바뀌었다.** Day 3의 오류는 *체계적 능력 공백*(단위 환산 0%, 순위 카운팅 붕괴)이었지만, Day 4 이후 잔여 19건은 거의 전부 **조밀·누적 차트에서 거의 같은 값을 구분하는 near-tie 시지각 문제**다. 데이터로 메운 능력 공백은 닫혔고, 남은 건 해상도·판독의 어려운 꼬리.

## 4. Ablation — 1 epoch vs 3 epoch

Day 4에서 학습 손실이 step ~65(epoch 0.2)에 포화 → "1 epoch로 충분한가?" 검증. 동일 설정에서 epoch만 1로([ablation_1ep.yaml](ablation_1ep.yaml)) 재학습.

| 지표 | 1 epoch | 3 epoch |
|---|---|---|
| 학습 시간 | **34:55** | 1:41:14 |
| train_loss | 0.282 | 0.149 |
| **val 564 전체** | **95.7%** (540) | 96.6% (545) |
| val unit_convert | 97% | 96% |
| val rank_kth | 92% | 95% |
| val narrow_compare | 94% | 96% |
| **Day 1 OOD** | 75% (12/16) | 81% (13/16) |
| Day 1 07-q1(단위환산) | ❌ 85,000억 | ❌ 85,000억 |

**결론:**
- **1 epoch가 3 epoch 성능의 ~99%(95.7 vs 96.6%)를 1/3 시간에** 달성. 손실이 epoch 0.2에 포화한다는 Day 4 관찰과 일치 — 이 좁은 태스크는 사실상 1 epoch면 학습된다. **unit_convert는 1 epoch에 이미 완성(97%)**, 추가 epoch은 어려운 near-tie 유형(rank_kth·narrow_compare)만 몇 %p 개선.
- **3 epoch가 OOD에서 더 나쁘지 않다**(81%≥75%) → 이 데이터·규모에선 과적합 페널티 없음. 3 epoch는 소폭 이득.
- **Day 1 07-q1 단위환산 실패가 1·3 epoch 동일**(둘 다 85,000억, 10배 과대) → §2의 OOD 자릿수 오류는 **epoch 과적합이 아니라 렌더링 분포 이동**이 원인임을 확증.

비교: [ablation_compare.json](ablation_compare.json)

## 한계와 다음(Day 6)

- **최대 한계 = 합성 렌더링 과적합(§2).** val(동일 생성기)에선 96.6%지만 OOD 단위환산은 자릿수 판독에서 흔들린다. → 후속: **차트 렌더링 다양화**(폰트·색·축 스케일·스타일)와 OOD 검증셋 확대.
- 잔여 오류는 near-tie 시지각(§3) → 데이터로 더 밀어붙이기보다 해상도/프롬프트/CoT 여지.
- 예측이 빈 `<think></think>` + 정답 형태 → Day 6 vLLM 서빙 시 파싱 규칙 필요.

## 재현

```bash
# full val 564 (Day4 어댑터)
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  .venv-train/bin/python -u scripts/run_zeroshot.py --adapter experiments/day4_qlora/adapter \
  --qa data/synth/val.json --out experiments/day5_eval/val_adapter
uv run scripts/eval.py --results experiments/day5_eval/val_adapter/results.jsonl --out experiments/day5_eval/val_adapter

# Day 1 진단셋 재평가 (16, OOD)
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  .venv-train/bin/python -u scripts/run_zeroshot.py --adapter experiments/day4_qlora/adapter \
  --qa data/day1_charts/qa.json --out experiments/day5_eval/day1_adapter

# 1-epoch ablation
CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 WANDB_PROJECT=ko-chart-vlm \
  .venv-train/bin/llamafactory-cli train experiments/day5_eval/ablation_1ep.yaml
```

> RTX 4090은 `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` 필수. 대용량 모델 다중 동시 로드 시 NFS I/O 경합으로 로딩이 멈출 수 있어 순차 로드 권장(로드 후 page cache warm).
