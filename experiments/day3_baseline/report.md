# Day 3 — 평가 하네스 + 파인튜닝 전 baseline

- **날짜**: 2026-07-20 · **모델**: Qwen/Qwen3-VL-8B-Instruct (bf16, greedy) · **GPU**: RTX 4090 ×1 (peak 16.6GB)
- **평가셋**: [data/synth/val.json](../../data/synth/val.json) **564문항** (Day 2 합성, train과 시드 분리)
- **채점기**: [scripts/eval.py](../../scripts/eval.py) — eval_type별 자동 채점, `--selftest` 20/20 통과
- **목적**: 파인튜닝(Day 4) 전 baseline 확정 + Day 1 약점 가설을 대규모로 검증

## 결과: 70.2% (396/564)

| 채점유형 | 정확도 | 개수 | 기준 |
|---|---|---|---|
| category | 78.2% | 395 | 정규화 후 부분일치 |
| numeric_relaxed | 90.6% | 96 | 상대오차 ±5% |
| **numeric_strict** | **0.0%** | **73** | 상대오차 ±1% (= unit_convert) |

**질문유형별**

| 유형 | 정확도 | 개수 | |
|---|---|---|---|
| **unit_convert** | **0.0%** | 73 | ← 최대 약점 |
| rank_kth | 56.5% | 92 | ← 순위 약점 |
| narrow_compare | 78.9% | 114 | 근소 비교 |
| cross_series | 83.3% | 66 | |
| diff | 83.3% | 6 | |
| sum | 87.5% | 24 | |
| dual_trap | 88.9% | 27 | 이중축 함정에 잘 안 속음 |
| argmax | 91.7% | 96 | 강점 |
| value_read | 92.4% | 66 | 강점 |

## 핵심 발견

### 1. 단위 환산(조→억)은 **0/73 완전 실패** — 채점 버그 아님, 검증 완료

| 정답(조→억) | 모델 답변 | 오류 |
|---|---|---|
| 150,000억 (15조) | **15억** | 값만 읽고 단위만 교체 (×10,000 누락) |
| 37,000억 (3.7조) | **3.6억** | 동일 |
| 41,000억 (4.1조) | **410억** | ×100 부분 스케일 |
| 18,500억 (1.85조) | **175억** | ×100 |

모델은 **1조 = 10,000억** 관계를 전혀 적용하지 못한다. 조 값의 숫자(mantissa)는 정확히 읽지만(예: 15조→"15") 단위를 억으로 바꾸며 배율을 곱하지 않는다. 73문항 예측 중 '조'로 답한 것은 0개 → 환산을 시도조차 안 함. **파인튜닝으로 메울 headroom이 가장 큰 유형.**

### 2. Day 1 "라벨 없는 차트" 약점 = **읽기가 아니라 순위 매기기** (가설 정밀화)

값 하나 읽기는 라벨이 없어도 거의 안 나빠진다:

| value_read | 정확도 |
|---|---|
| 라벨 있음 | 96% (22/23) |
| 라벨 없음 | 91% (39/43) |

진짜 약점은 **여러 항목의 높이 순위를 매기는 것**이고, 막대가 많을수록 급격히 악화된다:

| rank_kth 차트 종류 | 정확도 |
|---|---|
| dense (10~14개 막대) | **41%** (15/37) |
| bar (4~6개) | 62% (25/40) |
| pie (라벨 있음) | 80% (12/15) |

> 예: "시도별 인구에서 3번째로 큰 항목?" 정답 부산 → 모델 인천. 조밀한 무라벨 막대에서 k번째 카운팅이 무너진다.

### 3. 근소 차이 비교(narrow_compare) 78.9% — near-tie 주입이 유효한 probe

> 예: "강원과 충남 중 더 큰 것?" 정답 강원 → 모델 충남. 시각적으로 거의 같은 높이에서 대소 판단이 흔들린다.

### 강점
argmax 91.7%, value_read 92.4%, dual_trap 88.9%(좌/우축 혼동에 잘 안 속음), cross_series 83.3%. **8B 최신 모델의 차트 기본기(읽기·최댓값·범례·이중축)는 이미 견고**하다. 남는 건 ①한국어 단위 체계 ②조밀 차트 순위 ③근소 비교.

## Day 4(파인튜닝)가 넘어야 할 baseline

| 지표 | baseline | 우선순위 |
|---|---|---|
| unit_convert (strict) | **0%** | ★★★ 최우선 — headroom 100%p |
| rank_kth (dense) | 41% | ★★ |
| narrow_compare | 79% | ★ |
| 전체 | 70.2% | |

## 참고: Day 1 진단셋

Day 1 curated 16문항은 이미 수기 채점(strict 75%)했고 정성 분석에 사용했다. 이 하네스는 파인튜닝 전후를 **동일 기준으로 자동 비교**할 대규모 val 세트(564)를 채점한다. 두 결과의 방향은 일치(단위 환산·순위·근소 비교가 약점).

## 재현

```bash
# 추론 (564문항 약 4.5분: 로드 13s + 0.45s/문항)
CUDA_VISIBLE_DEVICES=4 uv run scripts/run_zeroshot.py --qa data/synth/val.json --out experiments/day3_baseline
# 채점
uv run scripts/eval.py --results experiments/day3_baseline/results.jsonl --out experiments/day3_baseline
```

원본: [results.jsonl](results.jsonl) · 채점 상세: [scored.jsonl](scored.jsonl) · 지표: [metrics.json](metrics.json)
