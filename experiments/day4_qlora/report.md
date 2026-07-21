# Day 4 — QLoRA SFT로 한국어 차트 약점 메우기

- **날짜**: 2026-07-21 · **베이스 모델**: Qwen/Qwen3-VL-8B-Instruct · **프레임워크**: LLaMA-Factory (git main, 전용 venv)
- **GPU**: RTX 4090 ×1 · **학습 시간**: 1:41:14 (960 step, 3 epoch) · **wandb**: [yq0kja5c](https://wandb.ai/anears-vuno/ko-chart-vlm/runs/yq0kja5c)
- **학습 데이터**: [data/synth/train.json](../../data/synth/train.json) 5,107 QA → ShareGPT 변환([scripts/to_sharegpt.py](../../scripts/to_sharegpt.py))
- **목적**: Day 3 baseline(val 70.2%, **unit_convert 0%**)의 약점을 QLoRA 도메인 적응으로 메우고 개선폭을 정량 증명

## 학습 설정 — QLoRA (4bit) on LLM decoder

| 항목 | 값 |
|---|---|
| 방식 | QLoRA 4bit (bnb nf4) · LoRA r=16, α=32, dropout=0.05, target=all(LLM 선형층) |
| 동결 | ViT + multi-modal projector 동결 → **LLM 디코더만 학습** (표준 VLM 적응 레시피) |
| 학습 파라미터 | **43,646,976 / 8,810,770,672 (0.495%)** |
| 템플릿 / 이미지 | `qwen3_vl` · cutoff 1536 · image_max_pixels 802,816(네이티브 605k↑ → 다운스케일 없음) |
| 스케줄 | 3 epoch · 유효 배치 16(2×accum8) · lr 1e-4 cosine · warmup 0.05 · seed 20260721 |

설정 전체: [qwen3vl_qlora.yaml](qwen3vl_qlora.yaml) · 손실 곡선: [adapter/training_loss.png](adapter/training_loss.png)

**손실은 step ~65(epoch 0.2)에서 5.9 → ~0.04로 급락 후 평탄**(final train_loss 0.149). 정답이 짧고 정형화된 좁은 합성 태스크라 포맷·환산 매핑을 빠르게 학습한다. → **1 epoch로도 충분할 가능성**(Day 5 ablation 후보).

## 스모크 평가: 47.2% → 96.3% (+49.1%p)

동일 **163문항 서브셋**(val에서 unit_convert 전량 73 + 유형별 12씩 층화), **동일 추론 하네스**([run_zeroshot.py](../../scripts/run_zeroshot.py) `--adapter`)로 baseline과 어댑터를 비교. baseline 수치는 Day 3 예측([day3_baseline/scored.jsonl](../day3_baseline/scored.jsonl))을 같은 서브셋으로 필터한 것 — 완전 apples-to-apples.

| 질문유형 | baseline | **adapter** | Δ | |
|---|---|---|---|---|
| **unit_convert** | 0% (0/73) | **96% (70/73)** | **+96%p** | ← 최우선 약점 해소 |
| rank_kth | 50% (6/12) | **92% (11/12)** | +42%p | 조밀 차트 순위 |
| sum | 83% | 100% | +17%p | |
| diff | 83% | 100% | +17%p | |
| dual_trap | 92% | 100% | +8%p | |
| argmax | 92% | 100% | +8%p | |
| cross_series | 92% | 92% | +0%p | |
| narrow_compare | 92% | 92% | +0%p | 근소 비교(잔여 약점) |
| value_read | 100% | 100% | +0%p | |
| **전체** | **47.2% (77/163)** | **96.3% (157/163)** | **+49.1%p** | |

채점유형별: numeric_strict(=unit_convert) 0% → **95.9%**, category 78%→95%, numeric_relaxed 100%. 상세: [smoke_compare.json](smoke_compare.json) · [smoke_adapter/metrics.json](smoke_adapter/metrics.json)

## 핵심 발견

### 1. 조→억 단위 환산 = **완전 실패(0%) → 사실상 해결(96%)**

Day 3에서 모델은 `1조 = 10,000억` 관계를 전혀 적용하지 못하고 값의 숫자만 읽어 단위만 바꿨다(15조 → "15억"). QLoRA 후 **×10,000 환산을 안정적으로 수행**한다.

특히 **잔여 오답 3건이 모두 환산 오류가 아니라 판독 오류**다:

| 정답 | 모델 | 원인 |
|---|---|---|
| 46,000억 | 46,500억 | 무라벨 막대를 축 1스텝(0.05조=500억) 위로 읽음 |
| 27,500억 | 28,000억 | 동일 (한 스텝 오독) |
| 35,500억 | 36,000억 | 동일 |

즉 산술(조→억)은 학습됐고, 남은 오차는 **무라벨 차트의 값 판독 정밀도**(±1% strict를 살짝 벗어남)다. 약점의 성격이 '한국어 단위 체계 공백'에서 '시지각 정밀도'로 이동했다.

### 2. 조밀 차트 순위(rank_kth) 50% → 92%

Day 3 최대 약점 중 하나였던 무라벨 다범주 막대의 k번째 순위 매기기가 크게 개선. 잔여 오답 1건(석유화학→디스플레이)은 12~14개 막대 중 근소한 높이 차 구간.

### 3. 잔여 약점 = 근소 비교·조밀 순위의 '경계 사례'

전체 오답 6건 = unit_convert 판독 3 + rank_kth 1 + cross_series 1(수출/내수 근접) + narrow_compare 1(전북/세종 near-tie). 모두 **시각적으로 거의 같은 값을 구분하는 지각 과제**로, 데이터가 아니라 해상도·판독의 문제.

## 한계와 다음(Day 5)

- **스모크 서브셋(163)** 결과다. 정식 **val 564문항 full eval + Day 1 curated 16문항** 재평가는 Day 5.
- val은 train과 **시드 분리(seed+999)**로 이미지·값이 겹치지 않지만 **동일 생성기 분포**다. 합성 분포 밖 일반화(특히 Day 1 수기 진단셋)와 **일반 능력 퇴행(catastrophic forgetting) 점검**이 Day 5 과제.
- 손실이 극초반에 포화 → **epoch 1 vs 3 ablation**, near-tie/조밀 순위 잔여 약점 타깃 데이터 보강 검토.
- 예측은 빈 `<think></think>` + 정답 형태(Qwen3-VL non-thinking 포맷). 채점에는 무영향이나 서빙(Day 6) 시 파싱 고려.

## 재현

```bash
# 0) 데이터·환경 (메인 env는 그대로, 학습은 .venv-train 격리)
uv run scripts/make_chart_dataset.py --charts 2000 --val-frac 0.1   # 이미지 재생성(seed)
uv run scripts/to_sharegpt.py                                        # → data/sft/ (ShareGPT + dataset_info)
uv venv .venv-train --python 3.11
uv pip install --python .venv-train torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install --python .venv-train "llamafactory @ git+https://github.com/hiyouga/LLaMA-Factory.git" \
    bitsandbytes peft qwen-vl-utils wandb

# 1) 학습 (RTX 4090 P2P 이슈 → NCCL_*_DISABLE 필수)
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 WANDB_PROJECT=ko-chart-vlm \
  .venv-train/bin/llamafactory-cli train experiments/day4_qlora/qwen3vl_qlora.yaml

# 2) 스모크 평가 (어댑터 로드)
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  .venv-train/bin/python scripts/run_zeroshot.py --adapter experiments/day4_qlora/adapter \
  --qa experiments/day4_qlora/smoke_val.json --out experiments/day4_qlora/smoke_adapter
uv run scripts/eval.py --results experiments/day4_qlora/smoke_adapter/results.jsonl --out experiments/day4_qlora/smoke_adapter
```

설정: [meta.json](meta.json) · 어댑터 config: [adapter/adapter_config.json](adapter/adapter_config.json) · 학습 로그: [adapter/trainer_log.jsonl](adapter/trainer_log.jsonl)
