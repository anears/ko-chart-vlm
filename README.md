# ko-chart-vlm — 한국어 차트 이해 VLM 만들기 (7일 프로젝트)

오픈소스 VLM이 **한국어 차트·문서를 얼마나 잘 읽는지 정량적으로 측정**하고,
합성 데이터로 **QLoRA 도메인 적응**시켜 개선폭을 증명한 뒤, **vLLM으로 서빙**까지 하는 풀사이클 프로젝트.

> 배경: 국내 VLM 응용 채용 공고(업스테이지·그래파이·로민 등)의 공통 요구 역량이
> ① 도메인 파인튜닝(SFT/LoRA) ② 데이터 구축 ③ 평가 벤치마크 설계 ④ vLLM 서빙/경량화 —
> 이 프로젝트는 네 가지를 전부 커버하도록 설계했다.

## 로드맵

- [x] **Day 1** — VLM 개념 정리([docs/vlm-basics.html](docs/vlm-basics.html)), 환경 구축(uv), 한국어 차트 zero-shot 추론 및 실패 사례 수집
- [x] **Day 2** — 한국어 합성 차트 데이터셋 구축 ([5,671 QA쌍](experiments/day2_dataset/report.md), Day 1 약점 3종 타깃)
- [x] **Day 3** — 평가 하네스 + zero-shot 베이스라인 ([val 564문항 70.2%](experiments/day3_baseline/report.md), 단위환산 0%)
- [x] **Day 4** — LLaMA-Factory QLoRA SFT ([스모크 163문항 47.2%→96.3%](experiments/day4_qlora/report.md), **unit_convert 0%→96%**)
- [ ] **Day 5** — 평가·오류 분석·ablation (val 564 full eval, Day1 진단셋 재평가, epoch ablation)
- [ ] **Day 6** — vLLM 서빙 + Gradio 데모 (+AWQ 양자화)
- [ ] **Day 7** — 결과 정리, README/블로그

## 저장소 구조

```
docs/          개념 정리 문서
scripts/       데이터 생성·추론·평가 스크립트
data/          생성된 데이터셋 (소용량만 커밋)
experiments/   실험별 결과·리포트 — 모든 실험 이력은 여기 + git 커밋으로 남긴다
assets/fonts/  차트 렌더링용 나눔고딕 (OFL 라이선스)
```

## 실행 방법

```bash
uv sync                                   # 환경 재현 (python 3.11, uv.lock 기준)
uv run scripts/make_day1_charts.py        # Day 1 진단용 차트 8장 + QA 생성
CUDA_VISIBLE_DEVICES=1 uv run scripts/run_zeroshot.py \
    --qa data/day1_charts/qa.json --out experiments/day1_zeroshot

uv run scripts/make_chart_dataset.py --charts 2000 --val-frac 0.1  # Day 2 합성 데이터셋 재생성 (seed 고정)

CUDA_VISIBLE_DEVICES=4 uv run scripts/run_zeroshot.py \
    --qa data/synth/val.json --out experiments/day3_baseline    # Day 3 val 추론
uv run scripts/eval.py --results experiments/day3_baseline/results.jsonl \
    --out experiments/day3_baseline                             # Day 3 자동 채점

# Day 4 — QLoRA SFT (학습은 전용 venv .venv-train에 격리, 메인 env 불변)
uv run scripts/to_sharegpt.py                                   # 학습 데이터 → ShareGPT
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 WANDB_PROJECT=ko-chart-vlm \
    .venv-train/bin/llamafactory-cli train experiments/day4_qlora/qwen3vl_qlora.yaml
```

> 학습 venv 구성·평가 명령 전체는 [Day 4 리포트의 재현 절](experiments/day4_qlora/report.md#재현) 참고.
> RTX 4090에서는 `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`이 필수(P2P 미지원).

> Day 2 합성 이미지(`data/synth/train·val/`)는 용량이 커 git에 넣지 않는다.
> 위 명령으로 seed에서 동일하게 재생성되며, QA json·manifest·샘플만 커밋되어 있다.

## 실험 이력 규칙

- 실험 1건 = `experiments/<이름>/` 디렉토리 1개 (결과 jsonl + meta.json + report.md)
- 실험 완료 시점에 `exp(<이름>): ...` 커밋으로 코드·데이터·결과를 함께 고정
- 학습 실험(Day 4~)은 wandb 로깅 병행 예정

## 실험 로그

| 날짜 | 실험 | 요약 |
|---|---|---|
| 2026-07-16 | [day1_zeroshot](experiments/day1_zeroshot/) | Qwen3-VL-8B 한국어 차트 16문항 zero-shot — **strict 75%**. 약점 3종 발견: 조→억 단위 환산(10배 오류), 라벨 없는 차트 정밀 판독·순위, 근소 차이 비교 |
| 2026-07-20 | [day2_dataset](experiments/day2_dataset/) | 합성 차트 QA **5,671쌍**(train 5,107 + val 564) 생성. Day 1 약점 타깃: unit_convert 조→억 strict 687, 무라벨 차트 55.7% + rank_kth 884, narrow_compare 1,057. GT는 원본 데이터에서 계산 |
| 2026-07-20 | [day3_baseline](experiments/day3_baseline/) | 평가 하네스(`eval.py`, selftest 20/20) + val 564문항 zero-shot **70.2%**. **unit_convert 0/73**(조→억 환산 완전 실패), rank_kth dense 41%, narrow_compare 79%. 무라벨 읽기는 91%로 견고 → Day 1 약점 #2는 '읽기'가 아니라 '순위'로 정밀화 |
| 2026-07-21 | [day4_qlora](experiments/day4_qlora/) | LLaMA-Factory **QLoRA**(4bit, r16, LLM 디코더만, 43.6M/0.5%) 3 epoch, 1h41m. 스모크 163문항 **47.2%→96.3%(+49.1%p)**. **unit_convert 0%→96%**(조→억 환산 해결, 잔여 오답 3건은 환산이 아닌 무라벨 판독 오차), rank_kth 50%→92%. 잔여 약점은 근소 비교·조밀 순위의 지각 경계 사례 |

## 환경

- NVIDIA RTX 4090 24GB (공유 서버, 유휴 GPU 사용), driver 535 / CUDA 12.x
- Python 3.11 + [uv](https://docs.astral.sh/uv/), PyTorch, HF Transformers
