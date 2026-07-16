# ko-chart-vlm — 한국어 차트 이해 VLM 만들기 (7일 프로젝트)

오픈소스 VLM이 **한국어 차트·문서를 얼마나 잘 읽는지 정량적으로 측정**하고,
합성 데이터로 **QLoRA 도메인 적응**시켜 개선폭을 증명한 뒤, **vLLM으로 서빙**까지 하는 풀사이클 프로젝트.

> 배경: 국내 VLM 응용 채용 공고(업스테이지·그래파이·로민 등)의 공통 요구 역량이
> ① 도메인 파인튜닝(SFT/LoRA) ② 데이터 구축 ③ 평가 벤치마크 설계 ④ vLLM 서빙/경량화 —
> 이 프로젝트는 네 가지를 전부 커버하도록 설계했다.

## 로드맵

- [x] **Day 1** — VLM 개념 정리([docs/vlm-basics.html](docs/vlm-basics.html)), 환경 구축(uv), 한국어 차트 zero-shot 추론 및 실패 사례 수집
- [ ] **Day 2** — 한국어 합성 차트 데이터셋 구축 (5천~1만 QA쌍)
- [ ] **Day 3** — 평가 하네스(relaxed accuracy) + zero-shot 베이스라인 측정
- [ ] **Day 4** — LLaMA-Factory로 QLoRA SFT
- [ ] **Day 5** — 평가·오류 분석·ablation
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
```

## 실험 이력 규칙

- 실험 1건 = `experiments/<이름>/` 디렉토리 1개 (결과 jsonl + meta.json + report.md)
- 실험 완료 시점에 `exp(<이름>): ...` 커밋으로 코드·데이터·결과를 함께 고정
- 학습 실험(Day 4~)은 wandb 로깅 병행 예정

## 실험 로그

| 날짜 | 실험 | 요약 |
|---|---|---|
| 2026-07-16 | [day1_zeroshot](experiments/day1_zeroshot/) | Qwen3-VL-8B 한국어 차트 16문항 zero-shot — **strict 75%**. 약점 3종 발견: 조→억 단위 환산(10배 오류), 라벨 없는 차트 정밀 판독·순위, 근소 차이 비교 |

## 환경

- NVIDIA RTX 4090 24GB (공유 서버, 유휴 GPU 사용), driver 535 / CUDA 12.x
- Python 3.11 + [uv](https://docs.astral.sh/uv/), PyTorch, HF Transformers
