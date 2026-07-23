# 한국어 차트를 못 읽는 VLM, 7일 만에 고치기

이 글의 전문은 개인 사이트로 옮겼습니다.

**→ [anears.github.io/blog/2026/ko-chart-vlm](https://anears.github.io/blog/2026/ko-chart-vlm/)**

> 오픈소스 VLM(Qwen3-VL-8B)이 한국어 차트를 얼마나 읽는지 숫자로 측정하고,
> 약점을 겨냥한 합성 데이터로 QLoRA 파인튜닝해 **val 70.2% → 96.6%(+26.4%p)**,
> vLLM 서빙과 AWQ 경량화까지 진행한 7일 사이드 프로젝트입니다.

일자별 리포트·코드·wandb 로그는 이 저장소의 [experiments/](../experiments)에 그대로 있습니다.
