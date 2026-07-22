# Day 6 — vLLM 서빙 + Gradio 데모 (+AWQ 경량화)

- **날짜**: 2026-07-22 · **대상**: Day4 QLoRA 병합 모델 vs 원본 Qwen3-VL-8B
- **서빙 스택**: 전용 `.venv-serve` — **vLLM 0.11.0 + torch 2.8.0+cu128 + transformers 4.57.6** (driver 535=CUDA 12.2 호환)
- **GPU**: RTX 4090 ×2 (base=GPU0:8000, finetuned=GPU1:8001) · **데모**: [scripts/serve_demo.py](../../scripts/serve_demo.py)
- **목적**: 파인튜닝 모델을 실서빙하고 base와 라이브 비교, AWQ 경량화 시도. 채용 요구역량 "④ vLLM 서빙/경량화" 커버.

## 1. 아키텍처

```
Day4 어댑터 ──(llamafactory export 병합)──▶ merged (bf16, 17GB)
                                              │
        vLLM OpenAI 서버 ×2 ──────────────────┤ base(8000) / finetuned(8001)
                                              │
        Gradio 데모(serve_demo.py) ──同一 이미지·질문 동시 질의──▶ 좌/우 비교
```

- **병합 서빙** 선택(LoRA-at-serve 대신): vLLM이 단일 `Qwen3VLForConditionalGeneration`으로 로드 → 견고 + AWQ 대상. 병합 설정: [export.yaml](export.yaml).
- Gradio: 이미지 업로드 + 한국어 질문 → 두 서버에 동시 요청 → base/finetuned 답 좌/우. `<think></think>` 스트립. 프롬프트는 학습/평가와 동일.

## 2. 서빙 스택 구성 — 드라이버 제약과 버전 정합 (실전 이슈 5종)

| # | 증상 | 원인 | 해결 |
|---|---|---|---|
| 1 | 실 GPU 연산 "driver too old (12020)" | 최신 vLLM 0.25가 torch **cu130**(CUDA 13) 요구, 서버 driver 535=CUDA 12.2 | **vllm 0.11.0 + torch cu128** 핀 (CUDA 12 minor 호환) |
| 2 | `Qwen2Tokenizer has no attribute all_special_tokens_extended` | vllm 0.11이 쓰는 tokenizer API를 transformers 5.14가 제거 | **transformers 4.57.6**로 다운그레이드 |
| 3 | `'list' has no attribute 'keys'` (tokenizer 로드) | 병합 모델(5.8 export)의 tokenizer가 5.x 포맷 | base 원본 tokenizer/processor로 교체(병합은 tokenizer 불변) |
| 4 | `No available memory for cache blocks` | 24GB에 16GB 모델 → KV 부족 | `--max-model-len 3072 --max-num-seqs 4 --enforce-eager --gpu-memory-utilization 0.95` |
| 5 | 재기동 시 GPU 메모리 점유 | EngineCore **고아 자식** 프로세스가 VRAM 붙잡음 | nvidia-smi로 GPU별 PID 매핑 후 직접 종료 |

> `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1`은 RTX 4090에서 필수(Day 4~ 동일).

## 3. 데모 — base vs 파인튜닝 (라이브)

| 차트 / 질문 | 정답 | base | finetuned |
|---|---|---|---|
| Day1 07, 연구개발 예산 (조→억) | 8,500억 | 850억 (×10⁴ 누락) | **8,500억 ✓** |
| val 00007, 누적합계 (조→억) | 150,000억 | 15억 (환산 안 함) | 60,000억 (환산O, 판독 오류) |

원본은 **조→억 환산을 시도조차 안 하는** 반면 파인튜닝 모델은 환산을 수행한다 — 개선이 서빙에 그대로 드러난다. 전체 예시: [demo_examples.txt](demo_examples.txt).

## 4. 🔑 핵심 발견 — train/serve preprocessing skew

**동일 이미지·동일 병합 가중치인데 추론 스택에 따라 답이 다르다:**

| 입력 (val_00007, GT 150,000억) | transformers(오프라인) | vLLM(서빙) |
|---|---|---|
| 파인튜닝 모델 답 | **150,000억 ✓** | 60,000억 ✗ |

- 서빙 **unit_convert 정확도 ≈ 25%(3/12)** vs 오프라인 평가 **96%**. (base는 서빙에서도 0/12 — 환산 자체를 안 하므로 개선 방향성은 유지)
- 원인: **이미지 전처리 경로 불일치.** 평가(`run_zeroshot`)는 **qwen_vl_utils(factor 28)**, 서빙은 **vLLM의 Qwen3-VL 프로세서(patch16/factor 32)** → 리사이즈 그리드가 달라 조밀 차트의 **값 판독**이 흔들린다. 단위 환산 *산술*은 전이되나 *지각*이 저하.
- 고해상도 강제(`mm_processor_kwargs` min_pixels↑)는 학습 해상도를 벗어나고 24GB OOM → 부적절.
- **교훈**: 오프라인 벤치 정확도는 서빙 스택으로 자동 전이되지 않는다. **서빙의 멀티모달 전처리를 학습/평가와 정합**시키거나, **서빙 경로 그대로 재평가**해야 공정하다. (Day 7 후속: 평가 하네스를 vLLM 경로로 통일)

## 5. AWQ 경량화 (W4A16)

**양자화 = 성공, 멀티모달 서빙 = 구버전 vLLM 제약으로 차단(폴백: bf16).**

`llmcompressor`(격리 `.venv-quant`, torch cu128)로 병합 모델을 **W4A16 AWQ** 양자화.
Qwen3-VL vision-merger 키 불일치를 피하려 **language_model 선형층만** 양자화(visual·merger·lm_head 제외), 텍스트 캘리브레이션 128샘플([quantize_awq.py](quantize_awq.py)).

| 항목 | bf16 | AWQ W4A16 |
|---|---|---|
| 디스크 풋프린트 | 17 GB | **6.8 GB (~2.5×↓)** |
| vLLM 로드 | ✓ | ✓ (compressed-tensors 인식) |
| 텍스트 추론 | ✓ | ✓ — 예: "1조는 몇 억?" → **"1조 = 10,000억 원"** 정확 |
| 멀티모달(이미지) 서빙 | ✓ | ✗ 400 `Failed to apply Qwen3VLProcessor` |

- **양자화 자체는 입증**: 2.5배 축소 + 단위 추론 정확도 유지(텍스트).
- **멀티모달 서빙 차단**: driver 535 제약으로 고정된 **vLLM 0.11**에서 compressed-tensors Qwen3-VL의 **이미지 요청 처리 경로가 실패**(vision_config는 bf16과 동일 — 양자화 config만 차이). 초기 조사에서 예고된 "Qwen3-VL 양자화 서빙 취약성"과 동일 계열. 서빙 중 두 스키마 이슈(`scale_dtype`/`zp_dtype` config, 5.x tokenizer)도 config 패치로 우회했으나 멀티모달 processor는 미해결.
- **폴백**: bf16 finetuned 서버가 실 멀티모달 서빙 경로. AWQ 멀티모달 서빙은 **최신 vLLM(CUDA 13 드라이버 환경)** 필요 → Day 7 후속.

산출: `merged-awq/`(6.8GB, gitignore) · 레시피 [quantize_awq.py](quantize_awq.py)

## 한계와 다음(Day 7)

- **서빙 정확도 격차(§4)가 최우선 후속** — 평가 하네스를 vLLM 서빙 경로로 통일해 서빙 기준 정확도를 재측정.
- driver 535(CUDA 12.2) 제약으로 구버전 스택(vllm 0.11)에 고정 — 최신 vLLM/양자화 도구를 쓰려면 CUDA 13 드라이버 환경 필요.
- Gradio 데모는 로컬(0.0.0.0:7860) 기동 확인. 공개 데모는 Day 7에서 정리.

## 재현

```bash
# 1) 병합
CUDA_VISIBLE_DEVICES=2 .venv-train/bin/llamafactory-cli export experiments/day6_serving/export.yaml
# 병합 모델 tokenizer/processor를 base 원본으로 교체(4.57 호환) — README 참고

# 2) 서빙 venv (driver 535 → cu128)
uv venv .venv-serve --python 3.11
uv pip install --python .venv-serve --torch-backend=cu128 "vllm==0.11.0" gradio openai
uv pip install --python .venv-serve "transformers>=4.57,<4.58"

# 3) 두 서버
CUDA_VISIBLE_DEVICES=0 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 .venv-serve/bin/vllm serve \
  Qwen/Qwen3-VL-8B-Instruct --port 8000 --served-model-name base \
  --max-model-len 3072 --max-num-seqs 4 --enforce-eager --gpu-memory-utilization 0.95
CUDA_VISIBLE_DEVICES=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 .venv-serve/bin/vllm serve \
  experiments/day6_serving/merged --port 8001 --served-model-name finetuned \
  --max-model-len 3072 --max-num-seqs 4 --enforce-eager --gpu-memory-utilization 0.95

# 4) 데모 / 검증
.venv-serve/bin/python scripts/serve_demo.py                      # Gradio UI :7860
.venv-serve/bin/python scripts/serve_demo.py --smoke data/day1_charts/07_units_krw.png "연구개발 부문 예산은 몇 억 원인가요?"
```
