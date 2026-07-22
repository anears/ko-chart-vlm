"""Day 6 — 병합 모델 AWQ W4A16 양자화 (llmcompressor).

Qwen3-VL의 vision tower/merger는 AWQ 커널 키 불일치 이슈가 알려져 있어
**language_model 선형층만** 양자화하고 visual·lm_head는 bf16로 남긴다.
캘리브레이션은 우리 차트 QA 질문(텍스트)으로 수행(LM만 통과, 이미지 토큰 없음).

실행(.venv-quant):
  CUDA_VISIBLE_DEVICES=2 .venv-quant/bin/python experiments/day6_serving/quantize_awq.py
출력: experiments/day6_serving/merged-awq  (W4A16 compressed-tensors)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = str(ROOT / "experiments/day6_serving/merged")
DST = str(ROOT / "experiments/day6_serving/merged-awq")
N_CALIB = 128

from datasets import Dataset
from transformers import AutoModelForImageTextToText, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier

def main():
    qa = json.loads((ROOT / "data/synth/train.json").read_text(encoding="utf-8"))
    texts = [q["question"] + "\n차트를 보고 한국어로 간결하게 답하세요." for q in qa[:N_CALIB]]
    ds = Dataset.from_dict({"text": texts})

    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForImageTextToText.from_pretrained(SRC, torch_dtype="auto", device_map="cuda")

    def tokize(b):
        return tok(b["text"], padding=False, truncation=True, max_length=512)
    ds = ds.map(tokize, remove_columns=["text"])

    recipe = AWQModifier(
        targets=["Linear"],
        scheme="W4A16",
        ignore=["lm_head", "re:.*visual.*", "re:.*merger.*", "re:.*patch_embed.*"],
    )
    oneshot(model=model, dataset=ds, recipe=recipe, output_dir=DST,
            max_seq_length=512, num_calibration_samples=N_CALIB, processor=tok)
    print("AWQ_DONE ->", DST)

if __name__ == "__main__":
    main()
