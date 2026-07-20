"""VLM zero-shot 추론 러너.

qa.json(이미지·질문·정답 목록)을 받아 모델 답변을 results.jsonl로 저장한다.
Day 1: 정성 진단용. Day 3: 평가 하네스가 이 출력을 채점한다.

사용 예:
    CUDA_VISIBLE_DEVICES=1 uv run scripts/run_zeroshot.py \
        --qa data/day1_charts/qa.json --out experiments/day1_zeroshot
"""

import argparse
import json
import time
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent

PROMPT_SUFFIX = "\n차트를 보고 한국어로 간결하게 답하세요."


def load_model(model_id: str):
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return model, processor, time.time() - t0


@torch.inference_mode()
def answer(model, processor, image_path: Path, question: str) -> tuple[str, float]:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": question + PROMPT_SUFFIX},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos,
                       padding=True, return_tensors="pt").to(model.device)
    t0 = time.time()
    out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    latency = time.time() - t0
    new_tokens = out[:, inputs.input_ids.shape[1]:]
    decoded = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return decoded, latency


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-VL-8B-Instruct")
    ap.add_argument("--qa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="앞 N개만 추론(0=전체, smoke용)")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    qa_list = json.loads((ROOT / args.qa).read_text(encoding="utf-8"))
    if args.limit:
        qa_list = qa_list[:args.limit]

    model, processor, load_sec = load_model(args.model)

    results = []
    for i, item in enumerate(qa_list, start=1):
        pred, latency = answer(model, processor, ROOT / item["image"], item["question"])
        results.append({**item, "prediction": pred, "latency_sec": round(latency, 2)})
        print(f"[{i:2d}/{len(qa_list)}] {item['id']}  GT: {item['answer_gt']}  |  모델: {pred}")

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "model": args.model,
        "dtype": "bfloat16",
        "decoding": "greedy, max_new_tokens=96",
        "gpu": torch.cuda.get_device_name(0),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
        "model_load_sec": round(load_sec, 1),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "n_questions": len(results),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nmeta:", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
