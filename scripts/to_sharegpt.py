"""Day 4 — 합성 학습셋(QA json) → LLaMA-Factory ShareGPT 멀티모달 포맷 변환.

data/synth/train.json (Day 2, 5,107 QA)을 LLaMA-Factory가 바로 읽는 sharegpt
포맷으로 바꾼다. 포맷은 LLaMA-Factory의 mllm_demo.json과 동일하게 맞춘다
(messages[{role,content}] + images[]).

핵심 원칙
  - user turn = "<image>" + 질문 + run_zeroshot.py와 **동일한** PROMPT_SUFFIX
    → 학습/추론 프롬프트를 일치시켜 도메인 적응 효과를 그대로 평가에 반영.
  - assistant turn = answer_gt (예: "150,000억 원", "부산") — 간결한 정답 그대로.
  - 이미지 경로는 절대경로(dataset_dir 밖의 data/synth/를 참조하므로).
  - 대화당 <image> 토큰 개수 = images 개수 = 1 (모두 단일 이미지).

출력
  data/sft/train_sharegpt.json   (재생성 가능 → gitignore)
  data/sft/dataset_info.json     (LLaMA-Factory 데이터셋 등록 — 커밋)

사용 예:
    uv run scripts/to_sharegpt.py
    uv run scripts/to_sharegpt.py --qa data/synth/train.json --name ko_chart_train
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# run_zeroshot.py:22 와 반드시 동일해야 학습/추론 프롬프트가 일치한다.
PROMPT_SUFFIX = "\n차트를 보고 한국어로 간결하게 답하세요."


def convert(qa_list: list[dict]) -> list[dict]:
    out = []
    for item in qa_list:
        img_abs = str((ROOT / item["image"]).resolve())
        out.append({
            "messages": [
                {"role": "user", "content": "<image>" + item["question"] + PROMPT_SUFFIX},
                {"role": "assistant", "content": item["answer_gt"]},
            ],
            "images": [img_abs],
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="data/synth/train.json")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--name", default="ko_chart_train", help="dataset_info.json 등록 이름")
    ap.add_argument("--file-name", default="train_sharegpt.json")
    args = ap.parse_args()

    qa_list = json.loads((ROOT / args.qa).read_text(encoding="utf-8"))
    records = convert(qa_list)

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / args.file_name).write_text(
        json.dumps(records, ensure_ascii=False, indent=1), encoding="utf-8")

    # LLaMA-Factory 데이터셋 등록 (mllm_demo 포맷과 동일한 태그)
    dataset_info = {
        args.name: {
            "file_name": args.file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")

    # 검증: <image> 토큰 수 == images 수
    bad = [i for i, r in enumerate(records)
           if r["messages"][0]["content"].count("<image>") != len(r["images"])]
    n_missing = sum(1 for r in records if not Path(r["images"][0]).exists())

    print(f"[done] {len(records)} records → {out_dir/args.file_name}")
    print(f"  dataset '{args.name}' 등록 → {out_dir/'dataset_info.json'}")
    print(f"  image-token 불일치: {len(bad)}건 · 존재하지 않는 이미지: {n_missing}건")
    print("  sample[0]:", json.dumps(records[0], ensure_ascii=False)[:200], "...")


if __name__ == "__main__":
    main()
