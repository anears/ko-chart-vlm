"""Day 6 — base vs 파인튜닝 Qwen3-VL 한국어 차트 QA 비교 데모.

두 개의 vLLM OpenAI 호환 서버에 같은 (이미지, 질문)을 보내 좌/우로 답을 비교한다.
  - base      : 원본 Qwen/Qwen3-VL-8B-Instruct        (http://localhost:8000)
  - finetuned : Day4 QLoRA 병합 모델                   (http://localhost:8001)

프롬프트는 run_zeroshot.py와 동일(PROMPT_SUFFIX)하게 맞춰 학습/서빙을 일치시킨다.
파인튜닝 모델은 빈 <think></think> 블록 뒤 정답을 내므로 표시 전에 제거한다.

실행:
  .venv-serve/bin/python scripts/serve_demo.py                       # Gradio UI (0.0.0.0:7860)
  .venv-serve/bin/python scripts/serve_demo.py --smoke <img> "<질문>"  # 헤드리스 1건 좌우 비교
"""

import argparse
import base64
import io
import re
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
PROMPT_SUFFIX = "\n차트를 보고 한국어로 간결하게 답하세요."  # run_zeroshot.py:22 와 동일

SERVERS = {
    "base": {"url": "http://localhost:8000/v1", "model": "base",
             "title": "원본 Qwen3-VL-8B"},
    "finetuned": {"url": "http://localhost:8001/v1", "model": "finetuned",
                  "title": "QLoRA 파인튜닝"},
}

EXAMPLES = [
    [str(ROOT / "data/day1_charts/07_units_krw.png"), "연구개발 부문 예산은 몇 억 원인가요?"],
    [str(ROOT / "data/day1_charts/08_bar_dense.png"), "인구가 세 번째로 많은 시도는 어디인가요?"],
    [str(ROOT / "data/day1_charts/05_bar_stacked.png"), "2022년 수출과 내수를 합친 전체 매출은 몇 조 원인가요?"],
]


def _b64_data_url(image) -> str:
    if isinstance(image, (str, Path)):
        data = Path(image).read_bytes()
    else:  # PIL.Image
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        data = buf.getvalue()
    return "data:image/png;base64," + base64.b64encode(data).decode()

def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def ask(server: dict, image, question: str) -> str:
    client = OpenAI(base_url=server["url"], api_key="EMPTY", timeout=60)
    resp = client.chat.completions.create(
        model=server["model"], temperature=0, max_tokens=128,
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": _b64_data_url(image)}},
            {"type": "text", "text": question + PROMPT_SUFFIX},
        ]}],
    )
    return _strip_think(resp.choices[0].message.content)


def compare(image, question: str):
    if image is None or not (question or "").strip():
        return "이미지와 질문을 입력하세요.", ""
    outs = {}
    for key, server in SERVERS.items():
        try:
            outs[key] = ask(server, image, question)
        except Exception as e:  # 서버 미기동 등
            outs[key] = f"[서버 오류] {e}"
    return outs["base"], outs["finetuned"]


def build_ui():
    import gradio as gr
    with gr.Blocks(title="한국어 차트 VLM — base vs 파인튜닝") as demo:
        gr.Markdown("# 한국어 차트 이해 VLM — base vs QLoRA 파인튜닝\n"
                    "같은 차트·질문을 원본/파인튜닝 모델에 동시 질의해 비교합니다. "
                    "(조→억 단위 환산, 무라벨 순위 등 Day 3 약점 개선을 확인해 보세요.)")
        with gr.Row():
            img = gr.Image(type="pil", label="차트 이미지")
            with gr.Column():
                q = gr.Textbox(label="질문(한국어)", placeholder="예: 연구개발 부문 예산은 몇 억 원인가요?")
                btn = gr.Button("비교 실행", variant="primary")
        with gr.Row():
            out_base = gr.Textbox(label=f"base — {SERVERS['base']['title']}")
            out_ft = gr.Textbox(label=f"finetuned — {SERVERS['finetuned']['title']}")
        btn.click(compare, [img, q], [out_base, out_ft])
        gr.Examples(EXAMPLES, [img, q])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", nargs=2, metavar=("IMAGE", "QUESTION"),
                    help="헤드리스로 1건 좌우 비교 후 종료(서버 검증용)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    if args.smoke:
        image, question = args.smoke
        b, f = compare(image, question)
        print(f"질문: {question}")
        print(f"[base]      {b}")
        print(f"[finetuned] {f}")
        return

    build_ui().launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
