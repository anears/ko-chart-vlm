"""Day 3 — 한국어 차트 QA 평가 하네스(채점기).

run_zeroshot.py가 낸 results.jsonl(예측 + 정답·채점 필드 포함)을 받아
문항별 eval_type에 따라 자동 채점하고 지표를 낸다.

채점 유형(데이터 생성 시 부여, data/synth/manifest.json 참고):
  category         — 카테고리 정답. 정규화 후 부분일치.
  numeric_relaxed  — 수치. 상대오차 ±5% (chart QA relaxed accuracy 관행).
  numeric_strict   — 수치. 상대오차 ±1%. 단위 환산(조→억) 자릿수 오류 검출용.

수치 채점의 핵심은 한국어 단위(조/억/만/천)를 정규(canonical) 원-환산값으로 바꿔
모델이 어떤 단위로 답하든("8,500억"·"0.85조"·"850000000000") 동일 기준으로 비교하는 것.

사용 예:
    uv run scripts/eval.py --results experiments/day3_baseline/results.jsonl \
        --out experiments/day3_baseline
    uv run scripts/eval.py --selftest        # 채점기 자체검증(모델 불필요)
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

UNIT_FACTOR = {"천": 1e3, "만": 1e4, "억": 1e8, "조": 1e12}
RELAXED_TOL = 0.05
STRICT_TOL = 0.01

# 숫자(콤마·소수 포함) + 뒤따르는 한국어 단위(선택)
_NUM_UNIT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(조|억|만|천)?")
_PLAIN = re.compile(r"-?\d+(?:\.\d+)?")


def _norm(s: str) -> str:
    """공백 제거 + 소문자화 — 카테고리 비교용."""
    return re.sub(r"\s+", "", str(s)).lower()


def score_category(pred: str, gt: str) -> bool:
    p, g = _norm(pred), _norm(gt)
    return bool(g) and g in p


def _plain_numbers(text: str) -> list[float]:
    return [float(x) for x in _PLAIN.findall(text.replace(",", ""))]


def parse_to_base(text: str, expected_unit: str) -> list[float]:
    """예측 문자열 → 정규 원-환산값 후보 리스트.

    단위 표기가 있으면 그대로 합산(예: '4만 5천'→45000). 단위 없이 숫자만 있으면
    (a) 그 숫자 자체, (b) 그 숫자×expected_unit 배율 — 두 해석을 모두 후보로 둔다.
    (질문이 단위를 명시하지 않아 '8500'이 8,500억을 뜻할 수도, 원값일 수도 있음)
    """
    text = text.replace(",", "")
    tagged = [(float(n), u) for n, u in _NUM_UNIT.findall(text) if n not in ("", "-", ".") and u]
    if tagged:
        return [sum(n * UNIT_FACTOR[u] for n, u in tagged)]
    nums = _plain_numbers(text)
    if not nums:
        return []
    b = max(nums, key=abs)
    return [b, b * UNIT_FACTOR[expected_unit]]


def _close(a: float, b: float, tol: float, abs_floor: float = 0.0) -> bool:
    if b == 0:
        return abs(a - b) <= max(abs_floor, 1e-9)
    return abs(a - b) <= max(tol * abs(b), abs_floor)


def score_numeric(pred: str, gt_value: float, unit: str, tol: float) -> bool:
    if unit == "%":
        return any(_close(c, gt_value, tol, abs_floor=0.5) for c in _plain_numbers(pred))
    if unit == "°C":
        nums = _plain_numbers(pred)
        if "영하" in pred:  # "영하 2도" = -2
            nums = [-abs(n) for n in nums]
        return any(_close(c, gt_value, tol, abs_floor=1.0) for c in nums)
    gt_base = gt_value * UNIT_FACTOR[unit]
    return any(_close(c, gt_base, tol) for c in parse_to_base(pred, unit))


def verdict(item: dict) -> bool:
    et = item["eval_type"]
    pred = item.get("prediction", "")
    if et == "category":
        return score_category(pred, item["answer_gt"])
    tol = STRICT_TOL if et == "numeric_strict" else RELAXED_TOL
    return score_numeric(pred, item["answer_value"], item["answer_unit"], tol)


# ── 집계 ───────────────────────────────────────────────────────────────────
def _agg(items: list[dict], key) -> dict:
    out: dict[str, dict] = {}
    for it in items:
        k = key(it)
        b = out.setdefault(k, {"n": 0, "correct": 0})
        b["n"] += 1
        b["correct"] += int(it["ok"])
    for b in out.values():
        b["acc"] = round(b["correct"] / b["n"], 3)
    return dict(sorted(out.items(), key=lambda x: -x[1]["n"]))


def evaluate(results: list[dict]) -> tuple[list[dict], dict]:
    scored = [{**it, "ok": verdict(it)} for it in results]
    n = len(scored)
    correct = sum(int(it["ok"]) for it in scored)
    metrics = {
        "n": n,
        "accuracy": round(correct / n, 3) if n else 0.0,
        "n_correct": correct,
        "by_eval_type": _agg(scored, lambda it: it["eval_type"]),
        "by_question_type": _agg(scored, lambda it: it["question_type"]),
        "by_labeled": _agg(scored, lambda it: "labeled" if it.get("has_value_labels") else "unlabeled"),
        "by_chart_kind": _agg(scored, lambda it: it.get("chart_kind", "?")),
    }
    return scored, metrics


# ── 자체검증 ────────────────────────────────────────────────────────────────
def selftest() -> None:
    cases = [
        # (pred, gt_value, unit, tol, expected)
        ("8,500억 원", 8500, "억", STRICT_TOL, True),      # 정확
        ("850억 원", 8500, "억", STRICT_TOL, False),        # 10배 오류 검출
        ("0.85조 원", 8500, "억", STRICT_TOL, True),        # 다른 단위로 답해도 정답
        ("8500", 8500, "억", STRICT_TOL, True),             # 단위 생략
        ("85,000억", 8500, "억", STRICT_TOL, False),        # 10배 초과
        ("약 110만 명", 110, "만", RELAXED_TOL, True),
        ("약 1,100,000명", 110, "만", RELAXED_TOL, True),   # 원값으로 답
        ("약 100만 명", 110, "만", RELAXED_TOL, False),     # ±5% 밖
        ("약 4만 5천 대", 45, "천", RELAXED_TOL, True),     # 복합 단위
        ("약 45,000대", 45, "천", RELAXED_TOL, True),
        ("34%", 34, "%", RELAXED_TOL, True),
        ("약 45퍼센트입니다", 45, "%", RELAXED_TOL, True),
        ("18도", 18, "°C", RELAXED_TOL, True),
        ("영하 2도", -2, "°C", RELAXED_TOL, True),
        ("약 2.4조 원", 2.35, "조", RELAXED_TOL, True),     # ±5% 안(1.1%)
    ]
    fails = 0
    for pred, gv, unit, tol, exp in cases:
        got = score_numeric(pred, gv, unit, tol)
        if got != exp:
            fails += 1
            print(f"  FAIL numeric: {pred!r} vs {gv}{unit} tol={tol} → {got}, 기대 {exp}")
    cat_cases = [
        ("부산입니다.", "부산", True),
        ("2023년", "2023", True),
        ("가장 큰 항목은 A전자입니다", "A전자", True),
        ("제주", "서울", False),
        ("경남", "경북", False),
    ]
    for pred, gt, exp in cat_cases:
        got = score_category(pred, gt)
        if got != exp:
            fails += 1
            print(f"  FAIL category: {pred!r} vs {gt!r} → {got}, 기대 {exp}")
    total = len(cases) + len(cat_cases)
    print(f"selftest: {total - fails}/{total} passed")
    sys.exit(1 if fails else 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", help="results.jsonl (run_zeroshot 출력)")
    ap.add_argument("--out", help="metrics.json·scored.jsonl 저장 디렉토리")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()

    if not args.results:
        ap.error("--results 또는 --selftest 필요")

    results = [json.loads(l) for l in Path(args.results).read_text(encoding="utf-8").splitlines() if l.strip()]
    scored, metrics = evaluate(results)

    if args.out:
        out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        with (out / "scored.jsonl").open("w", encoding="utf-8") as f:
            for it in scored:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"\n전체 정확도: {metrics['accuracy']:.1%}  ({metrics['n_correct']}/{metrics['n']})\n")
    print(f"{'채점유형':<16}{'정확도':>8}{'개수':>8}")
    for k, v in metrics["by_eval_type"].items():
        print(f"{k:<16}{v['acc']:>8.1%}{v['n']:>8}")
    print(f"\n{'질문유형':<16}{'정확도':>8}{'개수':>8}")
    for k, v in metrics["by_question_type"].items():
        print(f"{k:<16}{v['acc']:>8.1%}{v['n']:>8}")
    print(f"\n{'라벨':<16}{'정확도':>8}{'개수':>8}")
    for k, v in metrics["by_labeled"].items():
        print(f"{k:<16}{v['acc']:>8.1%}{v['n']:>8}")


if __name__ == "__main__":
    main()
