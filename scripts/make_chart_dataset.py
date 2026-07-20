"""Day 2 — 한국어 차트 QA 합성 데이터셋 생성기.

Day 1 zero-shot 진단(experiments/day1_zeroshot/report.md)에서 드러난 3대 약점을
집중 공략하도록 데이터 분포를 설계한다:

  ① 한국어 수 단위 환산(조↔억↔만↔천)  — unit_convert 유형을 크게 확대, strict 채점
  ② 값 라벨 없는 차트의 정밀 판독·순위 — 차트 절반 이상을 라벨 없이 렌더(축만 보고 읽기)
  ③ 근소한 차이 비교                    — 일부러 near-tie를 주입한 narrow_compare 유형

완전히 결정적(seeded)이라 언제 다시 돌려도 동일한 데이터셋이 나온다. 즉 대용량
PNG는 git에 넣지 않고(.gitignore), 생성기+seed로 재현한다.

출력:
  data/synth/train/00000.png ...        (gitignore)
  data/synth/val/00000.png ...          (gitignore)
  data/synth/train.json, val.json       (QA 리스트, 채점용 구조화 필드 포함)
  data/synth/manifest.json              (seed·분포 통계)

사용 예:
    uv run scripts/make_chart_dataset.py --charts 1600 --val-frac 0.1
"""

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "synth"
FONT = ROOT / "assets" / "fonts" / "NanumGothic-Regular.ttf"

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]

# ── 한국어 수 단위 ─────────────────────────────────────────────────────────
UNIT_FACTOR = {"천": 1e3, "만": 1e4, "억": 1e8, "조": 1e12}


def setup_korean_font() -> None:
    fm.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def kfmt(v: float) -> str:
    """천단위 콤마 + 불필요한 소수점 제거. 8500 -> '8,500', 0.85 -> '0.85'."""
    if abs(v - round(v)) < 1e-9:
        return f"{round(v):,}"
    return f"{v:,.2f}".rstrip("0").rstrip(".")


def convert(value: float, from_unit: str, to_unit: str) -> float:
    return value * UNIT_FACTOR[from_unit] / UNIT_FACTOR[to_unit]


def ans_str(value: float, unit: str, noun: str, approx: bool = False) -> str:
    if unit in ("%", "°C"):
        return f"{kfmt(value)}{unit}"
    prefix = "약 " if approx else ""
    space = f" {noun}" if noun else ""
    return f"{prefix}{kfmt(value)}{unit}{space}"


# ── 카테고리·도메인 풀 ─────────────────────────────────────────────────────
POOLS = {
    "regions": ["서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"],
    "depts": ["반도체", "디스플레이", "바이오", "2차전지", "자동차", "철강",
              "조선", "석유화학", "가전", "통신", "항공", "방산"],
    "companies": ["A전자", "B모바일", "C테크", "D산업", "E바이오", "F에너지",
                  "G물산", "H화학", "I소재", "J로보틱스"],
    "products": ["노트북", "스마트폰", "태블릿", "웨어러블", "데스크톱", "모니터", "이어폰"],
    "years": [str(y) for y in range(2015, 2027)],
}


def pick_categories(rng: random.Random, pool: str, n: int) -> list[str]:
    items = POOLS[pool]
    n = min(n, len(items))
    if pool == "years":  # 시계열은 연속 구간
        start = rng.randint(0, len(items) - n)
        return items[start:start + n]
    return rng.sample(items, n)


def round_step(v: float, step: float) -> float:
    return round(round(v / step) * step, 4)


def sample_values(rng: random.Random, n: int, lo: float, hi: float, step: float) -> list[float]:
    return [round_step(rng.uniform(lo, hi), step) for _ in range(n)]


def inject_near_tie(rng: random.Random, vals: list[float], step: float) -> None:
    """근소 비교 유형을 위해 두 값을 1스텝 차이로 붙인다(값은 서로 다르게 유지)."""
    if len(vals) < 2:
        return
    i, j = rng.sample(range(len(vals)), 2)
    vals[j] = round(vals[i] + rng.choice([-1, 1]) * step, 4)
    if vals[j] <= 0:
        vals[j] = round(vals[i] + step, 4)


# 단일 시리즈(막대) 차트용 도메인: (unit, noun, lo, hi, step, 환산 대상 단위, 제목)
# unit_convert는 Day1 약점 그대로 조→억(1조=10,000억)만. 조 값은 step에 맞춰
# 반올림돼 있어 *10^4가 항상 깔끔한 정수 → strict 채점 안전.
BAR_DOMAINS = [
    dict(pool="years", unit="조", noun="원", lo=0.5, hi=4.0, step=0.05, conv="억", title="연도별 매출 추이"),
    dict(pool="depts", unit="조", noun="원", lo=0.3, hi=3.5, step=0.05, conv="억", title="산업별 수출액"),
    dict(pool="depts", unit="조", noun="원", lo=0.5, hi=4.5, step=0.05, conv="억", title="부문별 투자 예산"),
    dict(pool="companies", unit="조", noun="원", lo=0.4, hi=5.0, step=0.05, conv="억", title="기업별 시가총액"),
    dict(pool="regions", unit="만", noun="명", lo=40, hi=1400, step=1, conv=None, title="시도별 인구"),
    dict(pool="products", unit="천", noun="대", lo=15, hi=95, step=1, conv=None, title="제품별 판매량"),
]


# ── 차트 스펙 ──────────────────────────────────────────────────────────────
@dataclass
class Series:
    name: str
    values: list[float]


@dataclass
class ChartSpec:
    kind: str
    title: str
    categories: list[str]
    series: list[Series]
    unit: str
    noun: str
    show_labels: bool = True
    conv: str | None = None          # unit_convert 대상 단위(조/억 도메인만)
    y_label: str = ""
    extra: dict = field(default_factory=dict)


# ── 렌더링 ────────────────────────────────────────────────────────────────
def _bar_labels(ax, bars, fmt="{:.0f}") -> None:
    for b in bars:
        ax.annotate(fmt.format(b.get_height()),
                    (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)


def render(spec: ChartSpec, path: Path) -> None:
    fmt = "{:.2f}" if spec.unit == "조" else "{:.0f}"

    if spec.kind in ("bar", "dense"):
        vals = spec.series[0].values
        wide = len(spec.categories) > 8
        fig, ax = plt.subplots(figsize=(10 if wide else 8, 5))
        bars = ax.bar(spec.categories, vals, color=PALETTE[0], width=0.6)
        if spec.show_labels:
            _bar_labels(ax, bars, fmt)
        ax.set_ylabel(spec.y_label)
        ax.grid(axis="y", alpha=0.3)
        if wide:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    elif spec.kind == "grouped":
        import numpy as np
        x = np.arange(len(spec.categories))
        w = 0.8 / len(spec.series)
        fig, ax = plt.subplots(figsize=(9, 5))
        for k, s in enumerate(spec.series):
            off = (k - (len(spec.series) - 1) / 2) * w
            bars = ax.bar(x + off, s.values, w, label=s.name, color=PALETTE[k])
            if spec.show_labels:
                _bar_labels(ax, bars, fmt)
        ax.set_xticks(x, spec.categories)
        ax.set_ylabel(spec.y_label)
        ax.legend()

    elif spec.kind == "line":
        fig, ax = plt.subplots(figsize=(9, 5))
        for k, s in enumerate(spec.series):
            ax.plot(spec.categories, s.values, marker="o", label=s.name, color=PALETTE[k])
            if spec.show_labels:
                for xc, yv in zip(spec.categories, s.values):
                    ax.annotate(fmt.format(yv), (xc, yv), textcoords="offset points",
                                xytext=(0, 6), ha="center", fontsize=7)
        ax.set_ylabel(spec.y_label)
        ax.grid(alpha=0.3)
        ax.legend()

    elif spec.kind == "pie":
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.pie(spec.series[0].values, labels=spec.categories, autopct="%1.0f%%",
               startangle=90, colors=PALETTE[:len(spec.categories)])
        ax.axis("equal")

    elif spec.kind == "stacked":
        import numpy as np
        fig, ax = plt.subplots(figsize=(8, 5))
        bottom = np.zeros(len(spec.categories))
        for k, s in enumerate(spec.series):
            ax.bar(spec.categories, s.values, bottom=bottom, label=s.name, color=PALETTE[k])
            if spec.show_labels:
                for xi, (b, v) in enumerate(zip(bottom, s.values)):
                    ax.annotate(fmt.format(v), (xi, b + v / 2), ha="center",
                                va="center", color="white", fontsize=8)
            bottom += np.array(s.values)
        ax.set_ylabel(spec.y_label)
        ax.legend()

    elif spec.kind == "dual":
        bar, line = spec.series[0], spec.series[1]
        fig, ax1 = plt.subplots(figsize=(9, 5))
        bars = ax1.bar(spec.categories, bar.values, color="#B0C4DE", label=f"{bar.name} (좌축)")
        if spec.show_labels:
            _bar_labels(ax1, bars, fmt)
        ax1.set_ylabel(spec.y_label)
        ax1.set_ylim(0, max(bar.values) * 1.35)
        ax2 = ax1.twinx()
        ax2.plot(spec.categories, line.values, marker="D", color="#C44E52", label=f"{line.name} (우축)")
        if spec.show_labels:
            for xc, yv in zip(spec.categories, line.values):
                ax2.annotate(f"{kfmt(yv)}%", (xc, yv), textcoords="offset points",
                             xytext=(0, 8), ha="center", color="#C44E52", fontsize=8)
        ax2.set_ylabel(f"{line.name} (%)")
        ax2.set_ylim(0, max(line.values) * 1.5)
        ax1.legend([bars, ax2.get_lines()[0]],
                   [f"{bar.name} (좌축)", f"{line.name} (우축)"], loc="upper left")
    else:
        raise ValueError(spec.kind)

    fig.suptitle(spec.title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ── QA 생성 ───────────────────────────────────────────────────────────────
def _cat(q: str, gt: str, qtype: str) -> dict:
    return {"question": q, "answer_gt": gt, "answer_value": None, "answer_unit": None,
            "eval_type": "category", "question_type": qtype}


def _num(q: str, value: float, unit: str, noun: str, qtype: str,
         strict: bool, approx: bool = False) -> dict:
    return {"question": q, "answer_gt": ans_str(value, unit, noun, approx),
            "answer_value": round(value, 4), "answer_unit": unit,
            "eval_type": "numeric_strict" if strict else "numeric_relaxed",
            "question_type": qtype}


def _order(vals: list[float]) -> list[int]:
    return sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)


def _closest_pair(vals: list[float]) -> tuple[int, int]:
    best, bi, bj = float("inf"), 0, 1
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            d = abs(vals[i] - vals[j])
            if 0 < d < best:
                best, bi, bj = d, i, j
    return bi, bj


def qa_single(spec: ChartSpec, rng: random.Random) -> list[dict]:
    cats, vals = spec.categories, spec.series[0].values
    order = _order(vals)
    out: list[dict] = []
    t = spec.title

    # unit_convert (조/억 도메인) — 최우선 타깃, strict
    if spec.conv:
        i = order[rng.randint(0, min(2, len(order) - 1))]
        gv = convert(vals[i], spec.unit, spec.conv)
        out.append(_num(f"'{t}'에서 {cats[i]}의 값을 {spec.conv} {spec.noun} 단위로 나타내면 얼마인가요?",
                        gv, spec.conv, spec.noun, "unit_convert", strict=True))

    # argmax (값 유일할 때만)
    if vals[order[0]] != vals[order[1]]:
        out.append(_cat(f"'{t}'에서 값이 가장 큰 항목은 무엇인가요?", cats[order[0]], "argmax"))

    # rank_kth (경계 값이 유일할 때만)
    if len(vals) >= 4:
        k = rng.choice([2, 3])
        a, b, c = order[k - 2], order[k - 1], order[k]
        if vals[a] != vals[b] != vals[c]:
            out.append(_cat(f"'{t}'에서 값이 {k}번째로 큰 항목은 무엇인가요?", cats[b], "rank_kth"))

    # narrow_compare
    i, j = _closest_pair(vals)
    if vals[i] != vals[j]:
        big = cats[i] if vals[i] > vals[j] else cats[j]
        out.append(_cat(f"{cats[i]}와(과) {cats[j]} 중 값이 더 큰 것은 무엇인가요?", big, "narrow_compare"))

    # value_read (라벨 없으면 근사·relaxed)
    i = order[rng.randint(0, len(order) - 1)]
    approx = not spec.show_labels
    q = f"'{t}'에서 {cats[i]}의 값은 약 얼마인가요?" if approx else f"'{t}'에서 {cats[i]}의 값은 얼마인가요?"
    out.append(_num(q, vals[i], spec.unit, spec.noun, "value_read", strict=False, approx=approx))
    return out


def qa_multi(spec: ChartSpec, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    cats, series = spec.categories, spec.series
    t = spec.title
    ci = rng.randrange(len(cats))
    at = [(s.name, s.values[ci]) for s in series]
    hi = max(at, key=lambda p: p[1])
    lo = min(at, key=lambda p: p[1])
    if hi[1] != lo[1]:
        out.append(_cat(f"'{t}'에서 {cats[ci]}에 값이 가장 큰 항목은 무엇인가요?", hi[0], "cross_series"))
    s = rng.choice(series)
    approx = not spec.show_labels
    out.append(_num(f"'{t}'에서 {cats[ci]}의 {s.name} 값은 약 얼마인가요?" if approx
                    else f"'{t}'에서 {cats[ci]}의 {s.name} 값은 얼마인가요?",
                    s.values[ci], spec.unit, spec.noun, "value_read", strict=False, approx=approx))
    # 두 시리즈가 근접한 지점에서 대소 비교
    if len(series) >= 2:
        gaps = [(abs(series[0].values[k] - series[1].values[k]), k) for k in range(len(cats))]
        _, k = min(gaps)
        if series[0].values[k] != series[1].values[k]:
            big = series[0].name if series[0].values[k] > series[1].values[k] else series[1].name
            out.append(_cat(f"'{t}'에서 {cats[k]}에는 {series[0].name}와(과) {series[1].name} 중 "
                            f"무엇이 더 큰가요?", big, "narrow_compare"))
    return out


def qa_pie(spec: ChartSpec, rng: random.Random) -> list[dict]:
    labels, sh = spec.categories, spec.series[0].values
    order = _order(sh)
    out = [_cat(f"'{spec.title}'에서 점유율이 가장 높은 항목은 무엇인가요?", labels[order[0]], "argmax")]
    i, j = order[0], order[1]
    out.append(_num(f"{labels[i]}와(과) {labels[j]}의 점유율을 합치면 몇 %인가요?",
                    sh[i] + sh[j], "%", "", "sum", strict=False))
    out.append(_num(f"{labels[i]}는 {labels[j]}보다 점유율이 몇 %p 더 높은가요?",
                    sh[i] - sh[j], "%", "", "diff", strict=False))
    if len(sh) >= 3 and sh[order[1]] != sh[order[2]]:
        out.append(_cat(f"'{spec.title}'에서 점유율이 2번째로 높은 항목은 무엇인가요?",
                        labels[order[1]], "rank_kth"))
    return out


def qa_stacked(spec: ChartSpec, rng: random.Random) -> list[dict]:
    out: list[dict] = []
    cats, series = spec.categories, spec.series
    t = spec.title
    ci = rng.randrange(len(cats))
    total = sum(s.values[ci] for s in series)
    # 합산 + (조/억이면) 단위 환산
    if spec.conv:
        gv = convert(total, spec.unit, spec.conv)
        out.append(_num(f"'{t}'에서 {cats[ci]}의 전체 합계를 {spec.conv} {spec.noun} 단위로 나타내면 얼마인가요?",
                        gv, spec.conv, spec.noun, "unit_convert", strict=True))
    else:
        out.append(_num(f"'{t}'에서 {cats[ci]}의 전체 합계는 얼마인가요?",
                        total, spec.unit, spec.noun, "sum", strict=False))
    # 어느 구성요소가 더 큰가
    at = [(s.name, s.values[ci]) for s in series]
    hi, lo = max(at, key=lambda p: p[1]), min(at, key=lambda p: p[1])
    if hi[1] != lo[1]:
        out.append(_cat(f"'{t}'에서 {cats[ci]}에는 어느 항목의 값이 가장 큰가요?", hi[0], "cross_series"))
    # 특정 구성요소 최대 연도
    s = rng.choice(series)
    o = _order(s.values)
    if s.values[o[0]] != s.values[o[1]]:
        out.append(_cat(f"'{t}'에서 {s.name}의 값이 가장 큰 항목은 무엇인가요?", cats[o[0]], "argmax"))
    return out


def qa_dual(spec: ChartSpec, rng: random.Random) -> list[dict]:
    cats = spec.categories
    bar, line = spec.series[0], spec.series[1]
    t = spec.title
    out: list[dict] = []
    lo = _order(line.values)
    bo = _order(bar.values)
    if line.values[lo[0]] != line.values[lo[1]]:
        out.append(_cat(f"'{t}'에서 {line.name}(우축)이 가장 높은 항목은 무엇인가요?", cats[lo[0]], "dual_trap"))
    if bar.values[bo[0]] != bar.values[bo[1]]:
        out.append(_cat(f"'{t}'에서 {bar.name}(좌축)이 가장 큰 항목은 무엇인가요?", cats[bo[0]], "argmax"))
    i = rng.randrange(len(cats))
    approx = not spec.show_labels
    out.append(_num(f"'{t}'에서 {cats[i]}의 {bar.name}은 약 얼마인가요?" if approx
                    else f"'{t}'에서 {cats[i]}의 {bar.name}은 얼마인가요?",
                    bar.values[i], spec.unit, spec.noun, "value_read", strict=False, approx=approx))
    return out


# ── 차트 스펙 샘플러 ───────────────────────────────────────────────────────
def make_spec(rng: random.Random, kind: str) -> ChartSpec:
    if kind in ("bar", "dense"):
        d = rng.choice([x for x in BAR_DOMAINS if (kind != "dense" or x["pool"] in ("regions", "depts", "companies"))])
        n = rng.randint(10, min(14, len(POOLS[d["pool"]]))) if kind == "dense" else rng.randint(4, 6)
        cats = pick_categories(rng, d["pool"], n)
        vals = sample_values(rng, len(cats), d["lo"], d["hi"], d["step"])
        if rng.random() < 0.5:
            inject_near_tie(rng, vals, d["step"])
        return ChartSpec(kind, d["title"], cats, [Series("", vals)], d["unit"], d["noun"],
                         conv=d["conv"], y_label=f"{d['title'].split('별')[-1].strip()} ({d['unit']} {d['noun']})".strip())

    if kind == "grouped":
        cats = pick_categories(rng, "years", rng.randint(4, 5))
        names = rng.sample(["온라인", "오프라인", "직영점", "대리점", "해외"], 2)
        unit, noun, step = rng.choice([("천", "대", 1), ("억", "원", 10), ("만", "명", 1)])
        series = [Series(nm, sample_values(rng, len(cats), 20, 90 if unit == "천" else 900, step)) for nm in names]
        return ChartSpec(kind, "채널별 실적 비교", cats, series, unit, noun,
                         y_label=f"({unit} {noun})")

    if kind == "line":
        cats = [f"{m}월" for m in range(1, rng.randint(7, 13))]
        theme = rng.choice(["기온", "판매"])
        if theme == "기온":
            names = rng.sample(["서울", "부산", "대구", "인천", "제주", "강릉"], 3)
            unit, noun = "°C", ""
            series = [Series(nm, sample_values(rng, len(cats), 0, 30, 1)) for nm in names]
            title, ylab = "도시별 월평균 기온", "기온 (°C)"
        else:
            names = rng.sample(POOLS["products"], 3)
            unit, noun = "천", "대"
            series = [Series(nm, sample_values(rng, len(cats), 10, 80, 1)) for nm in names]
            title, ylab = "제품별 월간 판매량", "판매량 (천 대)"
        return ChartSpec(kind, title, cats, series, unit, noun, y_label=ylab)

    if kind == "pie":
        n = rng.randint(4, 6)
        labels = pick_categories(rng, "companies", n)
        raw = sample_values(rng, n, 5, 30, 1)
        s = sum(raw)
        sh = [round(v / s * 100) for v in raw]
        sh[0] += 100 - sum(sh)  # 합계 100 보정
        return ChartSpec(kind, "시장 점유율", labels, [Series("", [float(x) for x in sh])], "%", "")

    if kind == "stacked":
        cats = pick_categories(rng, "years", rng.randint(3, 5))
        pair = rng.choice([("수출", "내수"), ("국내", "해외"), ("상반기", "하반기")])
        unit, noun, lo, hi, step, conv = rng.choice([
            ("조", "원", 5, 20, 1, "억"), ("억", "원", 200, 900, 10, None)])
        series = [Series(nm, sample_values(rng, len(cats), lo, hi, step)) for nm in pair]
        return ChartSpec(kind, "구성별 매출", cats, series, unit, noun, conv=conv, y_label=f"({unit} {noun})")

    if kind == "dual":
        cats = pick_categories(rng, "years", 5)
        rev = sample_values(rng, 5, 800, 1500, 10)
        margin = sample_values(rng, 5, 6, 14, 0.1)
        return ChartSpec(kind, "매출과 영업이익률", cats,
                         [Series("매출", rev), Series("영업이익률", margin)], "억", "원",
                         y_label="매출 (억 원)")

    raise ValueError(kind)


KINDS = ["dense", "bar", "stacked", "dual", "grouped", "line", "pie"]
KIND_WEIGHTS = [25, 20, 13, 10, 12, 12, 8]
QA_DISPATCH = {"bar": qa_single, "dense": qa_single, "grouped": qa_multi,
               "line": qa_multi, "pie": qa_pie, "stacked": qa_stacked, "dual": qa_dual}


def decide_labels(rng: random.Random, kind: str, qa: list[dict]) -> bool:
    """pie는 항상 라벨. unit_convert 차트는 절반만 라벨(조 값이 축 스텝에 맞춰
    있어 무라벨도 읽을 수 있음). 그 외에는 대부분 무라벨(축 판독 강제).
    → 전체 무라벨 비율 ≥50% 목표 (Day1 약점 #2)."""
    if kind == "pie":
        return True
    if any(q["question_type"] == "unit_convert" for q in qa):
        return rng.random() < 0.5      # 50% 라벨
    return rng.random() >= 0.65         # 65% 무라벨


def build_split(rng: random.Random, split: str, n_charts: int) -> list[dict]:
    split_dir = OUT / split
    split_dir.mkdir(parents=True, exist_ok=True)
    qa_all: list[dict] = []
    for idx in range(n_charts):
        kind = rng.choices(KINDS, weights=KIND_WEIGHTS)[0]
        spec = make_spec(rng, kind)
        qa = QA_DISPATCH[kind](spec, rng)
        spec.show_labels = decide_labels(rng, kind, qa)

        img_rel = f"data/synth/{split}/{idx:05d}.png"
        render(spec, ROOT / img_rel)

        # 타깃 유형(unit_convert·rank_kth·narrow_compare·dual_trap) 우선 정렬 후 2~4개 채택
        target = {"unit_convert", "rank_kth", "narrow_compare", "dual_trap"}
        qa.sort(key=lambda q: 0 if q["question_type"] in target else 1)
        keep = qa[:rng.choice([2, 3, 3, 4])]
        for qi, q in enumerate(keep, start=1):
            qa_all.append({
                "id": f"{split}_{idx:05d}-q{qi}",
                "image": img_rel,
                **q,
                "has_value_labels": spec.show_labels,
                "chart_kind": kind,
            })
    return qa_all


def summarize(qa: list[dict]) -> dict:
    def counts(key):
        c: dict[str, int] = {}
        for q in qa:
            c[q[key]] = c.get(q[key], 0) + 1
        return dict(sorted(c.items(), key=lambda x: -x[1]))
    imgs = {q["image"] for q in qa}
    no_label = len({q["image"] for q in qa if not q["has_value_labels"]})
    return {
        "n_qa": len(qa),
        "n_charts": len(imgs),
        "no_label_charts": no_label,
        "no_label_ratio": round(no_label / len(imgs), 3),
        "question_type": counts("question_type"),
        "eval_type": counts("eval_type"),
        "chart_kind": counts("chart_kind"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charts", type=int, default=1600, help="총 차트 수 (train+val)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=20260720)
    args = ap.parse_args()

    setup_korean_font()
    OUT.mkdir(parents=True, exist_ok=True)

    n_val = max(1, int(args.charts * args.val_frac))
    n_train = args.charts - n_val
    t0 = time.time()

    # train/val은 서로 다른 시드 오프셋 → 데이터 누수 없음
    train = build_split(random.Random(args.seed), "train", n_train)
    val = build_split(random.Random(args.seed + 999), "val", n_val)

    (OUT / "train.json").write_text(json.dumps(train, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "val.json").write_text(json.dumps(val, ensure_ascii=False, indent=1), encoding="utf-8")

    manifest = {
        "seed": args.seed,
        "generator": "scripts/make_chart_dataset.py",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_sec": round(time.time() - t0, 1),
        "targets": {
            "unit_convert_strict": "조↔억↔만 환산 (Day1 약점 #1)",
            "no_label_ratio_goal": ">=0.5 (Day1 약점 #2)",
            "narrow_compare + rank_kth": "순위/근소 비교 (Day1 약점 #2,#3)",
        },
        "train": summarize(train),
        "val": summarize(val),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    s = manifest["train"]
    print(f"[done] {args.charts} charts, {s['n_qa'] + manifest['val']['n_qa']} QA "
          f"in {manifest['elapsed_sec']}s")
    print(f"  train: {s['n_qa']} QA / {s['n_charts']} charts, 무라벨 {s['no_label_ratio']:.0%}")
    print(f"  유형 분포: {json.dumps(s['question_type'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()
