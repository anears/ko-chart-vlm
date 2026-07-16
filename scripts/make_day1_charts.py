"""Day 1 진단용 한국어 차트 8장 + QA(정답 포함) 생성.

각 차트는 VLM이 자주 틀리는 유형 하나씩을 겨냥한다:
값 읽기, 범례 매칭, 비율, 누적 합산, 이중축 혼동, 한국어 단위 환산, 조밀한 막대 판독.
출력: data/day1_charts/*.png + qa.json (완전 결정적 — 언제 다시 돌려도 동일)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "day1_charts"
FONT = ROOT / "assets" / "fonts" / "NanumGothic-Regular.ttf"


def setup_korean_font() -> None:
    fm.fontManager.addfont(str(FONT))
    family = fm.FontProperties(fname=str(FONT)).get_name()
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)


def bar_labels(ax, bars, fmt="{:.0f}") -> None:
    for b in bars:
        ax.annotate(
            fmt.format(b.get_height()),
            (b.get_x() + b.get_width() / 2, b.get_height()),
            ha="center", va="bottom", fontsize=9,
        )


def chart1_bar_simple():
    years = ["2020", "2021", "2022", "2023", "2024"]
    sales = [120, 145, 150, 180, 210]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(years, sales, color="#4C72B0")
    bar_labels(ax, bars)
    ax.set_title("연도별 매출 추이")
    ax.set_ylabel("매출 (억 원)")
    save(fig, "01_bar_simple.png")
    return [
        ("2023년 매출은 몇 억 원인가요?", "180억 원", "숫자 180"),
        ("매출이 가장 컸던 해는 언제인가요?", "2024년", "2024"),
    ]


def chart2_bar_grouped():
    quarters = ["1분기", "2분기", "3분기", "4분기"]
    online = [30, 36, 45, 52]
    offline = [38, 35, 33, 30]
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - 0.2, online, 0.4, label="온라인", color="#4C72B0")
    b2 = ax.bar(x + 0.2, offline, 0.4, label="오프라인", color="#DD8452")
    bar_labels(ax, b1)
    bar_labels(ax, b2)
    ax.set_xticks(x, quarters)
    ax.set_title("분기별 판매 채널 비교")
    ax.set_ylabel("판매량 (천 대)")
    ax.legend()
    save(fig, "02_bar_grouped.png")
    return [
        ("3분기 온라인 판매량은 몇 천 대인가요?", "45천 대 (4만 5천 대)", "숫자 45"),
        ("온라인 판매량이 오프라인을 처음으로 앞선 분기는 언제인가요?", "2분기", "2분기(36>35)"),
    ]


def chart3_line_multi():
    months = np.arange(1, 13)
    seoul = [-2, 0, 6, 13, 18, 23, 26, 27, 22, 15, 8, 1]
    busan = [3, 5, 9, 14, 18, 21, 25, 26, 22, 17, 11, 5]
    jeju = [6, 7, 10, 15, 19, 22, 26, 27, 23, 18, 13, 8]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, vals, c in [("서울", seoul, "#4C72B0"), ("부산", busan, "#DD8452"), ("제주", jeju, "#55A868")]:
        ax.plot(months, vals, marker="o", label=name, color=c)
    ax.set_xticks(months, [f"{m}월" for m in months])
    ax.set_title("도시별 월평균 기온")
    ax.set_ylabel("기온 (°C)")
    ax.grid(alpha=0.3)
    ax.legend()
    save(fig, "03_line_multi.png")
    return [
        ("5월 부산의 평균 기온은 몇 도인가요?", "18°C", "부산 시리즈의 5월 값 18"),
        ("1월에 가장 따뜻한 도시는 어디인가요?", "제주", "제주(6°C)"),
    ]


def chart4_pie():
    labels = ["A전자", "B모바일", "C테크", "D산업", "기타"]
    shares = [34, 27, 18, 12, 9]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(shares, labels=labels, autopct="%1.0f%%", startangle=90,
           colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"])
    ax.set_title("2025년 국내 스마트폰 시장 점유율")
    save(fig, "04_pie.png")
    return [
        ("점유율이 가장 높은 회사와 그 비율은?", "A전자, 34%", "A전자 34%"),
        ("B모바일과 C테크의 점유율을 합치면 몇 %인가요?", "45%", "27+18=45"),
    ]


def chart5_bar_stacked():
    years = ["2021", "2022", "2023"]
    exports = [12, 15, 13]
    domestic = [8, 9, 11]
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(years, exports, label="수출", color="#4C72B0")
    b2 = ax.bar(years, domestic, bottom=exports, label="내수", color="#DD8452")
    for xi, (e, d) in enumerate(zip(exports, domestic)):
        ax.annotate(f"{e}", (xi, e / 2), ha="center", color="white")
        ax.annotate(f"{d}", (xi, e + d / 2), ha="center", color="white")
    ax.set_title("연도별 수출·내수 매출 구성")
    ax.set_ylabel("매출 (조 원)")
    ax.legend()
    save(fig, "05_bar_stacked.png")
    return [
        ("2022년 수출과 내수를 합친 전체 매출은 몇 조 원인가요?", "24조 원", "15+9=24"),
        ("내수 매출이 가장 컸던 해는 언제인가요?", "2023년", "내수 11조"),
    ]


def chart6_dual_axis():
    years = ["2020", "2021", "2022", "2023", "2024"]
    revenue = [850, 920, 1010, 980, 1150]
    margin = [8.5, 10.2, 7.8, 12.4, 11.0]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    bars = ax1.bar(years, revenue, color="#B0C4DE", label="매출 (좌축)")
    bar_labels(ax1, bars)
    ax1.set_ylabel("매출 (억 원)")
    ax1.set_ylim(0, 1400)
    ax2 = ax1.twinx()
    ax2.plot(years, margin, marker="D", color="#C44E52", label="영업이익률 (우축)")
    for x, m in zip(years, margin):
        ax2.annotate(f"{m}%", (x, m), textcoords="offset points", xytext=(0, 8),
                     ha="center", color="#C44E52", fontsize=9)
    ax2.set_ylabel("영업이익률 (%)")
    ax2.set_ylim(0, 16)
    fig.suptitle("연도별 매출과 영업이익률")
    lines = [bars, ax2.get_lines()[0]]
    ax1.legend(lines, ["매출 (좌축)", "영업이익률 (우축)"], loc="upper left")
    save(fig, "06_dual_axis.png")
    return [
        ("영업이익률이 가장 높았던 해는 언제인가요?", "2023년 (12.4%)", "매출 최고인 2024와 혼동 유도"),
        ("2022년 매출은 몇 억 원인가요?", "1,010억 원", "1010"),
    ]


def chart7_units_krw():
    depts = ["반도체", "연구개발", "바이오", "신재생에너지"]
    budget_jo = [1.2, 0.85, 0.62, 0.31]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(depts, budget_jo, color="#55A868")
    for b, v in zip(bars, budget_jo):
        ax.annotate(f"{v}조", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom")
    ax.set_title("2026년 부문별 정부 투자 예산")
    ax.set_ylabel("예산 (조 원)")
    save(fig, "07_units_krw.png")
    return [
        ("연구개발 부문 예산은 몇 억 원인가요?", "8,500억 원", "0.85조 → 억 환산"),
        ("네 부문 예산의 총합은 몇 조 원인가요?", "2.98조 원", "1.2+0.85+0.62+0.31"),
    ]


def chart8_bar_dense():
    regions = ["서울", "부산", "대구", "인천", "광주", "대전",
               "울산", "세종", "경기", "강원", "충북", "충남"]
    pops = [941, 329, 237, 299, 142, 144, 110, 39, 1363, 153, 160, 213]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(regions, pops, color="#8172B3", width=0.6)
    ax.set_title("시도별 인구 (2025년 추계)")
    ax.set_ylabel("인구 (만 명)")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save(fig, "08_bar_dense.png")
    return [
        ("울산의 인구는 약 몇 만 명인가요?", "약 110만 명", "값 라벨 없음, ±15 허용"),
        ("인구가 세 번째로 많은 시도는 어디인가요?", "부산", "경기>서울>부산 순"),
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_korean_font()
    charts = [
        ("01_bar_simple.png", chart1_bar_simple),
        ("02_bar_grouped.png", chart2_bar_grouped),
        ("03_line_multi.png", chart3_line_multi),
        ("04_pie.png", chart4_pie),
        ("05_bar_stacked.png", chart5_bar_stacked),
        ("06_dual_axis.png", chart6_dual_axis),
        ("07_units_krw.png", chart7_units_krw),
        ("08_bar_dense.png", chart8_bar_dense),
    ]
    qa = []
    for fname, fn in charts:
        for i, (q, a, note) in enumerate(fn(), start=1):
            qa.append({
                "id": f"{fname.split('_')[0]}-q{i}",
                "image": f"data/day1_charts/{fname}",
                "question": q,
                "answer_gt": a,
                "note": note,
            })
    (OUT / "qa.json").write_text(json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"charts: {len(charts)}, QA pairs: {len(qa)} -> {OUT}")


if __name__ == "__main__":
    main()
