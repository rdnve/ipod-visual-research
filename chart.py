"""Шаг 3. Два итоговых графика по роликам.

    out/chart_attention.png — какую долю внимания забирает белая деталь (в среднем по кадрам ролика);
    out/chart_lift.png      — во сколько раз внимания к белому больше, чем его площадь
                              (сумма внимания / сумма площади по кадрам ролика: крошечные
                              детали не раздувают число).

Пример:
    python chart.py --results out/results.csv --out out
"""
import argparse
import csv
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK_2, GRID, BAR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def pretty(stem: str) -> str:
    """'02_2003_jet' -> '2003\\njet'"""
    stem = re.sub(r"^\d+_", "", stem)
    year, _, rest = stem.partition("_")
    return f"{year}\n{rest.replace('_', ' ')}" if rest else year


def bar_chart(names, values, empty, title, subtitle, ylabel, out, ref=None):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(range(len(names)), values, width=0.6, color=BAR, zorder=3)
    top = max(values) if max(values) > 0 else 1
    for i, is_empty in enumerate(empty):
        if is_empty:
            ax.text(i, top * 0.02, "нет\nбелой\nдетали", ha="center", va="bottom", color=INK_2, fontsize=8)
    if ref is not None:
        ax.axhline(ref, color=INK_2, lw=1, ls=(0, (4, 3)), zorder=4)
    ax.set_xticks(range(len(names)), [pretty(n) for n in names], color=INK, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_title(title, loc="left", color=INK, fontsize=13, pad=26)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=9, va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=INK_2, length=0)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"Готово: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("out/results.csv"))
    ap.add_argument("--out", type=Path, default=Path("out"), help="папка для графиков")
    args = ap.parse_args()

    rows: "OrderedDict[str, list[tuple[float, float]]]" = OrderedDict()
    with open(args.results, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.setdefault(Path(r["video"]).stem, []).append((float(r["area_pct"]), float(r["attention_pct"])))
    if not rows:
        raise SystemExit("results.csv пустой")
    names = sorted(rows)  # номер в начале имени файла = хронология
    area = [sum(a for a, _ in rows[n]) for n in names]
    attention_mean = [sum(t for _, t in rows[n]) / len(rows[n]) for n in names]
    lift = [sum(t for _, t in rows[n]) / a if a > 0 else 0.0 for n, a in zip(names, area)]
    empty = [a == 0 for a in area]

    args.out.mkdir(parents=True, exist_ok=True)
    bar_chart(names, attention_mean, empty,
              "Сколько внимания забирает белая деталь",
              "Средняя доля внимания на проводе, наушниках и плеере по ключевым кадрам ролика, %.",
              "доля внимания, %", args.out / "chart_attention.png")
    bar_chart(names, lift, empty,
              "Во сколько раз белая деталь забирает больше внимания, чем занимает места",
              "Пунктир ×1 — внимания ровно столько, сколько места. Выше — белое притягивает взгляд.",
              "внимание / площадь, раз", args.out / "chart_lift.png", ref=1)


if __name__ == "__main__":
    main()
