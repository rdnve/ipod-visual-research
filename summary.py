"""Сводка по роликам: числа для таблицы в исследовании.

    mean_attention_pct — средняя доля внимания на белой детали по ключевым кадрам ролика, %;
    pooled_lift        — сумма внимания / сумма площади по кадрам ролика (во сколько раз
                         внимание больше площади). Так же считает chart.py.

Пример:
    python summary.py --results out/results.csv --out out/summary.csv
"""
import argparse
import csv
from collections import OrderedDict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("out/results.csv"))
    ap.add_argument("--out", type=Path, default=Path("out/summary.csv"))
    args = ap.parse_args()

    rows: "OrderedDict[str, list[tuple[float, float]]]" = OrderedDict()
    with open(args.results, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.setdefault(Path(r["video"]).stem, []).append((float(r["area_pct"]), float(r["attention_pct"])))

    out = []
    for name in sorted(rows):
        area = sum(a for a, _ in rows[name])
        attention = sum(t for _, t in rows[name])
        out.append({
            "video": name,
            "frames": len(rows[name]),
            "mean_attention_pct": f"{attention / len(rows[name]):.3f}",
            "pooled_lift": f"{attention / area:.2f}" if area > 0 else "",
        })
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(f"{r['video']:24s} внимание {r['mean_attention_pct']:>7s}%   ×{r['pooled_lift'] or '—'}")
    print(f"Готово: {args.out}")


if __name__ == "__main__":
    main()
