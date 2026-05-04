"""
evaluate_joint.py — Tạo bảng benchmark so sánh tất cả experiments.

Đọc results.json từ mỗi experiment output dir → tổng hợp thành bảng markdown.
"""

import json
import sys
from pathlib import Path
from tabulate import tabulate

from .config_joint import JOINT_OUTPUT_DIR


def load_results(output_dir: Path) -> list:
    """Load tất cả results.json từ các subdirectories."""
    results = []
    for exp_dir in sorted(output_dir.iterdir()):
        if exp_dir.is_dir():
            result_file = exp_dir / "results.json"
            if result_file.exists():
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append(data)
    return results


def generate_benchmark_table(results: list) -> str:
    """Tạo bảng so sánh markdown."""
    headers = [
        "Experiment", "Backbone", "Best Epoch",
        "Topic F1", "Intent Acc", "NER F1", "Sem Acc",
    ]
    rows = []
    for r in results:
        tm = r.get("test_metrics", {})
        rows.append([
            r.get("experiment", "?"),
            r.get("backbone", "?"),
            r.get("best_epoch", "?"),
            f"{tm.get('topic_macro_f1', 0):.4f}",
            f"{tm.get('intent_accuracy', 0):.4f}",
            f"{tm.get('ner_f1', 0):.4f}",
            f"{tm.get('semantic_accuracy', 0):.4f}",
        ])

    return tabulate(rows, headers=headers, tablefmt="github")


def main():
    print("=" * 60)
    print("📊 KEHN Benchmark Results")
    print("=" * 60)

    results = load_results(JOINT_OUTPUT_DIR)
    if not results:
        print(f"\n⚠️ No results found in {JOINT_OUTPUT_DIR}")
        print("   Run train_joint.py first!")
        return

    table = generate_benchmark_table(results)
    print(f"\n{table}")

    # Save to markdown
    report_path = JOINT_OUTPUT_DIR / "benchmark_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# KEHN Benchmark Results\n\n")
        f.write(table)
        f.write("\n\n## Detailed Results\n\n")
        for r in results:
            f.write(f"### {r['experiment']}\n")
            f.write(f"- Backbone: {r['backbone']}\n")
            f.write(f"- Best epoch: {r['best_epoch']}\n")
            tm = r.get("test_metrics", {})
            f.write(f"- Topic Macro-F1: **{tm.get('topic_macro_f1', 0):.4f}**\n")
            f.write(f"- Intent Accuracy: {tm.get('intent_accuracy', 0):.4f}\n")
            f.write(f"- NER Entity-F1: {tm.get('ner_f1', 0):.4f}\n")
            f.write(f"- Semantic Accuracy: {tm.get('semantic_accuracy', 0):.4f}\n\n")

    print(f"\n💾 Report saved to {report_path}")


if __name__ == "__main__":
    main()
