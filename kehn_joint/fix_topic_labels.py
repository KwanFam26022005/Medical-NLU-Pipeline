import argparse
import importlib.util
import json
import sys
from pathlib import Path

from tqdm import tqdm


def _load_config(config_path: str):
    config_path = str(Path(config_path).resolve())
    spec = importlib.util.spec_from_file_location("kehn_joint_config_joint", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load config from: {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = _load_config(args.config)
    topic2id = getattr(cfg, "TOPIC2ID")
    n_topic = getattr(cfg, "N_TOPIC")

    in_path = Path(args.input)
    out_path = Path(args.output)

    remapped = 0
    dropped = 0
    passed = 0
    total = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin, desc="fix_topic_labels", unit="lines")):
            s = line.strip()
            if not s:
                continue
            total += 1
            rec = json.loads(s)

            topic = rec.get("topic_label")
            if topic == "traditional_medicine":
                dropped += 1
                print(f"[DROP] idx={idx} topic_label=traditional_medicine", file=sys.stdout)
                continue

            if topic == "oncology":
                rec["topic_label"] = "internal_medicine"
                topic = rec["topic_label"]
                remapped += 1

            if topic not in topic2id:
                raise ValueError(f"Topic not in TOPIC2ID at idx={idx}: topic_label={topic!r}")

            rec["topic_label_id"] = int(topic2id[topic])

            tid = rec["topic_label_id"]
            if not (0 <= tid < int(n_topic)):
                raise ValueError(f"topic_label_id out of range at idx={idx}: {tid} (n_topic={n_topic})")

            passed += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[DONE] fix_topic_labels total={total} remapped={remapped} dropped={dropped} asserted_ok={passed}",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
