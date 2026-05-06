import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--stride", type=int, default=72)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    window = int(args.window)
    stride = int(args.stride)

    long_records = 0
    total_chunks = 0
    dropped_chunks = 0
    total = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin, desc="chunk_long_samples", unit="lines")):
            s = line.strip()
            if not s:
                continue
            total += 1
            rec = json.loads(s)

            words = rec.get("words")
            ner_tags = rec.get("ner_tags")

            if not isinstance(words, list) or not isinstance(ner_tags, list):
                print(f"[PASS] idx={idx} reason=missing_words_or_ner_tags", file=sys.stdout)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            if len(words) <= 128:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            long_records += 1

            parent_id = rec.get("id")
            base_id = parent_id if isinstance(parent_id, str) and parent_id else str(idx)

            n = len(words)
            chunk_n = 0
            for start in range(0, n, stride):
                end = start + window
                if start >= n:
                    break
                w_slice = words[start:end]
                t_slice = ner_tags[start:end]

                if len(w_slice) < 10:
                    dropped_chunks += 1
                    print(f"[DROP_CHUNK] idx={idx} chunk={chunk_n} reason=short len={len(w_slice)}", file=sys.stdout)
                    chunk_n += 1
                    continue

                out = {
                    "id": f"{base_id}_chunk_{chunk_n}",
                    "words": w_slice,
                    "ner_tags": t_slice,
                    "intent_label": rec.get("intent_label"),
                    "intent_label_id": rec.get("intent_label_id"),
                    "topic_label": rec.get("topic_label"),
                    "topic_label_id": rec.get("topic_label_id"),
                    "topic_confidence": rec.get("topic_confidence"),
                    "source": rec.get("source"),
                }

                total_chunks += 1
                fout.write(json.dumps(out, ensure_ascii=False) + "\n")
                chunk_n += 1

                if end >= n:
                    break

    print(
        f"[DONE] chunk_long_samples total={total} long_records={long_records} chunks={total_chunks} dropped_chunks={dropped_chunks}",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
