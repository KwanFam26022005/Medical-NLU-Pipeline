import argparse
import importlib.util
import json
import sys
from pathlib import Path

from tqdm import tqdm


def _load_config(config_path: str):
    p = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("kehn_joint_config_joint", str(p))
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load config from: {p}")
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
    ner2id = dict(getattr(cfg, "NER2ID"))

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"--input not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_missing_text = 0
    fixed_missing_ner_tag_ids = 0
    bio_fixes = 0
    len_repaired = 0
    dropped_short = 0
    dropped_unfixable = 0
    total = 0
    written = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin, desc="repair_ready_jsonl", unit="lines")):
            s = line.strip()
            if not s:
                continue
            total += 1
            rec = json.loads(s)

            words = rec.get("words")
            ner_tags = rec.get("ner_tags")
            ner_tag_ids = rec.get("ner_tag_ids")

            if not isinstance(words, list) or not isinstance(ner_tags, list):
                print(f"[DROP] idx={idx} reason=missing_words_or_ner_tags id={rec.get('id')!r}", file=sys.stdout)
                dropped_unfixable += 1
                continue

            # A6: fill missing text
            if "text" not in rec or not isinstance(rec.get("text"), str) or rec.get("text") == "":
                rec["text"] = " ".join(words)
                fixed_missing_text += 1

            # A2.3: length sync repair
            if isinstance(ner_tag_ids, list):
                pass
            else:
                ner_tag_ids = None

            if ner_tag_ids is None:
                rec["ner_tag_ids"] = [int(ner2id.get(t, 0)) for t in ner_tags]
                fixed_missing_ner_tag_ids += 1
                ner_tag_ids = rec["ner_tag_ids"]

            if not (len(words) == len(ner_tags) == len(ner_tag_ids)):
                m = min(len(words), len(ner_tags), len(ner_tag_ids))
                if m <= 0:
                    print(f"[DROP] idx={idx} reason=empty_after_len_sync id={rec.get('id')!r}", file=sys.stdout)
                    dropped_unfixable += 1
                    continue
                rec["words"] = words[:m]
                rec["ner_tags"] = ner_tags[:m]
                rec["ner_tag_ids"] = ner_tag_ids[:m]
                words = rec["words"]
                ner_tags = rec["ner_tags"]
                ner_tag_ids = rec["ner_tag_ids"]
                len_repaired += 1

            # A2: BIO repair (O->I-*, and also invalid I transitions)
            fixed_tags = list(ner_tags)
            if fixed_tags and isinstance(fixed_tags[0], str) and fixed_tags[0].startswith("I-"):
                fixed_tags[0] = "B-" + fixed_tags[0][2:]
                bio_fixes += 1

            for i in range(1, len(fixed_tags)):
                cur = fixed_tags[i]
                prev = fixed_tags[i - 1]
                if not (isinstance(cur, str) and isinstance(prev, str)):
                    continue
                if cur.startswith("I-"):
                    cur_t = cur[2:]
                    if prev == "O" or (prev.startswith(("B-", "I-")) and prev[2:] != cur_t) or (not prev.startswith(("B-", "I-"))):
                        fixed_tags[i] = "B-" + cur_t
                        bio_fixes += 1

            rec["ner_tags"] = fixed_tags
            rec["ner_tag_ids"] = [int(ner2id.get(t, 0)) for t in fixed_tags]

            # A3.2: drop chunks < 10 words
            if len(rec["words"]) < 10:
                print(f"[DROP] idx={idx} reason=short_lt_10 id={rec.get('id')!r} len={len(rec['words'])}", file=sys.stdout)
                dropped_short += 1
                continue

            written += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        "[DONE] repair_ready_jsonl "
        f"total={total} written={written} "
        f"fixed_missing_text={fixed_missing_text} fixed_missing_ner_tag_ids={fixed_missing_ner_tag_ids} "
        f"bio_fixes={bio_fixes} len_repaired={len_repaired} dropped_short={dropped_short} dropped_unfixable={dropped_unfixable}",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()

