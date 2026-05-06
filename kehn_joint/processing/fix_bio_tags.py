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
    ner2id = getattr(cfg, "NER2ID")

    in_path = Path(args.input)
    out_path = Path(args.output)

    rule_a_fixes = 0
    rule_b_fixes = 0
    skipped_len_mismatch = 0
    total = 0
    written = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin, desc="fix_bio_tags", unit="lines")):
            s = line.strip()
            if not s:
                continue
            total += 1
            rec = json.loads(s)

            words = rec.get("words")
            ner_tags = rec.get("ner_tags")
            ner_tag_ids = rec.get("ner_tag_ids")

            if not isinstance(words, list) or not isinstance(ner_tags, list) or not isinstance(ner_tag_ids, list):
                print(f"[SKIP] idx={idx} reason=missing_or_invalid_fields", file=sys.stdout)
                skipped_len_mismatch += 1
                continue

            fixed = list(ner_tags)

            # Rule B: starting with I-X -> B-X
            if fixed and isinstance(fixed[0], str) and fixed[0].startswith("I-"):
                fixed[0] = "B-" + fixed[0][2:]
                rule_b_fixes += 1

            # Rule A: I-X preceded by O or different entity type -> B-X
            for i in range(1, len(fixed)):
                cur = fixed[i]
                prev = fixed[i - 1]
                if not isinstance(cur, str) or not isinstance(prev, str):
                    continue
                if cur.startswith("I-"):
                    cur_t = cur[2:]
                    if prev == "O" or (prev.startswith(("B-", "I-")) and prev[2:] != cur_t) or (not prev.startswith(("B-", "I-"))):
                        fixed[i] = "B-" + cur_t
                        rule_a_fixes += 1

            if not (len(words) == len(fixed) == len(ner_tag_ids)):
                rec_id = rec.get("id")
                print(
                    f"[SKIP] idx={idx} id={rec_id!r} reason=len_mismatch "
                    f"len(words)={len(words)} len(ner_tags)={len(fixed)} len(ner_tag_ids)={len(ner_tag_ids)}",
                    file=sys.stdout,
                )
                skipped_len_mismatch += 1
                continue

            rec["ner_tags"] = fixed
            rec["ner_tag_ids"] = [int(ner2id.get(t, 0)) for t in fixed]

            written += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[DONE] fix_bio_tags total={total} written={written} ruleA={rule_a_fixes} ruleB={rule_b_fixes} skipped_len_mismatch={skipped_len_mismatch}",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
