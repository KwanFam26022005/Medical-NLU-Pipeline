import argparse
import json
import re
import sys
from pathlib import Path

from tqdm import tqdm


def _normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([?.!,;:])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    return text.strip()


def _find_repeat_prefix(words):
    # Longest-common-prefix between words[0:] and words[shift:] for small shifts.
    # Returns (shift, lcp_len) for the best near-duplicate prefix; otherwise None.
    n = len(words)
    best_shift = None
    best_lcp = 0
    max_shift = min(80, max(0, n - 5))
    for shift in range(3, max_shift + 1):
        lcp = 0
        while (lcp < n - shift) and words[lcp] == words[shift + lcp]:
            lcp += 1
        if lcp > best_lcp:
            best_lcp = lcp
            best_shift = shift

    if best_shift is None or best_lcp < 6:
        return None

    # Heuristic: duplicate prefix tends to repeat immediately (shift ≈ prefix length)
    if abs(best_shift - best_lcp) > 6:
        return None

    return best_shift, best_lcp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    modified = 0
    flagged_partial = 0
    passed_unchanged = 0
    total = 0

    filler_tokens = {
        "Cảm_ơn", "cảm_ơn", "cám_ơn", "Xin", "xin", "chân_thành", "mong", "Mong",
        "bác_sĩ", "Bác_sĩ", "giúp", "giải_đáp", "tư_vấn", "ạ", "dạ",
        "trân_trọng", "kính", "chào", "Chào",
        ".", ",", "?", "!", ")", "(",
    }

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin, desc="hospital_text_cleaner", unit="lines")):
            s = line.strip()
            if not s:
                continue
            total += 1
            rec = json.loads(s)

            if rec.get("source") != "hospital":
                passed_unchanged += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            words = rec.get("words")
            ner_tags = rec.get("ner_tags")
            ner_tag_ids = rec.get("ner_tag_ids")
            text = rec.get("text")

            if not isinstance(words, list) or not isinstance(ner_tags, list) or not isinstance(ner_tag_ids, list) or not isinstance(text, str):
                print(f"[PASS] idx={idx} reason=missing_fields", file=sys.stdout)
                passed_unchanged += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            if not (len(words) == len(ner_tags) == len(ner_tag_ids)):
                rec_id = rec.get("id")
                print(f"[PASS] idx={idx} id={rec_id!r} reason=len_mismatch_pre", file=sys.stdout)
                passed_unchanged += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue

            orig_words = list(words)
            orig_tags = list(ner_tags)
            orig_ids = list(ner_tag_ids)

            # Step 1 — Sentence-level deduplication (remove duplicated prefix appearing again)
            rep = _find_repeat_prefix(words)
            if rep is not None:
                shift, lcp = rep
                del words[shift : shift + lcp]
                del ner_tags[shift : shift + lcp]
                del ner_tag_ids[shift : shift + lcp]

            # Step 2 — Content compression (trim tail filler while keeping entity spans)
            last_entity = -1
            for i, t in enumerate(ner_tags):
                if isinstance(t, str) and t != "O":
                    last_entity = i

            keep_min = min(len(words), 40)
            keep_upto = max(keep_min, last_entity + 4)
            keep_upto = min(len(words), keep_upto)

            # Trim long tails aggressively if no entities near the end.
            words2 = words[:keep_upto]
            tags2 = ner_tags[:keep_upto]
            ids2 = ner_tag_ids[:keep_upto]

            # Additional tail pop: social filler tokens at the very end.
            while words2 and words2[-1] in filler_tokens and len(words2) > 10:
                words2.pop()
                tags2.pop()
                ids2.pop()

            cleaned_text = _normalize_spaces(" ".join(words2))
            cleaned_words = cleaned_text.split() if cleaned_text else []

            # Step 3 — Sync words list + realign tags
            # Re-tokenize cleaned text and preserve surviving B/I spans when alignment is unambiguous.
            if cleaned_words == words2:
                rec["text"] = cleaned_text
                rec["words"] = words2
                rec["ner_tags"] = tags2
                rec["ner_tag_ids"] = ids2
                if (rec["text"] != text) or (rec["words"] != orig_words):
                    modified += 1
                else:
                    passed_unchanged += 1
            else:
                # Ambiguous: keep original and flag.
                rec["cleaning_partial"] = True
                flagged_partial += 1
                print(f"[FLAG] idx={idx} reason=ambiguous_retokenize", file=sys.stdout)
                rec["text"] = _normalize_spaces(text)
                rec["words"] = orig_words
                rec["ner_tags"] = orig_tags
                rec["ner_tag_ids"] = orig_ids

            # Step 4 — Normalize whitespace (already done for rec['text'])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(
        f"[DONE] hospital_text_cleaner total={total} modified={modified} flagged_partial={flagged_partial} passed_unchanged={passed_unchanged}",
        file=sys.stdout,
    )


if __name__ == "__main__":
    main()
