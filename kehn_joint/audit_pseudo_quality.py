import argparse
import json


def audit_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokens = 0
    stuck_punct_tokens = 0
    entity_tokens_with_punct = 0
    bio_invalid_i_tags = 0
    exact_half_dup_samples = 0

    for row in data:
        words = row.get("words", [])
        tags = row.get("ner_tags", [])
        tokens += len(words)

        half = len(words) // 2
        if half >= 3 and words[:half] == words[half : 2 * half]:
            exact_half_dup_samples += 1

        for i, w in enumerate(words):
            has_alnum = any(ch.isalnum() for ch in w)
            has_punct = any((not ch.isalnum()) and (not ch.isspace()) for ch in w)
            if has_alnum and has_punct:
                stuck_punct_tokens += 1
            tag = tags[i] if i < len(tags) else "O"
            if tag != "O" and has_punct:
                entity_tokens_with_punct += 1

        for i, tag in enumerate(tags):
            if tag.startswith("I-"):
                if i == 0 or tags[i - 1] == "O" or tags[i - 1][2:] != tag[2:]:
                    bio_invalid_i_tags += 1

    return {
        "samples": len(data),
        "tokens": tokens,
        "stuck_punct_tokens": stuck_punct_tokens,
        "stuck_punct_ratio": stuck_punct_tokens / max(tokens, 1),
        "entity_tokens_with_punct": entity_tokens_with_punct,
        "bio_invalid_i_tags": bio_invalid_i_tags,
        "exact_half_dup_samples": exact_half_dup_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pseudo data quality for KEHN")
    parser.add_argument("--input", required=True, help="Input JSON file")
    parser.add_argument("--output", default=None, help="Optional output JSON report path")
    args = parser.parse_args()

    report = audit_json(args.input)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
