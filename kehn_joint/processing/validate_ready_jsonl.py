import importlib.util
import json
from collections import Counter
from pathlib import Path


def _load_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("cfg_joint", str(config_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import config at: {config_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _status(pass_cond: bool, warn_cond: bool = False) -> str:
    if pass_cond:
        return "PASS"
    return "WARN" if warn_cond else "FAIL"


def main():
    workspace = Path(__file__).resolve().parent.parent
    ready_path = workspace / "kehn_joint" / "data" / "pseduo_kehn" / "merged_kehn.ready.jsonl"
    config_path = workspace / "kehn_joint" / "config_joint.py"
    alt_ready_path = workspace / "kehn_joint" / "data" / "pseduo_kehn" / "merged_kehn.ready.fixed.jsonl"
    out_path = workspace / "kehn_joint" / "data" / "pseduo_kehn" / "_readiness_validation_fixed.md"

    cfg = _load_config(config_path)
    n_topic = int(cfg.N_TOPIC)
    ner2id = dict(cfg.NER2ID)

    if alt_ready_path.exists():
        ready_path = alt_ready_path

    required_fields = [
        "text",
        "words",
        "ner_tags",
        "ner_tag_ids",
        "intent_label",
        "intent_label_id",
        "topic_label",
        "topic_label_id",
        "topic_confidence",
        "source",
    ]

    n_total = 0
    label_out_of_range = 0
    bad_topic_labels = 0
    bad_topic_label_examples = []
    bad_ner_tag_value_records = 0

    o_to_i_transitions = 0
    starts_with_i = 0
    len_mismatch_records = 0

    len_gt_128 = 0
    len_lt_10 = 0

    source_counts = Counter()
    intent_counts = Counter()
    topic_counts = Counter()

    schema_missing_records = 0
    schema_missing_examples = []

    MAX_EXAMPLES = 5

    with ready_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            n_total += 1
            rec = json.loads(s)

            missing = [k for k in required_fields if k not in rec]
            if missing:
                schema_missing_records += 1
                if len(schema_missing_examples) < MAX_EXAMPLES:
                    schema_missing_examples.append({"idx": idx, "id": rec.get("id"), "missing": missing})

            source_counts[rec.get("source")] += 1
            intent_counts[rec.get("intent_label")] += 1
            topic_counts[rec.get("topic_label")] += 1

            tlabel = rec.get("topic_label")
            if tlabel in {"oncology", "traditional_medicine"}:
                bad_topic_labels += 1
                if len(bad_topic_label_examples) < MAX_EXAMPLES:
                    bad_topic_label_examples.append({"idx": idx, "id": rec.get("id"), "topic_label": tlabel})

            tid = rec.get("topic_label_id")
            if isinstance(tid, int):
                if not (0 <= tid < n_topic):
                    label_out_of_range += 1
            else:
                label_out_of_range += 1

            words = rec.get("words")
            tags = rec.get("ner_tags")
            tag_ids = rec.get("ner_tag_ids")
            if not (isinstance(words, list) and isinstance(tags, list) and isinstance(tag_ids, list)):
                len_mismatch_records += 1
                continue
            if not (len(words) == len(tags) == len(tag_ids)):
                len_mismatch_records += 1

            if len(words) > 128:
                len_gt_128 += 1
            if len(words) < 10:
                len_lt_10 += 1

            bad_tag = False
            for t in tags:
                if t not in ner2id:
                    bad_tag = True
                    break
            if bad_tag:
                bad_ner_tag_value_records += 1

            if tags and isinstance(tags[0], str) and tags[0].startswith("I-"):
                starts_with_i += 1
            for prev, cur in zip(tags, tags[1:]):
                if prev == "O" and isinstance(cur, str) and cur.startswith("I-"):
                    o_to_i_transitions += 1

    hospital = int(source_counts.get("hospital", 0))
    vimq = int(source_counts.get("vimq", 0))
    imbalance_warn = False
    if n_total:
        imbalance_warn = (min(hospital, vimq) / n_total) < 0.30

    low_intent = {k: v for k, v in intent_counts.items() if isinstance(v, int) and v < 50}
    low_topic = {k: v for k, v in topic_counts.items() if isinstance(v, int) and v < 50}

    rows = []
    rows.append(
        (
            "A1.1 topic_label_id in [0, N_TOPIC)",
            _status(label_out_of_range == 0),
            f"N_TOPIC={n_topic}; out_of_range={label_out_of_range}",
        )
    )
    rows.append(
        (
            "A1.2 no topic_label in {oncology, traditional_medicine}",
            _status(bad_topic_labels == 0),
            f"bad_topic_labels={bad_topic_labels}; examples={bad_topic_label_examples}",
        )
    )
    rows.append(
        (
            "A1.3 all ner_tags are keys in NER2ID",
            _status(bad_ner_tag_value_records == 0),
            f"NER2ID_size={len(ner2id)}; records_with_bad_ner_tag={bad_ner_tag_value_records}",
        )
    )
    rows.append(
        (
            "A2.1 zero O → I-* transitions",
            _status(o_to_i_transitions == 0),
            f"o_to_i_transitions={o_to_i_transitions}",
        )
    )
    rows.append(
        (
            "A2.2 zero sequences starting with I-*",
            _status(starts_with_i == 0),
            f"starts_with_I={starts_with_i}",
        )
    )
    rows.append(
        (
            "A2.3 len(words)==len(ner_tags)==len(ner_tag_ids)",
            _status(len_mismatch_records == 0),
            f"len_mismatch_records={len_mismatch_records} (out of {n_total})",
        )
    )
    rows.append(
        (
            "A3.1 zero records with len(words) > 128",
            _status(len_gt_128 == 0),
            f"len_gt_128={len_gt_128}",
        )
    )
    rows.append(
        (
            "A3.2 zero records with len(words) < 10",
            _status(len_lt_10 == 0),
            f"len_lt_10={len_lt_10}",
        )
    )
    rows.append(
        (
            "A4 source distribution (hospital vs vimq)",
            _status(not imbalance_warn, warn_cond=True),
            f"total={n_total}; hospital={hospital} ({(hospital/n_total*100 if n_total else 0):.1f}%); "
            f"vimq={vimq} ({(vimq/n_total*100 if n_total else 0):.1f}%)",
        )
    )
    rows.append(
        (
            "A5.1 intent label counts",
            _status(len(low_intent) == 0, warn_cond=True),
            f"intent_counts={dict(intent_counts)}; <50={low_intent}",
        )
    )
    rows.append(
        (
            "A5.2 topic label counts",
            _status(len(low_topic) == 0, warn_cond=True),
            f"topic_unique={len(topic_counts)}; <50={low_topic}",
        )
    )
    rows.append(
        (
            "A6 schema completeness (required fields present)",
            _status(schema_missing_records == 0),
            f"missing_records={schema_missing_records}; examples={schema_missing_examples}",
        )
    )

    lines = ["| Check | Status | Detail |", "|---|---|---|"]
    for c, st, det in rows:
        det = str(det).replace("\n", " ")
        lines.append(f"| {c} | {st} | {det} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

