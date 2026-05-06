| Check | Status | Detail |
|---|---|---|
| A1.1 topic_label_id in [0, N_TOPIC) | PASS | N_TOPIC=16; out_of_range=0 |
| A1.2 no topic_label in {oncology, traditional_medicine} | PASS | bad_topic_labels=0; examples=[] |
| A1.3 all ner_tags are keys in NER2ID | PASS | NER2ID_size=7; records_with_bad_ner_tag=0 |
| A2.1 zero O → I-* transitions | FAIL | o_to_i_transitions=3 |
| A2.2 zero sequences starting with I-* | PASS | starts_with_I=0 |
| A2.3 len(words)==len(ner_tags)==len(ner_tag_ids) | FAIL | len_mismatch_records=79 (out of 16494) |
| A3.1 zero records with len(words) > 128 | PASS | len_gt_128=0 |
| A3.2 zero records with len(words) < 10 | FAIL | len_lt_10=1659 |
| A4 source distribution (hospital vs vimq) | PASS | total=16494; hospital=9727 (59.0%); vimq=6767 (41.0%) |
| A5.1 intent label counts | PASS | intent_counts={'method_diagnosis': 7622, 'severity': 3492, 'cause': 1157, 'treatment': 4223}; <50={} |
| A5.2 topic label counts | PASS | topic_unique=16; <50={} |
| A6 schema completeness (required fields present) | FAIL | missing_records=79; examples=[{'idx': 65, 'id': '65_chunk_0', 'missing': ['text', 'ner_tag_ids']}, {'idx': 66, 'id': '65_chunk_1', 'missing': ['text', 'ner_tag_ids']}, {'idx': 519, 'id': '518_chunk_0', 'missing': ['text', 'ner_tag_ids']}, {'idx': 520, 'id': '518_chunk_1', 'missing': ['text', 'ner_tag_ids']}, {'idx': 688, 'id': '686_chunk_0', 'missing': ['text', 'ner_tag_ids']}] |
