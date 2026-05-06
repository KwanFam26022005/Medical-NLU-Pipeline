
| Check                                                   | Status | Detail                                                                                               |
| ------------------------------------------------------- | ------ | ---------------------------------------------------------------------------------------------------- |
| A1.1 topic_label_id in [0, N_TOPIC)                     | PASS   | N_TOPIC=16; out_of_range=0                                                                           |
| A1.2 no topic_label in {oncology, traditional_medicine} | PASS   | bad_topic_labels=0; examples=[]                                                                      |
| A1.3 all ner_tags are keys in NER2ID                    | PASS   | NER2ID_size=7; records_with_bad_ner_tag=0                                                            |
| A2.1 zero O → I-* transitions                           | PASS   | o_to_i_transitions=0                                                                                 |
| A2.2 zero sequences starting with I-*                   | PASS   | starts_with_I=0                                                                                      |
| A2.3 len(words)==len(ner_tags)==len(ner_tag_ids)        | PASS   | len_mismatch_records=0 (out of 14835)                                                                |
| A3.1 zero records with len(words) > 128                 | PASS   | len_gt_128=0                                                                                         |
| A3.2 zero records with len(words) < 10                  | PASS   | len_lt_10=0                                                                                          |
| A4 source distribution (hospital vs vimq)               | PASS   | total=14835; hospital=9717 (65.5%); vimq=5118 (34.5%)                                                |
| A5.1 intent label counts                                | PASS   | intent_counts={'method_diagnosis': 6882, 'severity': 3324, 'cause': 1042, 'treatment': 3587}; <50={} |
| A5.2 topic label counts                                 | PASS   | topic_unique=16; <50={}                                                                              |
| A6 schema completeness (required fields present)        | PASS   | missing_records=0; examples=[]                                                                       |
