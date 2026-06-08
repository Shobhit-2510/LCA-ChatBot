"""Build the QA test set (Evaluation Part 1).

Feed each book section to an LLM ("write 10 Q&A pairs from this text"),
aim for 500-1,000 pairs, label Positive/Negative, tag by book chapter
(Goal & Scope, LCI, LCIA, ...). Cohen's Kappa for annotator agreement.

TODO: implement generate_pairs(), label/verify helpers
"""
