from pathlib import Path
import jsonlines as jsonl


summaries_fp = Path("data/output_summaries/textrank_french_semantic_summaries.jsonl").resolve()
with jsonl.open(summaries_fp, mode='r') as summaries_f:
    summaries = list(summaries_f)

print()
