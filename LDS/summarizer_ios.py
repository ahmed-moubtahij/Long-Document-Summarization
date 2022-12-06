from pathlib import Path
import json

import jsonlines as jsonl
import deal

output_file_contract = deal.chain(
    deal.raises(ValueError),
    deal.has('stdout', 'write', 'read'),
    deal.pre(lambda _: _.out_path.exists()),
    deal.pre(lambda _: _.out_path.is_dir()),
)

read_dir_contract = deal.chain(
    deal.has('read'),
    deal.pre(lambda refs_path: refs_path.exists()),
    deal.pre(lambda refs_path: refs_path.is_dir())
)

read_summaries_contract = deal.chain(
    deal.raises(ValueError),
    deal.has('read', 'stdout'),
    deal.pre(lambda _: _.summaries_path.exists()),
    deal.pre(lambda _: _.summaries_path.is_file()),
    deal.pre(lambda _: str(_.summaries_path).endswith(".jsonl"))
)

@read_dir_contract
def read_references(refs_path: Path) -> list[str]:
    return [
        ref_file.read_text(encoding="utf-8")
        for ref_file in sorted(refs_path.iterdir())
    ]

@output_file_contract
def output_summaries(summary_units: list[dict[str, int | str]],
                     out_path: Path,
                     model_name: str,
                     post_read_sample=True
                    ) -> None:

    out_path /= f"{model_name}_summaries.jsonl"
    with out_path.open(mode='w', encoding='utf-8') as out_jsonl:
        jsonl.Writer(out_jsonl).write_all(summary_units)
    print(f"\nOutput summaries to {out_path}\n")

    if post_read_sample:
        print("\nReading sample:\n")
        print_sample(out_path, just_first=True)

@read_summaries_contract
def print_sample(summaries_path: Path, just_first=True) -> None:

    with jsonl.open(summaries_path, mode='r') as summarization_units:
        for summary_unit in summarization_units:
            chapter, summary = (summary_unit["CHAPTER"],
                                summary_unit["SUMMARY"])
            print(f"CHAPTER: {chapter}\n")
            double_spacing_summary = summary.replace('\n', "\n\n")
            print(f"SUMMARY:\n{double_spacing_summary}\n")
            print('-' * 100)
            if just_first:
                break

@output_file_contract
def output_scores(scores: dict[str, dict[str, float]],
                  out_path: Path,
                  model_name: str,
                  post_read_sample=True
                ) -> None:

    out_path /= f"{model_name}_scores.json"
    with out_path.open(mode='w', encoding='utf-8') as out_json:
        json.dump(scores, out_json)
    print(f"Output scores to {out_path}\n")

    if post_read_sample:
        with out_path.open(mode='r', encoding='utf-8') as json_file:
            scores_dict = json.load(json_file)
        print({r_variant: score['fmeasure']
               for r_variant, score in scores_dict.items()})
        print()
