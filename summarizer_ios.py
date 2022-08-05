from pathlib import Path
import jsonlines as jsonl
import deal

@deal.pure
@deal.pre(lambda refs_path: refs_path.exists())
def read_references(refs_path: Path) -> list[str]:
    return [
        ref_file.read_text(encoding="utf-8")
        for ref_file in sorted(refs_path.iterdir())
    ]

@deal.safe
@deal.has('stdout', 'write')
@deal.pre(lambda _: _.out_path.exists())
def output_summaries(summary_units: list[dict[str, int | str]],
                     out_path: Path,
                     model_name: str) -> Path:

    out_path /= f"{model_name}_summaries.jsonl"
    with open(out_path, 'w', encoding='utf-8') as out_jsonl:
        jsonl.Writer(out_jsonl).write_all(summary_units)
    print(f"Output summaries to {out_path}\n")

    return out_path

@deal.raises(ValueError)
@deal.has('read', 'stdout')
@deal.pre(lambda _: _.out_path.exists())
def print_sample(out_path: Path, just_one=True) -> None:

    with jsonl.open(out_path, mode='r') as summarization_units:
        for summary_unit in summarization_units:
            chapter, summary = (summary_unit["CHAPTER"],
                                summary_unit["SUMMARY"])
            print(f"CHAPTER: {chapter}\n")
            double_spacing_summary = summary.replace('\n', "\n\n")
            print(f"SUMMARY:\n{double_spacing_summary}\n")
            print('-' * 100)
            if just_one:
                break
