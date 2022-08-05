from pathlib import Path
import jsonlines as jsonl
import deal

@deal.has('read')
@deal.pre(lambda refs_path: refs_path.exists())
@deal.pre(lambda refs_path: refs_path.is_dir())
def read_references(refs_path: Path) -> list[str]:
    return [
        ref_file.read_text(encoding="utf-8")
        for ref_file in sorted(refs_path.iterdir())
    ]

output_summaries_contract = deal.chain(
    deal.has('stdout', 'write'),
    deal.pre(lambda _: _.out_path.exists()),
    deal.pre(lambda _: _.out_path.is_dir()),
    deal.post(lambda result: result.is_file())
)
def output_summaries(summary_units: list[dict[str, int | str]],
                     out_path: Path,
                     model_name: str) -> Path:

    out_path /= f"{model_name}_summaries.jsonl"
    with out_path.open(mode='w', encoding='utf-8') as out_jsonl:
        jsonl.Writer(out_jsonl).write_all(summary_units)
    print(f"Output summaries to {out_path}\n")

    return out_path


print_sample_contract = deal.chain(
    deal.raises(ValueError),
    deal.has('read', 'stdout'),
    deal.pre(lambda _: _.out_path.exists()),
    deal.pre(lambda _: _.out_path.is_file()),
    deal.pre(lambda _: str(_.out_path).endswith(".jsonl"))
)
@print_sample_contract
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
