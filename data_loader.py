from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
from pathlib import Path
from typing import Iterator, List, Any
import docx
from simplify_docx import simplify

# TODO: write the DataLoader class then import it in data_analysis.ipynb

def main(args) -> None:
    data_path = make_data_path(args.data_dir, args.data_fn)

    data = DataLoader(data_path)
    print(data)


def make_data_path(data_dir: str, data_fn: str) -> Path:
    data_path = Path(data_dir + data_fn).expanduser().resolve()
    assert data_path.exists()
    return data_path


def get_args() -> argparse.Namespace:
    arg_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    arg_parser.add_argument('-d', '--data-dir', type=str,
                            default="data/",
                            help="Path to directory containing the input data file")
    arg_parser.add_argument('-f', "--data-fn", type=str,
                            default="D5627-Dolan.docx",
                            help=("File name with the text to summarize"))
    return arg_parser.parse_args()


class DataLoader:

    def __init__(self, data_path: Path) -> None:
        assert data_path.exists()
        self.raw_doc: dict = simplify(docx.Document(data_path))
        self.doc: List[dict] = self.raw_doc["VALUE"][0]["VALUE"]

        self.paragraphs: Iterator[List[dict]] = self.__class__._init_paragraphs(self.doc)

    @staticmethod
    def _init_paragraphs(doc: List[dict]) -> Iterator[List[dict]]:
        _paragraphs: Iterator[Any] = map(lambda p: p["VALUE"], doc)
        # https://github.com/microsoft/Simplify-Docx/blob/
        # ce493b60e3e4308bde7399257426c0c68a6c699b/src/
        # simplify_docx/iterators/body.py#L60
        return (p for p in _paragraphs if p != "[w:sdt]")


if __name__ == "__main__":
    main(get_args())
