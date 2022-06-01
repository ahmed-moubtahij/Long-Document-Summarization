from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import argparse
from pathlib import Path
import docx
from simplify_docx import simplify


def main(args) -> None:
    data_path = make_data_path(args.data_dir, args.data_fn)

    my_doc_as_json: dict = simplify(docx.Document(data_path))
    print(my_doc_as_json)

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


if __name__ == "__main__":
    main(get_args())
