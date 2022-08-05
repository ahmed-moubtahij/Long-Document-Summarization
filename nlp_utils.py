from spacy.lang.fr import French
import deal

@deal.pure
def trim(text: str) -> str:
    """Removes last sentence. Useful when the decoder generates it incompletely"""
    return '\n'.join(french_sentencizer(text)[:-1])

@deal.pure
def french_sentencizer(text: str) -> list[str]:
    # TODO: Verify whether `nlp` is cached, otherwise find a way to @cache it yourself
    nlp = French()
    nlp.add_pipe("sentencizer")

    return list(map(str, nlp(text).sents))
