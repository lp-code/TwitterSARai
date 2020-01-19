from pandas import Series
import pytest

from ..models.model_utils import preprocess_text, TextPreprocessor


txt1 = "Har du vært der? Ja, 33 ganger i fjor, og mange ganger før det. Tror du det?"
txt2 = "Bilen er funne, ikkje langt ifrå der han vart stjålen. Politiet søkjer etter spor."

@pytest.mark.parametrize(
    "rm_punctuation, rm_stopwords, stem, txt, expected",
    [
        (True, True, True, txt1, "gang fjor gang tror"),
        (False, True, True, txt1, "? , 33 gang fjor , gang . tror ?"),
        (True, False, True, txt1, "har du vært der ja gang i fjor og mang gang før det tror du det"),
        (True, True, False, txt1, "ganger fjor ganger tror"),
        (False, False, True, txt1, "har du vært der ? ja , 33 gang i fjor , og mang gang før det . tror du det ?"),
        (False, True, False, txt1, "? , 33 ganger fjor , ganger . tror ?"),
        (True, False, False, txt1, "har du vært der ja ganger i fjor og mange ganger før det tror du det"),
        (False, False, False, txt1, "har du vært der ? ja , 33 ganger i fjor , og mange ganger før det . tror du det ?"),
    ]
)
def test_preprocess_text(rm_punctuation, rm_stopwords, stem, txt, expected):
    assert preprocess_text(txt,
                           remove_punctuation=rm_punctuation,
                           remove_stopwords=rm_stopwords,
                           stem=stem) == expected


@pytest.mark.parametrize("inp_type", [Series, list])
def test_TextPreprocessor_Series(inp_type):
    # Arrange
    tp = TextPreprocessor()  # all settings at default
    X = inp_type([txt1, txt2])

    # Act
    Xt = tp.transform(X)

    # Assert
    assert (X == [txt1, txt2]
            if inp_type is list
            else (X == inp_type([txt1, txt2])).all())  # input values unchanged
    assert isinstance(Xt, inp_type)
    assert len(Xt) == 2
    assert Xt[0] == "gang fjor gang tror"
    assert Xt[1] == "bil funn langt ifrå stjål politi søkj spor"
