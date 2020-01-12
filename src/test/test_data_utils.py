import pytest

from ..data.data_utils import *

@pytest.mark.parametrize(
    "tweet, expected",
    [
        (
                "@Hordalandpoliti Det er stadfesta at personen har vore åleine ombord. Redningsaksjonen vert avslutta.",
                ("", "Det er stadfesta at personen har vore åleine ombord. Redningsaksjonen vert avslutta.")
        ),
        (
            "Fv 562 ved #Åsebø på #Askøy kl. 1920. Ein stor stein, tippa til rundt 100 kg, hadde rase ned i vegen.",
            ("Åsebø|Askøy", "Fv 562 ved  på  kl. 1920. Ein stor stein, tippa til rundt 100 kg, hadde rase ned i vegen.")
        ),
        (
            "Politiet på stedet. Bilen er fjernet. Trafikken flyter som normalt.",
            ("", "Politiet på stedet. Bilen er fjernet. Trafikken flyter som normalt.")
        ),
        (
            "#Nordfjord,#Stryn, #Langeset, #RV15 Meldt om trafikkulykke, personbil som har kjørt inn i vogntog.",
            ("Nordfjord|Stryn|Langeset|RV15", "Meldt om trafikkulykke, personbil som har kjørt inn i vogntog.")
        )
    ]
)
def test_split_into_tags_and_doc(tweet, expected):
    assert split_into_tags_and_doc(tweet) == expected


txt1 = "Har du vært der? Ja, 33 ganger i fjor, og mange ganger før det. Tror du det?"

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