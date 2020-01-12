import pytest

from ..data.data_utils import split_into_tags_and_doc

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
