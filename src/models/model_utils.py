import string

from nltk import word_tokenize
from nltk.stem import SnowballStemmer


# The following list of stopwords was retrieved from nltk.corpus.stopwords;
# it needs to be downloaded separately from the package install, that's why
# it is copied in here. Reformat as string to avoid the code formatter's
# line splitting.
_stopwords_no = (
    "alle at av bare begge ble blei bli blir blitt både båe da de deg dei "
    "deim deira deires dem den denne der dere deres det dette di din disse "
    "ditt du dykk dykkar då eg ein eit eitt eller elles en enn er et ett "
    "etter for fordi fra før ha hadde han hans har hennar henne hennes her "
    "hjå ho hoe honom hoss hossen hun hva hvem hver hvilke hvilken hvis hvor "
    "hvordan hvorfor i ikke ikkje ikkje ingen ingi inkje inn inni ja jeg kan "
    "kom korleis korso kun kunne kva kvar kvarhelst kven kvi kvifor man mange "
    "me med medan meg meget mellom men mi min mine mitt mot mykje ned no noe "
    "noen noka noko nokon nokor nokre nå når og også om opp oss over på samme "
    "seg selv si si sia sidan siden sin sine sitt sjøl skal skulle slik so "
    "som som somme somt så sånn til um upp ut uten var vart varte ved vere "
    "verte vi vil ville vore vors vort vår være være vært å"
).split()


def preprocess_text(
        text: str,
        remove_punctuation: bool = True,
        remove_stopwords: bool = True,
        stem: bool = True,
) -> str:
    """Text processing into suitable input for the vectorizer (or similar)."""
    _text = text.lower()

    if remove_punctuation:
        punctuation = set(string.punctuation + string.digits)
        _text = "".join([ch for ch in _text if ch not in punctuation])

    tokens = word_tokenize(_text)  # Returns a list of words.

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _stopwords_no]

    if stem:
        norwegian_stemmer = SnowballStemmer('norwegian')
        tokens = map(norwegian_stemmer.stem, tokens)

    # Return a string, this is the input for CountVectorizer etc.
    return " ".join(tokens)


class TextPreprocessor:
    def __init__(self, remove_punctuation=True, remove_stopwords=True, stem=True):
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stem = stem

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for i, s in X_copy.items():
            X_copy[i] = preprocess_text(
                s,
                self.remove_punctuation,
                self.remove_stopwords,
                self.stem,
            )

        return X_copy, y