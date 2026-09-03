"""
Shared text normalisation, used identically by Stage 1, Stage 2, and
before handing text to the Stage 3 NLP model — so all three stages see
the same normalised string, matching Section 3.1, step 1 of the design doc.
"""

import re
import string

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """Lowercase and strip punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalise_nospace(text: str) -> str:
    """normalise() plus remove all whitespace. Some bank/UPI reference
    strings run the merchant name together with no separators
    (e.g. 'UPI/AmazonIndia/Shopping'). Stage 1 checks both the spaced and
    the space-stripped form so it isn't fooled by this formatting quirk."""
    return normalise(text).replace(" ", "")


# --- spaCy-based lemmatiser for the NLP fallback stage -----------------
_nlp = None


def _get_spacy():
    global _nlp
    if _nlp is None:
        import spacy
        # disable components we don't need for lemmatisation -> faster
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    return _nlp


def lemmatise(text: str) -> str:
    """Lowercase, remove stopwords/punct, lemmatise. Used only for Stage 3
    (the NLP fallback classifier) so it generalises across surface forms
    ('booked', 'booking', 'book' -> 'book')."""
    nlp = _get_spacy()
    doc = nlp(normalise(text))
    tokens = [t.lemma_ for t in doc if not t.is_stop and not t.is_punct and t.lemma_.strip()]
    return " ".join(tokens)
