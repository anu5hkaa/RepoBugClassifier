
import emoji
import spacy

SPACY_MODEL = "en_core_web_sm"  # <-- confirm this matches what you used during training

try:
    nlp = spacy.load(SPACY_MODEL)
except OSError:
    raise OSError(
        f"spaCy model '{SPACY_MODEL}' is not installed. "
        f"Run: python -m spacy download {SPACY_MODEL}"
    )


def preprocess_text(text):
    if text is None:
        text = ""

    text = emoji.replace_emoji(text, replace="")
    doc = nlp(text)

    tokens = []

    for token in doc:

        if token.is_stop:
            continue

        if token.is_punct:
            continue

        if token.is_space:
            continue

        tokens.append(token.lemma_.lower())

    return " ".join(tokens)