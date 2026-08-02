import pandas as pd
import spacy
from collections import Counter


df = pd.read_csv("data/processed_corpus.csv")


nlp = spacy.load("en_core_web_sm")


def get_pos_features(text):

    doc = nlp(text)

    counter = Counter()
    total = 0

    for token in doc:

        if token.is_alpha:
            counter[token.pos_] += 1
            total += 1


    result = {}

    for pos in [
        "NOUN",
        "VERB",
        "ADJ",
        "ADV",
        "PRON",
        "DET",
        "AUX",
        "ADP",
        "CCONJ",
        "SCONJ",
        "PART"
    ]:

        result[pos] = (
            counter[pos] / total * 100
            if total > 0
            else 0
        )


    return result



# создаём признаки для каждого текста

pos_features = df["text"].apply(get_pos_features)


pos_df = pd.DataFrame(
    list(pos_features)
)


# добавляем метки

pos_df["label"] = df["label"]


# сохраняем

pos_df.to_csv(
    "data/pos_features.csv",
    index=False
)


print(pos_df.head())