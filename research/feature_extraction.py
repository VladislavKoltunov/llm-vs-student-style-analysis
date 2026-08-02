import pandas as pd
import re
import spacy


# load language model
nlp = spacy.load("en_core_web_sm")


# load corpus
df = pd.read_csv(
    "data/processed_corpus.csv"
)


def count_words(text):
    return len(str(text).split())


def count_sentences(text):
    sentences = re.split(
        r'[.!?]+',
        str(text)
    )
    return len(
        [s for s in sentences if s.strip()]
    )


def lexical_diversity(text):

    words = re.findall(
        r'\b[a-zA-Z]+\b',
        str(text).lower()
    )

    if len(words) == 0:
        return 0

    return len(set(words)) / len(words)



def get_pos_features(text):

    doc = nlp(str(text))

    total = len(
        [
            token 
            for token in doc
            if not token.is_space
        ]
    )

    pos_counts = {}

    for token in doc:

        pos = token.pos_

        pos_counts[pos] = (
            pos_counts.get(pos,0)+1
        )


    features = {}

    target_pos = [
        "NOUN",
        "VERB",
        "ADJ",
        "ADV",
        "PRON",
        "AUX",
        "ADP",
        "CCONJ",
        "SCONJ",
        "PART"
    ]


    for pos in target_pos:

        features[pos] = round(
            pos_counts.get(pos,0)
            /
            total
            *
            100,
            3
        )


    return features



# basic features

df["words"] = df["text"].apply(
    count_words
)

df["sentences"] = df["text"].apply(
    count_sentences
)

df["lexical_diversity"] = df["text"].apply(
    lexical_diversity
)



# POS features

pos_features = df["text"].apply(
    get_pos_features
)


pos_df = pd.DataFrame(
    list(pos_features)
)


df = pd.concat(
    [
        df,
        pos_df
    ],
    axis=1
)



# save

df.to_csv(
    "data/features.csv",
    index=False
)


print(df.head())

print(
    "\nSaved:",
    "data/features.csv"
)