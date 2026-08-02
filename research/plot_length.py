import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("data/processed_corpus.csv")


features = [
    "words",
    "sentences",
    "lexical_diversity"
]


def count_words(text):
    return len(str(text).split())


def count_sentences(text):
    import re
    return len(
        [
            s for s in re.split(
                r'[.!?]+',
                str(text)
            )
            if s.strip()
        ]
    )


def lexical_diversity(text):
    import re

    words = re.findall(
        r'\b[a-zA-Z]+\b',
        str(text).lower()
    )

    return len(set(words)) / len(words)



df["words"] = df["text"].apply(count_words)
df["sentences"] = df["text"].apply(count_sentences)
df["lexical_diversity"] = df["text"].apply(lexical_diversity)



for feature in features:

    df.boxplot(
        column=feature,
        by="label"
    )

    plt.title(feature)
    plt.suptitle("")

    plt.savefig(
        f"results/{feature}.png",
        dpi=300
    )

    plt.close()