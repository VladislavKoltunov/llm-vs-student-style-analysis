import pandas as pd


df = pd.read_csv(
    "backend/data/final_corpus_clean.csv"
)


print("CORPUS SIZE")
print(df["label"].value_counts())


def statistics(group):
    texts = group["text"]

    return pd.Series({
        "documents": len(texts),
        "avg_characters": texts.str.len().mean(),
        "avg_words": texts.str.split().apply(len).mean(),
        "avg_sentences": texts.str.count(r"[.!?]").mean(),
        "lexical_diversity": (
            texts.apply(
                lambda x: len(set(x.lower().split()))
                /
                max(1, len(x.split()))
            )
        ).mean()
    })


result = df.groupby("label").apply(statistics)

print("\nSTATISTICS")
print(result)


result.to_csv(
    "research/corpus_statistics.csv"
)