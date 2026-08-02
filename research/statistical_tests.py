import pandas as pd
from scipy.stats import mannwhitneyu


df = pd.read_csv(
    "backend/data/final_corpus_clean.csv"
)


def lexical_diversity(text):
    words = text.lower().split()
    return len(set(words)) / max(1, len(words))


df["words"] = df["text"].str.split().apply(len)

df["sentences"] = df["text"].str.count(r"[.!?]")

df["lexical_diversity"] = (
    df["text"]
    .apply(lexical_diversity)
)


features = [
    "words",
    "sentences",
    "lexical_diversity"
]


for feature in features:
    student = df[df.label=="student"][feature]
    llm = df[df.label=="llm"][feature]

    stat, p = mannwhitneyu(
        student,
        llm,
        alternative="two-sided"
    )

    print("\nFEATURE:", feature)
    print("U statistic:", stat)
    print("p-value:", p)