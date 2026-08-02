import pandas as pd
import re
from cliffs_delta import cliffs_delta
from scipy.stats import mannwhitneyu


df = pd.read_csv("data/processed_corpus.csv")


def count_words(text):
    return len(text.split())


def count_sentences(text):
    sentences = re.split(r'[.!?]+', str(text))
    return len([s for s in sentences if s.strip()])


def lexical_diversity(text):
    words = re.findall(
        r'\b[a-zA-Z]+\b',
        str(text).lower()
    )

    if len(words) == 0:
        return 0

    return len(set(words)) / len(words)


df["words"] = df["text"].apply(count_words)
df["sentences"] = df["text"].apply(count_sentences)
df["lexical_diversity"] = df["text"].apply(lexical_diversity)


features = [
    "words",
    "sentences",
    "lexical_diversity"
]


for feature in features:

    llm = df[df["label"] == "llm"][feature]
    student = df[df["label"] == "student"][feature]


    stat, p = mannwhitneyu(
        llm,
        student,
        alternative="two-sided"
    )


    delta, effect = cliffs_delta(
        llm,
        student
    )


    print("\nFEATURE:", feature)

    print(
        "LLM mean:",
        round(llm.mean(), 3)
    )

    print(
        "Student mean:",
        round(student.mean(), 3)
    )

    print(
        "p-value:",
        p
    )

    print(
        "Cliff's delta:",
        round(delta, 3),
        effect
    )