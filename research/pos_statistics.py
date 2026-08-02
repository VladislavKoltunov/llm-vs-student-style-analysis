import pandas as pd
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta


df = pd.read_csv("data/pos_features.csv")


features = [
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
]


for feature in features:

    llm = df[df["label"] == "llm"][feature]

    student = df[df["label"] == "student"][feature]


    # Mann–Whitney U

    stat, p = mannwhitneyu(
        llm,
        student,
        alternative="two-sided"
    )


    # Cliff's delta

    delta, effect = cliffs_delta(
        llm,
        student
    )


    print("\nPOS:", feature)

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
        "Cliff delta:",
        round(delta, 3),
        effect
    )