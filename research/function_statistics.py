import pandas as pd
import spacy
from scipy.stats import mannwhitneyu
from cliffs_delta import cliffs_delta
from collections import Counter


df = pd.read_csv("data/processed_corpus.csv")


nlp = spacy.load("en_core_web_sm")


function_words = [
    "the",
    "and",
    "of",
    "in",
    "a",
    "can",
    "to",
    "for",
    "that",
    "it",
    "they",
    "you",
    "we",
    "our",
    "because",
    "if",
    "would",
    "will"
]


def word_frequency(text, word):

    doc = nlp(text)

    words = [
        token.text.lower()
        for token in doc
        if token.is_alpha
    ]

    if len(words) == 0:
        return 0

    return (
        words.count(word)
        /
        len(words)
        *
        1000
    )



for word in function_words:

    llm = df[df["label"]=="llm"]["text"].apply(
        lambda x: word_frequency(x, word)
    )

    student = df[df["label"]=="student"]["text"].apply(
        lambda x: word_frequency(x, word)
    )


    stat, p = mannwhitneyu(
        llm,
        student,
        alternative="two-sided"
    )


    delta, effect = cliffs_delta(
        llm,
        student
    )


    print("\nWORD:", word)

    print(
        "LLM:",
        round(llm.mean(),3),
        "Student:",
        round(student.mean(),3)
    )

    print(
        "p:",
        p
    )

    print(
        "delta:",
        round(delta,3),
        effect
    )