import pandas as pd
from collections import Counter
import spacy


df = pd.read_csv("data/processed_corpus.csv")


nlp = spacy.load("en_core_web_sm")


def extract_function_words(text):

    doc = nlp(text)

    counter = Counter()
    total_words = 0


    for token in doc:

        if token.is_alpha:

            total_words += 1


            if token.pos_ in [
                "PRON",
                "AUX",
                "CCONJ",
                "SCONJ",
                "ADP",
                "DET"
            ]:

                counter[token.text.lower()] += 1


    return counter, total_words



for label in ["llm", "student"]:

    total_counter = Counter()
    total_words = 0


    texts = df[
        df["label"] == label
    ]["text"]


    for text in texts:

        counter, words = extract_function_words(text)

        total_counter.update(counter)

        total_words += words


    print("\n", label.upper())


    frequencies = []


    for word, count in total_counter.items():

        per_1000 = (
            count / total_words * 1000
        )

        frequencies.append(
            (
                word,
                per_1000
            )
        )


    frequencies.sort(
        key=lambda x: x[1],
        reverse=True
    )


    for word, freq in frequencies[:30]:

        print(
            word,
            round(freq, 2)
        )