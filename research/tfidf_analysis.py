import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("data/processed_corpus.csv")


# разделяем тексты
llm_texts = df[df["label"] == "llm"]["text"]

student_texts = df[df["label"] == "student"]["text"]


def get_top_terms(texts, n=20):

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )


    matrix = vectorizer.fit_transform(texts)


    scores = matrix.mean(axis=0)


    terms = vectorizer.get_feature_names_out()


    result = []


    for i, score in enumerate(scores.tolist()[0]):

        result.append(
            (
                terms[i],
                score
            )
        )


    result.sort(
        key=lambda x: x[1],
        reverse=True
    )


    return result[:n]



print("\nTOP LLM TERMS")

for term, score in get_top_terms(llm_texts):

    print(
        term,
        round(score, 4)
    )



print("\nTOP STUDENT TERMS")

for term, score in get_top_terms(student_texts):

    print(
        term,
        round(score, 4)
    )