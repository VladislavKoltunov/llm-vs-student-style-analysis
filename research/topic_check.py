import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv("data/processed_corpus.csv")


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=20
)


for label in ["llm", "student"]:

    texts = df[df["label"] == label]["text"]

    matrix = vectorizer.fit_transform(texts)

    terms = vectorizer.get_feature_names_out()

    print("\n", label.upper())

    print(list(terms))