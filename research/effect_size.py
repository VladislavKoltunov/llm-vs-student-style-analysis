import pandas as pd
import re
from cliffs_delta import cliffs_delta


# загружаем корпус
df = pd.read_csv("data/processed_corpus.csv")  


# функция подсчёта слов
def count_words(text):
    return len(text.split())


# функция подсчёта предложений
def count_sentences(text):
    sentences = re.split(r'[.!?]+', str(text))
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)


# lexical diversity (TTR)
def lexical_diversity(text):
    words = re.findall(
        r'\b[a-zA-Z]+\b',
        str(text).lower()
    )

    if len(words) == 0:
        return 0

    return len(set(words)) / len(words)


# создаём признаки

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


    delta, magnitude = cliffs_delta(
        llm,
        student
    )


    print("\nFEATURE:", feature)
    print("Cliff delta:", delta)
    print("Effect:", magnitude)