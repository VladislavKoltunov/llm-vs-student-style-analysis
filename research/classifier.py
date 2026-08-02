import pandas as pd


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)



df = pd.read_csv(
    "data/features.csv"
)



features = [
    "words",
    "sentences",
    "lexical_diversity",
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "AUX",
    "ADP",
    "CCONJ",
    "SCONJ",
    "PART"
]


X = df[features]


y = df["label"]



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



model = LogisticRegression(
    max_iter=1000
)



model.fit(
    X_train,
    y_train
)



prediction = model.predict(
    X_test
)



print(
    "Accuracy:",
    accuracy_score(
        y_test,
        prediction
    )
)



print(
    "\nClassification report:"
)

print(
    classification_report(
        y_test,
        prediction
    )
)



print(
    "\nConfusion matrix:"
)

print(
    confusion_matrix(
        y_test,
        prediction
    )
)



importance = pd.DataFrame(
    {
        "feature":features,
        "weight":model.coef_[0]
    }
)


print(
    "\nFeature importance:"
)

print(
    importance.sort_values(
        "weight",
        ascending=False
    )
)