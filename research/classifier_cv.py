import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_val_predict
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Load features
df = pd.read_csv("data/features.csv")

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

# Logistic Regression model
model = LogisticRegression(max_iter=1000)

# 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("5-fold Cross Validation Accuracy\n")

for i, score in enumerate(scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print("\nMean accuracy:", round(scores.mean(), 4))
print("Standard deviation:", round(scores.std(), 4))

# Predictions from cross-validation
pred = cross_val_predict(
    model,
    X,
    y,
    cv=cv
)

print("\nClassification Report\n")
print(classification_report(y, pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y, pred))

# Train once on the full dataset to inspect coefficients
model.fit(X, y)

importance = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

importance["Absolute"] = importance["Coefficient"].abs()

importance = importance.sort_values(
    "Absolute",
    ascending=False
)

print("\nFeature Coefficients\n")
print(importance)