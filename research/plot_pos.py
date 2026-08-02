import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv(
    "data/pos_features.csv"
)


features = [
    "NOUN",
    "ADJ",
    "PRON",
    "AUX"
]


means = (
    df.groupby("label")[features]
    .mean()
)


means.T.plot(
    kind="bar"
)


plt.ylabel(
    "Percentage"
)


plt.title(
    "POS distribution differences"
)


plt.tight_layout()


plt.savefig(
    "results/pos_comparison.png",
    dpi=300
)