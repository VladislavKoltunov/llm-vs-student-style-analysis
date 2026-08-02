from datasets import load_dataset
import pandas as pd


dataset = load_dataset(
    "knarasi1/student_and_llm_essays"
)


df = pd.DataFrame(
    dataset["train"]
)


df.to_csv(
    "data/raw_dataset.csv",
    index=False
)


print(df.head())