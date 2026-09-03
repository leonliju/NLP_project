import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("pilot_dataset.csv")

# stratify by category+bucket combined so both are balanced in train/test
strata = df["category"] + "_" + df["bucket"]
train_df, test_df = train_test_split(
    df, test_size=0.3, random_state=42, stratify=strata
)

train_df.to_csv("train_split.csv", index=False)
test_df.to_csv("test_split.csv", index=False)
print("train:", train_df.shape, "test:", test_df.shape)
print(train_df["bucket"].value_counts())
print(test_df["bucket"].value_counts())
