import pandas as pd
import os
from sklearn.model_selection import train_test_split
import re

# Paths
RAW_DATA_PATH = "data/raw/safety_dataset.csv"
PROCESSED_DIR = "data/processed"
TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.csv")
TEST_PATH = os.path.join(PROCESSED_DIR, "test.csv")

# Step 1: Load dataset
def load_data():
    df = pd.read_csv(RAW_DATA_PATH)
    return df

# Step 2: Clean text
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove special chars
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text

# Step 3: Encode labels
def encode_labels(df):
    df["label"] = df["label"].str.lower()   # normalize
    label_mapping = {"safe": 0, "unsafe": 1}
    df["label"] = df["label"].map(label_mapping)
    return df


# Step 4: Split dataset
def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    return train_df, test_df

# Step 5: Save processed data
def save_data(train_df, test_df):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)
    print(f" Data saved: {TRAIN_PATH}, {TEST_PATH}")

def main():
    print("🔹 Loading data...")
    df = load_data()

    print("🔹 Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    print("🔹 Encoding labels...")
    df = encode_labels(df)

    print("🔹 Splitting data...")
    train_df, test_df = split_data(df)

    print("🔹 Saving processed files...")
    save_data(train_df, test_df)

if __name__ == "__main__":
    main()
