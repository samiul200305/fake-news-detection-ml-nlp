import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# =========================
# PATH SETTINGS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FAKE_DATA_PATH = os.path.join(DATA_DIR, "Fake.csv")
REAL_DATA_PATH = os.path.join(DATA_DIR, "True.csv")

MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")


# =========================
# CREATE REQUIRED FOLDERS
# =========================
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset():
    """
    Load Fake.csv and True.csv,
    assign labels and combine them.
    """

    if not os.path.exists(FAKE_DATA_PATH):
        raise FileNotFoundError(
            f"\nFake.csv পাওয়া যায়নি.\n"
            f"এই location-এ file রাখো:\n{FAKE_DATA_PATH}"
        )

    if not os.path.exists(REAL_DATA_PATH):
        raise FileNotFoundError(
            f"\nTrue.csv পাওয়া যায়নি.\n"
            f"এই location-এ file রাখো:\n{REAL_DATA_PATH}"
        )

    print("\nLoading dataset...")

    fake_df = pd.read_csv(FAKE_DATA_PATH)
    real_df = pd.read_csv(REAL_DATA_PATH)

    # 0 = Fake
    # 1 = Real
    fake_df["label"] = 0
    real_df["label"] = 1

    df = pd.concat([fake_df, real_df], ignore_index=True)

    return df


def prepare_text(df):
    """
    Combine title and article text.
    Remove empty rows and duplicates.
    """

    print("Preparing text...")

    if "title" not in df.columns:
        df["title"] = ""

    if "text" not in df.columns:
        raise ValueError(
            "Dataset-এর মধ্যে 'text' column পাওয়া যায়নি."
        )

    df["title"] = df["title"].fillna("")
    df["text"] = df["text"].fillna("")

    # Combine title + article
    df["content"] = (
        df["title"].astype(str)
        + " "
        + df["text"].astype(str)
    )

    # Remove unnecessary spaces
    df["content"] = (
        df["content"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Remove empty articles
    df = df[df["content"].str.len() > 20]

    # Remove duplicate articles
    df = df.drop_duplicates(subset=["content"])

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    return df


def train_model(df):
    """
    Split dataset, convert text using TF-IDF,
    train Logistic Regression model.
    """

    X = df["content"]
    y = df["label"]

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")

    print("\nCreating TF-IDF features...")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=2,
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("Training Logistic Regression model...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
    )

    model.fit(X_train_tfidf, y_train)

    return (
        model,
        vectorizer,
        X_test_tfidf,
        y_test,
    )


def evaluate_model(model, X_test_tfidf, y_test):
    """
    Evaluate trained model.
    """

    print("\nEvaluating model...")

    predictions = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    matrix = confusion_matrix(y_test, predictions)

    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Fake", "Real"],
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": matrix.tolist(),
        "labels": {
            "0": "Fake",
            "1": "Real",
        },
    }

    return metrics


def save_files(model, vectorizer, metrics):
    """
    Save model, vectorizer and metrics.
    """

    print("\nSaving trained model...")

    joblib.dump(model, MODEL_PATH)

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH,
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    print("\nSaved files:")

    print(f"Model      : {MODEL_PATH}")
    print(f"Vectorizer : {VECTORIZER_PATH}")
    print(f"Metrics    : {METRICS_PATH}")


def main():
    print("\n======================================")
    print("FAKE NEWS DETECTION - MODEL TRAINING")
    print("======================================")

    df = load_dataset()

    df = prepare_text(df)

    print(f"\nTotal usable news articles: {len(df)}")

    print("\nClass distribution:")
    print(df["label"].value_counts())

    model, vectorizer, X_test_tfidf, y_test = train_model(df)

    metrics = evaluate_model(
        model,
        X_test_tfidf,
        y_test,
    )

    save_files(
        model,
        vectorizer,
        metrics,
    )

    print("\n======================================")
    print("TRAINING COMPLETED SUCCESSFULLY ✅")
    print("======================================\n")


if __name__ == "__main__":
    main()