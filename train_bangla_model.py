import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "bangla"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

AUTHENTIC_PATH = os.path.join(
    DATA_DIR,
    "Authentic-48K.csv"
)

FAKE_PATH = os.path.join(
    DATA_DIR,
    "Fake-1K.csv"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "bangla_fake_news_model.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "bangla_tfidf_vectorizer.pkl"
)

METRICS_PATH = os.path.join(
    MODEL_DIR,
    "bangla_metrics.json"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "bangla_threshold.json"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================
def load_dataset():

    if not os.path.exists(AUTHENTIC_PATH):
        raise FileNotFoundError(
            f"Missing file: {AUTHENTIC_PATH}"
        )

    if not os.path.exists(FAKE_PATH):
        raise FileNotFoundError(
            f"Missing file: {FAKE_PATH}"
        )

    print("\nLoading Bangla dataset...")

    real_df = pd.read_csv(
        AUTHENTIC_PATH,
        encoding="utf-8",
        encoding_errors="ignore"
    )

    fake_df = pd.read_csv(
        FAKE_PATH,
        encoding="utf-8",
        encoding_errors="ignore"
    )

    real_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat(
        [real_df, fake_df],
        ignore_index=True
    )

    return df


# =========================================================
# PREPARE TEXT
# =========================================================
def prepare_dataset(df):

    print("Preparing Bangla text...")

    headline_candidates = [
        "headline",
        "title",
        "Headline",
        "Title"
    ]

    content_candidates = [
        "content",
        "text",
        "article",
        "Content",
        "Text"
    ]

    headline_column = None
    content_column = None

    for column in headline_candidates:
        if column in df.columns:
            headline_column = column
            break

    for column in content_candidates:
        if column in df.columns:
            content_column = column
            break

    if headline_column is None:
        df["_headline"] = ""
        headline_column = "_headline"

    if content_column is None:
        raise ValueError(
            f"Content column not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    df[headline_column] = (
        df[headline_column]
        .fillna("")
        .astype(str)
    )

    df[content_column] = (
        df[content_column]
        .fillna("")
        .astype(str)
    )

    df["combined_text"] = (
        df[headline_column]
        + " "
        + df[content_column]
    )

    df["combined_text"] = (
        df["combined_text"]
        .str.replace(
            r"\s+",
            " ",
            regex=True
        )
        .str.strip()
    )

    df = df[
        df["combined_text"].str.len() > 20
    ]

    df = df.drop_duplicates(
        subset=["combined_text"]
    )

    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return df


# =========================================================
# FEATURE EXTRACTION
# =========================================================
def create_vectorizer():

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=60000,
        sublinear_tf=True
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=70000,
        sublinear_tf=True
    )

    return FeatureUnion([
        ("word_features", word_vectorizer),
        ("char_features", char_vectorizer),
    ])


# =========================================================
# FIND SAFER THRESHOLD
# =========================================================
def find_threshold(
    model,
    X_validation,
    y_validation
):

    probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    best_threshold = 0.50

    # We prioritize recognizing REAL articles.
    target_real_recall = 0.995

    candidates = np.linspace(
        0.01,
        0.50,
        100
    )

    valid_thresholds = []

    for threshold in candidates:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        real_recall = recall_score(
            y_validation,
            predictions,
            pos_label=1,
            zero_division=0
        )

        fake_recall = recall_score(
            y_validation,
            predictions,
            pos_label=0,
            zero_division=0
        )

        if real_recall >= target_real_recall:

            valid_thresholds.append(
                (
                    threshold,
                    real_recall,
                    fake_recall
                )
            )

    if valid_thresholds:

        # Highest threshold meeting our Real Recall target
        # preserves as much fake detection as possible.
        best_threshold = max(
            valid_thresholds,
            key=lambda item: item[0]
        )[0]

    print(
        f"\nSelected REAL threshold: "
        f"{best_threshold:.3f}"
    )

    return float(best_threshold)


# =========================================================
# TRAIN
# =========================================================
def train_model(df):

    X = df["combined_text"]
    y = df["label"]

    # First create untouched test set
    X_temp, X_test, y_temp, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )
    )

    # Validation set used only for threshold selection
    X_train, X_validation, y_train, y_validation = (
        train_test_split(
            X_temp,
            y_temp,
            test_size=0.20,
            random_state=42,
            stratify=y_temp
        )
    )

    print(
        f"\nTraining samples   : {len(X_train)}"
    )

    print(
        f"Validation samples : {len(X_validation)}"
    )

    print(
        f"Testing samples    : {len(X_test)}"
    )

    print(
        "\nCreating word + character TF-IDF features..."
    )

    vectorizer = create_vectorizer()

    X_train_vectorized = (
        vectorizer.fit_transform(
            X_train
        )
    )

    X_validation_vectorized = (
        vectorizer.transform(
            X_validation
        )
    )

    X_test_vectorized = (
        vectorizer.transform(
            X_test
        )
    )

    print(
        "Training Logistic Regression..."
    )

    # Milder weighting than class_weight='balanced'
    model = LogisticRegression(
        max_iter=2000,
        class_weight={
            0: 4.0,
            1: 1.0
        },
        random_state=42
    )

    model.fit(
        X_train_vectorized,
        y_train
    )

    threshold = find_threshold(
        model,
        X_validation_vectorized,
        y_validation
    )

    return (
        model,
        vectorizer,
        threshold,
        X_test_vectorized,
        y_test
    )


# =========================================================
# EVALUATE
# =========================================================
def evaluate_model(
    model,
    threshold,
    X_test,
    y_test
):

    real_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    predictions = (
        real_probabilities
        >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    real_recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0
    )

    fake_recall = recall_score(
        y_test,
        predictions,
        pos_label=0,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\n======================================")
    print("UPDATED BANGLA MODEL PERFORMANCE")
    print("======================================")

    print(
        f"Accuracy        : {accuracy:.4f}"
    )

    print(
        f"Macro Precision : {macro_precision:.4f}"
    )

    print(
        f"Macro Recall    : {macro_recall:.4f}"
    )

    print(
        f"Macro F1        : {macro_f1:.4f}"
    )

    print(
        f"REAL Recall     : {real_recall:.4f}"
    )

    print(
        f"FAKE Recall     : {fake_recall:.4f}"
    )

    print(
        f"Decision threshold: {threshold:.3f}"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Fake",
                "Real"
            ],
            zero_division=0
        )
    )

    print(
        "Confusion Matrix:"
    )

    print(matrix)

    metrics = {
        "accuracy": round(
            float(accuracy),
            4
        ),
        "precision": round(
            float(macro_precision),
            4
        ),
        "recall": round(
            float(macro_recall),
            4
        ),
        "f1_score": round(
            float(macro_f1),
            4
        ),
        "real_recall": round(
            float(real_recall),
            4
        ),
        "fake_recall": round(
            float(fake_recall),
            4
        ),
        "threshold": round(
            float(threshold),
            4
        ),
        "confusion_matrix": (
            matrix.tolist()
        ),
        "language": "Bangla",
        "labels": {
            "0": "Fake",
            "1": "Real"
        }
    }

    return metrics


# =========================================================
# SAVE
# =========================================================
def save_files(
    model,
    vectorizer,
    threshold,
    metrics
):

    joblib.dump(
        model,
        MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    with open(
        METRICS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4,
            ensure_ascii=False
        )

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "real_threshold": threshold
            },
            file,
            indent=4
        )

    print("\nSaved successfully.")

    print(
        f"Model     : {MODEL_PATH}"
    )

    print(
        f"Vectorizer: {VECTORIZER_PATH}"
    )

    print(
        f"Threshold : {THRESHOLD_PATH}"
    )

    print(
        f"Metrics   : {METRICS_PATH}"
    )


# =========================================================
# MAIN
# =========================================================
def main():

    print("\n======================================")
    print("IMPROVED BANGLA MODEL TRAINING")
    print("======================================")

    df = load_dataset()

    df = prepare_dataset(
        df
    )

    print(
        f"\nTotal usable articles: {len(df)}"
    )

    print(
        "\nClass distribution:"
    )

    print(
        df["label"].value_counts()
    )

    (
        model,
        vectorizer,
        threshold,
        X_test,
        y_test
    ) = train_model(
        df
    )

    metrics = evaluate_model(
        model,
        threshold,
        X_test,
        y_test
    )

    save_files(
        model,
        vectorizer,
        threshold,
        metrics
    )

    print("\n======================================")
    print("IMPROVED BANGLA TRAINING COMPLETED")
    print("======================================\n")


if __name__ == "__main__":
    main()