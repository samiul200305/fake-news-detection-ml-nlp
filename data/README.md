```markdown
# Dataset Setup

The datasets are not included directly in this repository because of their large file sizes.

This project uses separate datasets for English and Bangla fake news detection.

---

## 1. English Dataset

Dataset:

**Fake and Real News Dataset**

Source:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

After downloading and extracting the dataset, place these two files inside the `data/` folder:

```text
data/
├── Fake.csv
└── True.csv
```

### Labels

- `Fake.csv` → Fake News
- `True.csv` → Real News

The English model uses:

- TF-IDF Vectorization
- Word unigrams and bigrams
- Logistic Regression

---

## 2. Bangla Dataset

Dataset:

**BanFakeNews**

Source:

https://www.kaggle.com/datasets/cryptexcode/banfakenews

After downloading and extracting the dataset, create a folder named:

```text
data/bangla/
```

Place these files inside it:

```text
data/
└── bangla/
    ├── Authentic-48K.csv
    └── Fake-1K.csv
```

### Labels

- `Authentic-48K.csv` → Real News
- `Fake-1K.csv` → Fake News

The Bangla model uses:

- Word-level TF-IDF
- Character-level TF-IDF
- Logistic Regression
- Class weighting
- Validation-based decision threshold

---

## Final Dataset Structure

```text
data/
├── README.md
├── Fake.csv
├── True.csv
└── bangla/
    ├── Authentic-48K.csv
    └── Fake-1K.csv
```

---

## Train the English Model

```bash
python train_model.py
```

This generates:

```text
models/
├── fake_news_model.pkl
├── tfidf_vectorizer.pkl
└── metrics.json
```

---

## Train the Bangla Model

```bash
python train_bangla_model.py
```

This generates:

```text
models/
├── bangla_fake_news_model.pkl
├── bangla_tfidf_vectorizer.pkl
├── bangla_metrics.json
└── bangla_threshold.json
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Important Note

The datasets are used only for training and evaluation.

They are excluded from the GitHub repository because of their large file sizes.

The application outputs **Likely Real** or **Likely Fake** based on patterns learned from the training data.

These predictions should not be treated as independent proof that a real-world news story is factually true or false.
```