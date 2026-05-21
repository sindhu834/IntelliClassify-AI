import pandas as pd
import re
import string
import nltk
import pickle

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# =========================
# DOWNLOAD NLTK DATA
# =========================

nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv("train.csv")

print("\nDataset Loaded Successfully!\n")

print(df.head())

print("\nDataset Shape:")
print(df.shape)

# =========================
# REMOVE EMPTY VALUES
# =========================

df = df.dropna()

# =========================
# LABEL MAPPING
# =========================

label_map = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

# Convert labels to integer
df["Class Index"] = df["Class Index"].astype(int)

# Create category column
df["Category"] = df["Class Index"].map(label_map)

# Remove invalid rows
df = df.dropna(subset=["Category"])

print("\nUnique Categories:")
print(df["Category"].unique())

# =========================
# COMBINE TITLE + DESCRIPTION
# =========================

df["Text"] = (
    df["Title"].astype(str)
    + " "
    + df["Description"].astype(str)
)

# =========================
# NLP PREPROCESSING
# =========================

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\\S+", "", text)

    # Remove numbers
    text = re.sub(r"\\d+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Split words
    words = text.split()

    # Remove stopwords + lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply cleaning
df["Clean_Text"] = df["Text"].apply(clean_text)

# =========================
# FEATURES & LABELS
# =========================

X = df["Clean_Text"]

y = df["Category"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# TF-IDF VECTORIZATION
# =========================

vectorizer = TfidfVectorizer(
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# =========================
# MODEL TRAINING
# =========================

model = LinearSVC()

model.fit(X_train_vec, y_train)

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X_test_vec)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n======================")
print("Model Accuracy:", round(accuracy * 100, 2), "%")
print("======================")

# =========================
# SAVE MODEL
# =========================

pickle.dump(
    model,
    open("model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl", "wb")
)

print("\nModel Saved Successfully!")

# =========================
# SAMPLE TEST
# =========================

sample = [
    "Google launched a new AI technology."
]

cleaned = [
    clean_text(text)
    for text in sample
]

sample_vec = vectorizer.transform(cleaned)

result = model.predict(sample_vec)

print("\nSample Prediction:")
print(result[0])