import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample
import pickle

# ----------------- LOAD DATA -----------------
df = pd.read_csv("Kaggle Data.csv")  # replace with your CSV name
print("Columns:", df.columns)
print(df.head())

# Map labels to 0/1
# Adjust the mapping to your CSV labels
df['label_num'] = df['label'].map({'CG': 1, 'OR': 0})
print("Class counts before balancing:\n", df['label_num'].value_counts())

# ----------------- BALANCE DATA -----------------
genuine = df[df['label_num'] == 1]
deceptive = df[df['label_num'] == 0]

genuine_upsampled = resample(
    genuine,
    replace=True,
    n_samples=len(deceptive),
    random_state=42
)

df_balanced = pd.concat([deceptive, genuine_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print("Class counts after balancing:\n", df_balanced['label_num'].value_counts())

# ----------------- SPLIT TRAIN/TEST -----------------
X = df_balanced['text_']      # adjust column name if needed
y = df_balanced['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------- TF-IDF VECTORIZE -----------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------- TRAIN MODEL -----------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ----------------- EVALUATE -----------------
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ----------------- SAVE MODEL & VECTORIZER -----------------
pickle.dump(model, open("fake_review_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
print("Model and vectorizer saved!")

# ----------------- DEMO SAMPLE PREDICTIONS -----------------
samples = [
    "This product is amazing! Totally worth it.",
    "This is the worst product ever! Do not buy!",
    "Excellent quality, highly recommend it.",
    "Guaranteed best product, free gift included!",
    "Awful, completely broke in 1 day"
]

vec = vectorizer.transform(samples)
preds = model.predict(vec)
for s, p in zip(samples, preds):
    print(f'"{s}" ->', "Genuine" if p == 1 else "Deceptive")
