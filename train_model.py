import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# --- Step 1: Load Your Dataset ---
# IMPORTANT: Replace 'your_dataset_name.csv' with the actual name of your file.
try:
    df = pd.read_csv('/Users/vishwanathashrith/Downloads/fake reviews dataset (1).csv')
    print("✅ Dataset loaded successfully.")
    print("Your column names are:", df.columns.tolist())
except FileNotFoundError:
    print("🛑 Error: Dataset file not found! Make sure the filename is correct and it's in the same folder as the script.")
    exit()

# --- Step 2: Prepare the Data ---
# IMPORTANT: Replace these with the actual column names from your CSV file.
review_column = 'text_'  # The name of the column that contains the review text
label_column = 'label'       # The name of the column that contains the 'Genuine'/'Deceptive' labels

# Check if columns exist
if review_column not in df.columns or label_column not in df.columns:
    print(f"🛑 Error: Make sure your CSV has the columns '{review_column}' and '{label_column}'.")
    exit()

# --- Step 3: Train the Model (No changes needed here) ---
vectorizer = TfidfVectorizer(max_features=5000) # Using top 5000 features
X = vectorizer.fit_transform(df[review_column])
y = df[label_column]

model = LogisticRegression()
model.fit(X, y)

# --- Step 4: Save the Model (No changes needed here) ---
joblib.dump(model, 'review_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("✅ Model and vectorizer have been created successfully from your dataset!")