from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import os
import pickle

app = Flask(__name__, static_folder='.', template_folder='.')

# Load trained model & vectorizer
model = pickle.load(open("fake_review_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def index():
    return send_from_directory('.', 'Frontend.html')

# SINGLE REVIEW
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review = data.get("review", "").strip()
        vec = vectorizer.transform([review])
        pred = model.predict(vec)[0]
        
        # SWAPPED: 1 = Deceptive, 0 = Genuine
        return jsonify({"prediction": "Deceptive" if pred == 1 else "Genuine"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# BATCH UPLOAD
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files.get("file")
        if file is None:
            return jsonify({"error": "No file uploaded"}), 400

        df = pd.read_csv(file)
        if df.empty:
            return jsonify({"error": "Empty CSV"}), 400

        # Find text column
        possible_cols = ['review', 'reviews', 'text_', 'Review', 'Text', 'comment']
        review_col = next((c for c in df.columns if c in possible_cols), df.columns[0])

        # Clean & predict
        df[review_col] = df[review_col].astype(str)
        vecs = vectorizer.transform(df[review_col])
        preds = model.predict(vecs)

        # SWAPPED: 1 = Deceptive, 0 = Genuine
        df['prediction'] = ['Deceptive' if p == 1 else 'Genuine' for p in preds]
        genuine = (df['prediction'] == 'Genuine').sum()
        deceptive = (df['prediction'] == 'Deceptive').sum()

        results = df[[review_col, 'prediction']].rename(columns={review_col: 'review'}).to_dict(orient='records')
        return jsonify({
            "genuine_count": int(genuine),
            "deceptive_count": int(deceptive),
            "reviews": results
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    # python apprun.py