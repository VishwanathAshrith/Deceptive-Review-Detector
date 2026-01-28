from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np

app = Flask(__name__, static_folder='.', template_folder='.')

# Initialize with a pre-trained model or train on-the-fly
# For demo, we'll use a hybrid approach with feature engineering

class FakeReviewDetector:
    def __init__(self):
        # This would normally be loaded from a trained model file
        # For now, we'll use a more sophisticated scoring system
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english'
        )
        
    def extract_features(self, text):
        """Extract numerical features from review text"""
        text_lower = text.lower()
        features = {}
        
        # Length features
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
        
        # Punctuation features
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        # Pattern-based features
        features['has_url'] = 1 if re.search(r'http[s]?://', text_lower) else 0
        features['has_email'] = 1 if re.search(r'\S+@\S+', text_lower) else 0
        
        # Keyword features (weighted)
        spam_keywords = [
            'guarantee', 'best', 'amazing', 'free', 'buy', 'discount',
            'offer', 'limited', 'perfect', 'excellent', 'must'
        ]
        features['spam_keyword_count'] = sum(1 for kw in spam_keywords if kw in text_lower)
        
        # Sentiment extremity
        positive_words = ['love', 'great', 'amazing', 'excellent', 'perfect', 'best']
        negative_words = ['hate', 'terrible', 'awful', 'worst', 'horrible', 'bad']
        features['positive_count'] = sum(1 for pw in positive_words if pw in text_lower)
        features['negative_count'] = sum(1 for nw in negative_words if nw in text_lower)
        
        # Repetition
        words = text_lower.split()
        unique_words = set(words)
        features['repetition_ratio'] = len(unique_words) / max(len(words), 1)
        
        return features
    
    def predict(self, text):
        """Predict if review is fake using ensemble approach"""
        features = self.extract_features(text)
        
        # Scoring system
        score = 0
        
        # Length-based rules
        if features['word_count'] < 10:
            score += 2
        elif features['word_count'] > 200:
            score -= 1  # Very detailed reviews tend to be genuine
        
        # Caps and punctuation abuse
        if features['caps_ratio'] > 0.3:
            score += 3
        if features['exclamation_count'] > 2:
            score += 2
        
        # Spam keywords (threshold-based)
        if features['spam_keyword_count'] >= 3:
            score += 4
        elif features['spam_keyword_count'] >= 2:
            score += 2
        elif features['spam_keyword_count'] == 1:
            score += 1
        
        # Extreme sentiment
        if features['positive_count'] >= 4:
            score += 3
        if features['negative_count'] >= 4:
            score += 2
        
        # Low vocabulary diversity (repetitive = suspicious)
        if features['repetition_ratio'] < 0.5:
            score += 3
        
        # Contact info (spam indicator)
        if features['has_url'] or features['has_email']:
            score += 5
        
        # Balanced reviews (mention both positives and negatives)
        if features['positive_count'] > 0 and features['negative_count'] > 0:
            score -= 2  # Nuanced = likely genuine
        
        # Decision threshold
        return "Deceptive" if score >= 5 else "Genuine"


# Global detector instance
detector = FakeReviewDetector()

@app.route('/')
def index():
    return send_from_directory('.', 'sep10.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    prediction = detector.predict(review)
    return jsonify({"prediction": prediction})

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files["file"]
        df = pd.read_csv(file)

        total_reviews = len(df)
        print(f"=== Processing all {total_reviews} reviews ===")
        print(df.head())

        results = []
        genuine = deceptive = 0

        # Find the column with review text (try 'text_', 'text', 'review', or last column)
        if 'text_' in df.columns:
            review_column = 'text_'
        elif 'text' in df.columns:
            review_column = 'text'
        elif 'review' in df.columns:
            review_column = 'review'
        else:
            # Assume last column has the text
            review_column = df.columns[-1]
        
        print(f"Using column: '{review_column}' for review text")
        
        for idx, review in enumerate(df[review_column]):
            review_text = str(review)
            pred = detector.predict(review_text)
            
            # Truncate very long reviews for display
            display_text = review_text[:150] + "..." if len(review_text) > 150 else review_text
            results.append({"review": display_text, "prediction": pred})
            
            if pred == "Genuine":
                genuine += 1
            else:
                deceptive += 1
            
            # Progress updates every 500 reviews
            if (idx + 1) % 500 == 0:
                print(f"Processed {idx + 1}/{total_reviews} reviews... (G:{genuine}, D:{deceptive})")

        print(f"FINAL: {genuine} genuine, {deceptive} deceptive out of {total_reviews} reviews")
        
        detection_rate = (deceptive / total_reviews) * 100 if total_reviews > 0 else 0
        print(f"Detection rate: {detection_rate:.2f}%")

        return jsonify({
            "genuine_count": genuine,
            "deceptive_count": deceptive,
            "reviews": results,
            "total_processed": total_reviews
        })
    except Exception as e:
        print("UPLOAD ERROR:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, threaded=True)