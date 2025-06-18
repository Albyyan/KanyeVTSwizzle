from flask import Flask, request, render_template_string
import joblib
import os
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import numpy as np
import nltk

nltk.download('punkt')

# Initialize Flask app
app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(__file__)
TFIDF_MODEL_PATH = os.path.join(BASE_DIR, 'svm_lyric_classifier.joblib')
W2V_MODEL_PATH = os.path.join(BASE_DIR, 'w2v_svm_classifier.joblib')
W2V_BIN_PATH = os.path.join(BASE_DIR, 'word2vec', 'GoogleNews-vectors-negative300.bin')

# Load models
tfidf_clf = joblib.load(TFIDF_MODEL_PATH)
w2v_model_path, w2v_clf = joblib.load(W2V_MODEL_PATH)
w2v = KeyedVectors.load_word2vec_format(W2V_BIN_PATH, binary=True)
label_map = {0: 'Taylor', 1: 'Kanye'}

def classify_tfidf(text):
    proba = tfidf_clf.predict_proba([text])[0]
    return {label_map[i]: round(p * 100, 2) for i, p in enumerate(proba)}

def classify_w2v(text):
    tokens = word_tokenize(text.lower(), preserve_line=True)
    vecs = [w2v[word] for word in tokens if word in w2v]
    if vecs:
        avg_vector = np.mean(vecs, axis=0).reshape(1, -1)
    else:
        avg_vector = np.zeros((1, w2v.vector_size))
    proba = w2v_clf.predict_proba(avg_vector)[0]
    return {label_map[i]: round(p * 100, 2) for i, p in enumerate(proba)}

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lyric Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f9f9f9; padding: 2rem; }
        .container { max-width: 600px; margin: auto; background: white; padding: 2rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        h1 { text-align: center; }
        input[type=text], select { width: 100%; padding: 0.5rem; margin: 1rem 0; font-size: 1rem; }
        button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
        .result { margin-top: 1rem; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Lyric Classifier</h1>
        <form method="POST">
            <label for="lyric">Enter a lyric line:</label>
            <input type="text" id="lyric" name="lyric" placeholder="e.g. I got the horses in the back" required>
            
            <label for="model">Choose model:</label>
            <select id="model" name="model">
                <option value="tfidf" {% if model == 'tfidf' %}selected{% endif %}>TF-IDF + SVM</option>
                <option value="w2v" {% if model == 'w2v' %}selected{% endif %}>Word2Vec + SVM</option>
            </select>
            
            <button type="submit">Classify</button>
        </form>
        {% if scores %}
        <div class="result">
            <p>Taylor likelihood: {{ scores['Taylor'] }}%</p>
            <p>Kanye likelihood:  {{ scores['Kanye'] }}%</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    scores = None
    model = 'tfidf'
    if request.method == 'POST':
        text = request.form.get('lyric', '').strip()
        model = request.form.get('model', 'tfidf')
        if text:
            if model == 'w2v':
                scores = classify_w2v(text)
            else:
                scores = classify_tfidf(text)
    return render_template_string(HTML, scores=scores, model=model)

if __name__ == '__main__':
    app.run(debug=True)
