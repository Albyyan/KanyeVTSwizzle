from flask import Flask, request, render_template_string
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load trained classifier
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'lyric_classifier.joblib')
clf = joblib.load(MODEL_PATH)
label_map = {0: 'Taylor', 1: 'Kanye'}


def classify_line(text):
    """
    Returns percentage likelihoods for each artist given an input text.
    """
    proba = clf.predict_proba([text])[0]
    return {label_map[i]: round(p * 100, 2) for i, p in enumerate(proba)}

# HTML template
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
        input[type=text] { width: 100%; padding: 0.5rem; margin: 1rem 0; font-size: 1rem; }
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
    if request.method == 'POST':
        text = request.form.get('lyric', '').strip()
        if text:
            scores = classify_line(text)
    return render_template_string(HTML, scores=scores)

if __name__ == '__main__':
    app.run(debug=True)
