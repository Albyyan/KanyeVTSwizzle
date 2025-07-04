# 🎤 Kanye vs Taylor Lyric Classifier

This project is a web-based lyric classification tool that predicts whether a given lyric line was written by **Kanye West** or **Taylor Swift**. It uses two different machine learning models:

- **TF-IDF + SVM**
- **Word2Vec + SVM**

Built using Python, scikit-learn, Flask, and Gensim.

---

## 🧠 How It Works

### 1. TF-IDF + SVM
TF-IDF stands for **Term Frequency–Inverse Document Frequency**. It gives higher weight to rare but significant words that appear frequently in one artist's lyrics but rarely in the other's.

- Each lyric line is transformed into a **sparse vector** using TF-IDF.
- A **Support Vector Machine (SVM)** classifier is trained to distinguish between Kanye and Taylor lyrics based on these features.
- Particularly effective for **highlighting rare, discriminative words** (e.g., proper nouns, slang, stylistic phrases).
- This model has a slight bias towards Kanye, as Kanye uses a lot more distinct words in his lyrics. 

### 2. Word2Vec + SVM
Word2Vec transforms words into dense **300-dimensional vectors** based on their meanings and usage context, trained on the Google News dataset.

- Each lyric line is tokenized, and the vectors of all words are **averaged** to get a single vector.
- An SVM classifier is trained on these averaged vectors.
- Captures **semantic meaning** rather than raw word frequencies.
- Sometimes more subtle, but **less effective at highlighting rare or class-unique words** (e.g., the N-word).

## 📊 Results

Both models achieve around **80% accuracy** on the test set. Interestingly, both tend to perform **better at identifying Kanye lyrics** than Taylor Swift's.

This may be due to:
- **Stronger, more distinctive vocabulary** in Kanye's lyrics
- Use of **slang, unique phrasing, or uncommon words** that stand out more
- Taylor’s lyrics being more **linguistically generic**, which may overlap with common language across both classes

### 🔍 Model Comparison

| Metric            | TF-IDF + SVM | Word2Vec + SVM |
|------------------|--------------|----------------|
| Accuracy          | ~80%         | ~79%           |
| Strengths         | Highlights rare, discriminative words | Captures semantic similarity between words |
| Weaknesses        | Can't understand context or meaning | Averages can dilute strong signals |
| Bias              | Skews toward Kanye for rare or offensive words | More smoothed, less confident predictions |

---

## 🔧 Future Improvements

Some possible directions to improve performance and interpretability:

- 🔁 **Train Word2Vec on the lyrics dataset** rather than Google News for domain-specific embeddings
- 🧠 **Try transformer-based models** like BERT or RoBERTa for deeper contextual understanding
- 🗣️ Add **explanations for predictions**, e.g. which words contributed most (using SHAP or LIME)
- 📱 Make it mobile-friendly and deploy to a public URL
- 🎤 Add support for other artists (e.g. Drake, Lana Del Rey) and turn this into a full lyric-authorship classifier

