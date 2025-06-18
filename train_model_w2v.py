import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Paths
BASE_DIR = os.path.dirname(__file__)
data_dir = os.path.join(BASE_DIR, "Data")
w2v_path = os.path.join(BASE_DIR, "word2vec", "GoogleNews-vectors-negative300.bin")

def load_data(ts_path, ye_path):
    df_ts = pd.read_csv(ts_path)
    df_ye = pd.read_csv(ye_path)
    df_ts['label'] = 0   # Taylor → 0
    df_ye['label'] = 1   # Kanye  → 1
    return pd.concat([df_ts, df_ye], ignore_index=True)

def tokenize(texts):
    return [word_tokenize(line.lower(), preserve_line=True) for line in texts]

def vectorize(tokens_list, model, dim):
    vectors = []
    for tokens in tokens_list:
        vecs = [model[word] for word in tokens if word in model]
        if vecs:
            vectors.append(np.mean(vecs, axis=0))
        else:
            vectors.append(np.zeros(dim))
    return np.array(vectors)

if __name__ == "__main__":
    # 1. Load data
    df = load_data(
        os.path.join(data_dir, "TS_Lyrics.csv"),
        os.path.join(data_dir, "Ye_Lyrics.csv")
    )

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['line'], df['label'],
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )

    # 2. Tokenize
    X_train_tokens = tokenize(X_train_text)
    X_test_tokens = tokenize(X_test_text)

    # 3. Load pretrained Word2Vec
    print("🔍 Loading Word2Vec model (this may take a minute)...")
    w2v_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    embedding_dim = w2v_model.vector_size  # Should be 300
    print("✅ Word2Vec model loaded.")

    # 4. Convert lyrics to averaged embeddings
    X_train_vec = vectorize(X_train_tokens, w2v_model, embedding_dim)
    X_test_vec = vectorize(X_test_tokens, w2v_model, embedding_dim)

    # 5. Train SVM
    clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)
    clf.fit(X_train_vec, y_train)

    # 6. Evaluate
    preds = clf.predict(X_test_vec)
    print(f"SVM Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["Taylor", "Kanye"]))

    # 7. Save model
    joblib.dump((w2v_path, clf), os.path.join(BASE_DIR, "w2v_svm_classifier.joblib"))
    print("✅ Saved SVM model with Word2Vec to w2v_svm_classifier.joblib")
