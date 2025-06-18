import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Use absolute or script-relative path for robustness
BASE_DIR = os.path.dirname(__file__)
data_dir = os.path.join(BASE_DIR, "Data")

def load_data(ts_path, ye_path):
    df_ts = pd.read_csv(ts_path)
    df_ye = pd.read_csv(ye_path)
    df_ts['label'] = 0  # Taylor → 0
    df_ye['label'] = 1  # Kanye  → 1
    return pd.concat([df_ts, df_ye], ignore_index=True)

if __name__ == "__main__":
    # 1. Load and split dataset
    df = load_data(
        os.path.join(data_dir, "TS_Lyrics.csv"),
        os.path.join(data_dir, "Ye_Lyrics.csv")
    )
    X_train, X_test, y_train, y_test = train_test_split(
        df['line'], df['label'],
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )

    # 2. Define pipeline (TF-IDF → SVM)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.9, min_df=5)),
        ("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ])

    # 3. Train model
    pipeline.fit(X_train, y_train)

    # 4. Evaluate
    preds = pipeline.predict(X_test)
    print(f"SVM Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, target_names=["Taylor", "Kanye"]))

    # 5. Save model
    model_path = os.path.join(BASE_DIR, "svm_lyric_classifier.joblib")
    joblib.dump(pipeline, model_path)
    print(f"✅ Saved SVM model to {model_path}")
