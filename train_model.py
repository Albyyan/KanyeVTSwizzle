import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 1. Load & label
def load_data(ts_path, ye_path):
    df_ts = pd.read_csv(ts_path)                # has column 'line'
    df_ye = pd.read_csv(ye_path)
    df_ts['label'] = 0                          # Taylor → 0
    df_ye['label'] = 1                          # Kanye  → 1
    return pd.concat([df_ts, df_ye], ignore_index=True)

if __name__ == "__main__":
    # adjust paths if needed
    data_dir = "Data"
    df = load_data(os.path.join(data_dir, "TS_Lyrics.csv"),
                   os.path.join(data_dir, "Ye_Lyrics.csv"))

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['line'], df['label'],
        test_size=0.2,
        stratify=df['label'],
    )

    # 3. Build a pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_df=0.9,
            min_df=5
        )),
        ("clf", LogisticRegression(
            solver="liblinear",
            class_weight="balanced",
            random_state=42
        )),
    ])

    # 4. Train
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    preds = pipeline.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(
        y_test, preds,
        target_names=["Taylor", "Kanye"]
    ))

    # 6. Save
    joblib.dump(pipeline, "lyric_classifier.joblib")
    print("✅ Model saved to lyric_classifier.joblib")
