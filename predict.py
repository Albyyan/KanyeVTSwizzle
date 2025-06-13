import tkinter as tk
from tkinter import messagebox
import joblib
import os

# Load the trained pipeline
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


# Build GUI
root = tk.Tk()
root.title("Lyric Classifier")
root.resizable(False, False)

# Instruction label
tk.Label(root, text="Enter a lyric line:").pack(padx=10, pady=(10, 0))

# Text entry
entry = tk.Entry(root, width=60)
entry.pack(padx=10, pady=5)
entry.focus()

# Classification callback
def on_classify():
    text = entry.get().strip()
    if not text:
        messagebox.showwarning("Input needed", "Please enter a lyric line to classify.")
        return
    scores = classify_line(text)
    msg = f"Taylor likelihood: {scores['Taylor']}%\nKanye likelihood: {scores['Kanye']}%"
    messagebox.showinfo("Classification Result", msg)

# Classify button
btn = tk.Button(root, text="Classify", command=on_classify)
btn.pack(pady=(0, 10))

# Run the GUI loop
root.mainloop()
