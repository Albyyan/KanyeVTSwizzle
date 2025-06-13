import csv
import os
import string
import random

# Determine the folder where this script lives
BASE_DIR = os.path.dirname(__file__)

# Create a translation table that removes all punctuation
translator = str.maketrans('', '', string.punctuation)


def clean_and_write(input_fname, output_fname):
    """
    Reads lines from input_fname (inside BASE_DIR), strips blanks and [bracketed] lines,
    removes all punctuation, shuffles the order, and writes each cleaned line to output_fname as CSV.
    """
    in_path  = os.path.join(BASE_DIR, input_fname)
    out_path = os.path.join(BASE_DIR, output_fname)

    # Collect cleaned lines
    cleaned = []
    with open(in_path, 'r', encoding='utf-8') as fin:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue        # skip blank lines
            if '[' in line or ']' in line:
                continue        # skip bracketed metadata

            # remove punctuation
            line = line.translate(translator).strip()
            if not line:
                continue        # skip if empty after removing punctuation

            cleaned.append(line)

    # Shuffle the cleaned lines
    random.shuffle(cleaned)

    # Write out to CSV
    with open(out_path, 'w', newline='', encoding='utf-8') as fout:
        writer = csv.writer(fout)
        writer.writerow(['line'])  # header
        for line in cleaned:
            writer.writerow([line])


if __name__ == "__main__":
    clean_and_write('TS_Lyrics.txt', 'TS_Lyrics.csv')
    clean_and_write('Ye_Lyrics.txt', 'Ye_Lyrics.csv')
    print("✅ Done! — CSVs created with punctuation removed and lines shuffled.")
