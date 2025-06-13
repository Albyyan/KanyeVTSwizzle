import csv
import os
import string 

# Get the directory where this script lives
BASE_DIR = os.path.dirname(__file__)

# Create a translation table that removes all punctuation
translator = str.maketrans('', '', string.punctuation)

def clean_and_write(input_fname, output_fname):
    """
    Reads lines from input_fname (inside BASE_DIR), strips blanks and [bracketed] lines,
    removes all punctuation, and writes each cleaned line to output_fname as CSV.
    """
    in_path  = os.path.join(BASE_DIR, input_fname)
    out_path = os.path.join(BASE_DIR, output_fname)

    with open(in_path, 'r', encoding='utf-8') as fin, \
         open(out_path, 'w', newline='', encoding='utf-8') as fout:

        writer = csv.writer(fout)

        for raw in fin:
            line = raw.strip()
            if not line:
                continue        # skip blank lines
            if '[' in line or ']' in line:
                continue        # skip bracketed metadata

            # remove punctuation
            line = line.translate(translator).strip()
            if not line:
                continue        # skip if line is empty after stripping punctuation

            writer.writerow([line])

if __name__ == "__main__":
    clean_and_write('TS_Lyrics.txt', 'TS_Lyrics.csv')
    clean_and_write('Ye_Lyrics.txt', 'Ye_Lyrics.csv')
    print("✅ Done! — CSVs created with punctuation removed.")
