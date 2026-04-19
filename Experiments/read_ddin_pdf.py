import PyPDF2
import glob
import re
import sys

out_path = r'c:\Users\amiku\Downloads\AI Research New Paradigm\Experiments\ddin_old_paper_clean.md'

pattern = r'c:\Users\amiku\Downloads\AI Research New Paradigm\Dev*.pdf'
matches = glob.glob(pattern)

if not matches:
    sys.exit("No PDF found.")

pdf_path = matches[0]
reader = PyPDF2.PdfReader(pdf_path)
total = len(reader.pages)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write("# DDIN Old Paper — Clean Extract (Notes Only — Verify Before Use)\n\n")
    f.write(f"Total Pages: {total}\n\n---\n\n")
    for i, page in enumerate(reader.pages):
        raw = page.extract_text() or ""
        # Collapse runs of whitespace/newlines to single space
        cleaned = re.sub(r'[ \t]+', ' ', raw)
        # Merge short lines that look like word-tokenized characters
        cleaned = re.sub(r'(?<=[a-zA-Zāīūṭḍṅñṇṃḥśṣ])\s(?=[a-zA-Zāīūṭḍṅñṇṃḥśṣ])', '', cleaned)
        # Normalize remaining multiple newlines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        f.write(f"## Page {i+1}\n\n")
        f.write(cleaned.strip() + "\n\n")

print(f"Done. Clean extract written to: ddin_old_paper_clean.md")
