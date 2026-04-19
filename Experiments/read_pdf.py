import PyPDF2
import os

pdf_path = r'c:\Users\amiku\Downloads\AI Research New Paradigm\PhonoSemantics\phonosemantics.pdf'
out_path = r'c:\Users\amiku\Downloads\AI Research New Paradigm\PhonoSemantics\phonosemantics_extracted.md'

try:
    reader = PyPDF2.PdfReader(pdf_path)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("# PDF Extract\n\n")
        f.write(f"Total Pages: {len(reader.pages)}\n\n")
        for i, page in enumerate(reader.pages):
            f.write(f"## Page {i + 1}\n")
            f.write(page.extract_text() + "\n\n")
    print(f"Successfully extracted {len(reader.pages)} pages to {out_path}")
except Exception as e:
    print(f"Error: {e}")
