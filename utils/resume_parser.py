import fitz  # PyMuPDF
import docx2txt
import os

def extract_text(file):
    if file.name.endswith(".pdf"):
        return extract_pdf_text(file)
    elif file.name.endswith(".docx"):
        return docx2txt.process(file)
    return ""

def extract_pdf_text(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text