import os

import PyPDF2

from pe import TextPreprocessor


def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text.split()


def extract_text_from_file(file_path):
    if file_path.endswith('.pdf') or file_path.endswith('.PDF'):
        return extract_text_from_pdf(file_path)
    else:
        return ""

def load_data(folder_path):
    all_words = set()

    for subclass in os.listdir(folder_path):
        subclass_path = os.path.join(folder_path, subclass)
        if os.path.isdir(subclass_path):
            subclass_words = set()
            for file_name in os.listdir(subclass_path):
                file_path = os.path.join(subclass_path, file_name)
                text = extract_text_from_file(file_path)
                subclass_words.update(text)  # Add words from each file to the set
            if not all_words:
                all_words = subclass_words
            else:
                all_words &= subclass_words  # Intersect with words from previous subclass
    return list(all_words)
print(load_data("Train_Data"))
