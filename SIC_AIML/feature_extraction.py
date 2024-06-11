import os
import PyPDF2
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page_num in range(reader.numPages):
                page = reader.getPage(page_num)
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def process_text(text):
    try:
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        return tokens
    except Exception as e:
        print(f"Error processing text: {e}")
        return []

def extract_features(tokens):
    try:
        freq_dist = FreqDist(tokens)
        common_words = freq_dist.most_common(10)  # Adjust number as needed
        return common_words
    except Exception as e:
        print(f"Error extracting features: {e}")
        return []

def find_pdfs_in_directory(directory_path):
    pdf_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            print(f"Found file: {file}")  # Print each file found
            if file.lower().endswith('.pdf'):  # Consider both uppercase and lowercase extensions
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def process_pdfs_in_directory(directory_path):
    features = {}
    
    if not os.path.isdir(directory_path):
        print(f"Directory {directory_path} does not exist.")
        return features

    pdf_files = find_pdfs_in_directory(directory_path)
    print(f"PDF files found: {pdf_files}")
    if not pdf_files:
        print(f"No PDF files found in directory {directory_path}.")
        return features
    
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        text = extract_text_from_pdf(pdf_path)
        if not text:
            print(f"No text extracted from {pdf_path}")
            continue
        print(f"Extracted text from {pdf_path}:\n{text[:500]}...")  # Print first 500 chars for brevity
        
        tokens = process_text(text)
        if not tokens:
            print(f"No tokens extracted from {pdf_path}")
            continue
        print(f"Tokens from {pdf_path}:\n{tokens[:50]}...")  # Print first 50 tokens for brevity
        
        features[os.path.basename(pdf_path)] = extract_features(tokens)
        print(f"Features for {os.path.basename(pdf_path)}: {features[os.path.basename(pdf_path)]}")
    
    return features

if __name__ == "__main__":
    directory_path = 'Train_Data'  # Change this to your actual directory path
    features = process_pdfs_in_directory(directory_path)
    
    for filename, feature_list in features.items():
        print(f"Features for {filename}: {feature_list}")
