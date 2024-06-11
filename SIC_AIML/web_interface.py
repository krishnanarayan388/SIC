import os
import re
import PyPDF2
import docx
import pytesseract
from PIL import Image
import io
import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xgboost import XGBClassifier
import gradio as gr
from pe import TextPreprocessor

class DocumentClassifier:
    def __init__(self, model_type, model_path, tokenizer_or_vectorizer_path, label_to_index):
        self.model_type = model_type
        self.label_to_index = label_to_index
        self.index_to_label = {v: k for k, v in label_to_index.items()}
        
        if model_type in ['CNN', 'RNN', 'LSTM']:
            self.model = tf.keras.models.load_model(model_path)
            self.tokenizer = joblib.load(tokenizer_or_vectorizer_path)
            self.max_seq_length = self.model.input_shape[1]
        else:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(tokenizer_or_vectorizer_path)

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_text_from_docx(self, docx_path):
        text = ""
        doc = docx.Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                image_part = rel.target_part
                image_data = image_part.blob
                image = Image.open(io.BytesIO(image_data))
                image_text = pytesseract.image_to_string(image)
                text += image_text
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess(text)
        return processed_text

    def classify_document(self, file_path):
        # Print all labels at the beginning
        print(f"All labels (label_to_index): {self.label_to_index}")
        
        if file_path.endswith('.pdf') or file_path.endswith('.PDF'):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            text = self.extract_text_from_docx(file_path)
        else:
            print("Unsupported file type")
            return

        if not text:
            print("Failed to extract text from the document")
            return

        if self.model_type in ['CNN', 'RNN', 'LSTM']:
            # Tokenize text
            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_seq_length)

            # Predict
            predictions = self.model.predict(padded_sequence)
            predicted_class_index = np.argmax(predictions, axis=1)[0]
            print("Donme")
            # Debugging statements
            print(f"Predictions: {predictions}")
            print(f"Predicted class index: {predicted_class_index}")
            print(f"Index to label mapping: {self.index_to_label}")

            # Ensure the predicted class index is in the index_to_label dictionary
            if predicted_class_index in self.index_to_label:
                predicted_class = self.index_to_label[predicted_class_index]
            else:
                print("Error: Predicted class index not found in index_to_label mapping")
                return "Error: Classification failed"

            return predicted_class
        else:
            features = self.vectorizer.transform([text])
            predicted_class_indices = self.model.predict(features)

            # Convert predicted class indices to class labels
            predicted_classes = [self.index_to_label[index] for index in predicted_class_indices]

            return predicted_classes[0]

def extract_labels_from_folder(folder_path):
    label_to_index = {}
    index = 0
    for subclass in os.listdir(folder_path):
        subclass_path = os.path.join(folder_path, subclass)
        if os.path.isdir(subclass_path):
            label_to_index[subclass] = index
            index += 1
    return label_to_index

def run(file_name, model_choice):
    model_paths = {
        "Convolutional Neural Network (CNN)": "text_classifier_model.h5",
        "Recurrent Neural Network (RNN)": "text_classifier_model_RNNModel.h5",
        "Long Short-Term Memory (LSTM)": "text_classifier_model_lstm.h5",
        "Random Forest": "random_forest_model.joblib",
        "Naive Bayes": "naive_bayes_model.joblib",
        "XGBoost": "xgb_model.joblib"
    }
    
    tokenizer_vectorizer_paths = {
        "Convolutional Neural Network (CNN)": "tokenizer.joblib",
        "Recurrent Neural Network (RNN)": "tokenizer.joblib",
        "Long Short-Term Memory (LSTM)": "tokenizer.joblib",
        "Random Forest": "vectorizer.joblib",
        "Naive Bayes": "vectorizer.joblib",
        "XGBoost": "vectorizer.joblib"
    }

    TRAIN_DATA_FOLDER = "Train_Data"
    
    label_to_index = extract_labels_from_folder(TRAIN_DATA_FOLDER)
    
    try:
        classifier = DocumentClassifier(model_choice, model_paths[model_choice], tokenizer_vectorizer_paths[model_choice], label_to_index)
    except Exception as e:
        return str(e)

    return classifier.classify_document(file_name)

all_models = [
    "Convolutional Neural Network (CNN)",
    "Recurrent Neural Network (RNN)",
    "Long Short-Term Memory (LSTM)",
    "Random Forest",
    "Naive Bayes",
    "XGBoost"
]

demo = gr.Interface(
    fn=run,
    inputs=["file", gr.Dropdown(all_models, label="MODEL")],
    outputs="text",
)

demo.launch()
