import os
import re
import PyPDF2
import docx
import pytesseract
from PIL import Image
import io
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from pe import TextPreprocessor

class TextClassifier:
    def __init__(self, model_type='RandomForest'):
        self.model = None
        self.vectorizer = None
        self.label_encoder = LabelEncoder()
        self.model_type = model_type

    def extract_text_from_pdf(self, pdf_path):
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess(text)
        return processed_text

    def extract_text_from_docx(self, docx_path):
        text = ""
        doc = docx.Document(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text
        # Extract text from images
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

    def extract_text_from_file(self, file_path):
        if file_path.endswith('.pdf') or file_path.endswith('.PDF'):
            return self.extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            return self.extract_text_from_docx(file_path)
        else:
            return ""

    def load_data(self, folder_path):
        data = []
        labels = []
        for subclass in os.listdir(folder_path):
            subclass_path = os.path.join(folder_path, subclass)
            if os.path.isdir(subclass_path):
                for file_name in os.listdir(subclass_path):
                    file_path = os.path.join(subclass_path, file_name)
                    text = self.extract_text_from_file(file_path)
                    if text:
                        data.append(text)
                        labels.append(subclass)
        return data, labels

    def train_and_evaluate_model(self, train_folder, test_folder, model_save_path, vectorizer_save_path):
        # Load data
        X_train, y_train = self.load_data(train_folder)
        X_test, y_test = self.load_data(test_folder)

        # Ensure all documents are strings
        X_train = [" ".join(doc) if isinstance(doc, list) else doc for doc in X_train]
        X_test = [" ".join(doc) if isinstance(doc, list) else doc for doc in X_test]

        # Label encoding
        y_train = self.label_encoder.fit_transform(y_train)
        y_test = self.label_encoder.transform(y_test)

        # Feature extraction using TF-IDF
        self.vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = self.vectorizer.fit_transform(X_train).toarray()
        X_test_tfidf = self.vectorizer.transform(X_test).toarray()

        # Choose model
        if self.model_type == 'RandomForest':
            self.model = RandomForestClassifier()
        elif self.model_type == 'XGBClassifier':
            self.model = XGBClassifier()
        elif self.model_type == 'NaiveBayes':
            self.model = MultinomialNB()
        else:
            raise ValueError("Invalid model type. Please choose 'RandomForest', 'XGBClassifier', or 'NaiveBayes'.")

        # Train model
        self.model.fit(X_train_tfidf, y_train)

        # Save model and vectorizer
        joblib.dump(self.model, model_save_path)
        joblib.dump(self.vectorizer, vectorizer_save_path)

        # Evaluate model
        y_pred = self.model.predict(X_test_tfidf)
        y_pred_proba = self.model.predict_proba(X_test_tfidf)

        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Plot heatmap for confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Heatmap')
        plt.show()

        # AUC ROC plot
        plt.figure(figsize=(6, 6))
        for i, label in enumerate(self.label_encoder.classes_):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def classify_document(self, file_name, model_path, vectorizer_path):
        text = self.extract_text_from_file(file_name)
        if text:
            # Load the trained model and vectorizer
            if not self.model:
                self.model = joblib.load(model_path)
            if not self.vectorizer:
                self.vectorizer = joblib.load(vectorizer_path)

            features = self.vectorizer.transform([text]).toarray()
            predicted_class_indices = self.model.predict(features)
            return self.label_encoder.inverse_transform(predicted_class_indices)[0]
        else:
            return "Unable to extract text from document"

# Usage
# Path to train and test folders
train_folder = "Train_Data"
test_folder = "Test_Data"

# Train and evaluate Random Forest model
rf_classifier = TextClassifier(model_type='RandomForest')
rf_classifier.train_and_evaluate_model(train_folder, test_folder, "random_forest_model.joblib", "vectorizer.joblib")

# Train and evaluate XGBoost model
xgb_classifier = TextClassifier(model_type='XGBClassifier')
xgb_classifier.train_and_evaluate_model(train_folder, test_folder, "xgb_model.joblib", "vectorizer.joblib")

# Train and evaluate Naive Bayes model
nb_classifier = TextClassifier(model_type='NaiveBayes')
nb_classifier.train_and_evaluate_model(train_folder, test_folder, "naive_bayes_model.joblib", "vectorizer.joblib")

# Example of classifying a document
file_name = "path_to_document"
model_path = "random_forest_model.joblib"
vectorizer_path = "vectorizer.joblib"
print(rf_classifier.classify_document(file_name, model_path, vectorizer_path))
