import os
import re
import PyPDF2
import docx
import pytesseract
from PIL import Image
import io
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

class BaseModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_seq_length = None
        self.label_to_index = None
        self.unique_labels = None

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
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

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
        unique_labels = set()
        for subclass in os.listdir(folder_path):
            subclass_path = os.path.join(folder_path, subclass)
            if os.path.isdir(subclass_path):
                for file_name in os.listdir(subclass_path):
                    file_path = os.path.join(subclass_path, file_name)
                    text = self.extract_text_from_file(file_path)
                    if text:
                        data.append(text)
                        labels.append(subclass)
                        unique_labels.add(subclass)
        self.unique_labels = list(unique_labels)
        return data, labels

    def prepare_data(self, X, y):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(X)
        X = self.tokenizer.texts_to_sequences(X)
        self.max_seq_length = max(len(seq) for seq in X)
        X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.max_seq_length)
        self.label_to_index = {label: i for i, label in enumerate(self.unique_labels)}
        y = np.array([self.label_to_index[label] for label in y])
        return np.array(X), y

    def evaluate_model(self, model, X_test, y_test, history=None):
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        if history:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.subplot(1, 2, 2)
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.tight_layout()
            plt.show()
        plt.figure(figsize=(6, 6))
        for i in range(len(self.unique_labels)):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

def full_model_fine_tuning(model_path, train_folder, epochs=10, batch_size=16, learning_rate=1e-5):
    class FineTuningModel(BaseModel):
        def __init__(self, model_path):
            super().__init__()
            self.model = load_model(model_path)

        def fine_tune(self, train_folder, epochs, batch_size, learning_rate):
            X_train, y_train = self.load_data(train_folder)
            X_train, y_train = self.prepare_data(X_train, y_train)
            for layer in self.model.layers:
                layer.trainable = True
            self.model.compile(optimizer=Adam(learning_rate=learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
            return history

    fine_tuning_model = FineTuningModel(model_path)
    history = fine_tuning_model.fine_tune(train_folder, epochs, batch_size, learning_rate)
    return fine_tuning_model.model, history

# Path to the pre-trained model and train folder
model_path = "text_classifier_model.h5"
train_folder = "Train_Data"

# Fine-tune the full model
model, history = full_model_fine_tuning(model_path, train_folder)

# Evaluate the fine-tuned model on test data
test_folder = "Test_Data"
fine_tuning_model = BaseModel()
X_test, y_test = fine_tuning_model.load_data(test_folder)
X_test, y_test = fine_tuning_model.prepare_data(X_test, y_test)
fine_tuning_model.evaluate_model(model, X_test, y_test, history=history)
