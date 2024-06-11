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
from tensorflow.keras import layers, models
import keras
from pe import TextPreprocessor
import seaborn as sns

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
        print('starting preprocessing')
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess(text)
        print(processed_text)
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

    def train_and_save_model(self, train_folder, model_save_path):
        # Load data
        X_train, y_train = self.load_data(train_folder)

        # Get unique labels
        unique_labels = set(y_train)

        # Convert labels to numerical values
        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}
        y_train = [self.label_to_index[label] for label in y_train]

        # Tokenize text data
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(X_train)
        X_train = self.tokenizer.texts_to_sequences(X_train)
        self.max_seq_length = max(len(seq) for seq in X_train)

        # Pad sequences
        X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=self.max_seq_length)

        # Convert data to NumPy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Save tokenizer
        joblib.dump(self.tokenizer, "tokenizer.joblib")

        # Define model architecture (to be implemented in child classes)
        self.define_model()

        # Compile model (to be implemented in child classes)
        self.compile_model()

        # Train model (to be implemented in child classes)
        history = self.train_model(X_train, y_train)

        # Save model
        self.model.save(model_save_path)

        return history

    def evaluate_model(self, model, X_test, y_test, history=None):
        # Predict probabilities
        y_pred_proba = model.predict(X_test)

        # Convert probabilities to class labels
        y_pred = np.argmax(y_pred_proba, axis=1)

        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.unique_labels))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)

        # Plot heatmap for confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.unique_labels,
                    yticklabels=self.unique_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Heatmap')
        plt.show()

        # Loss and accuracy plot
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

        # AUC ROC plot
        plt.figure(figsize=(6, 6))
        for i, label in enumerate(self.unique_labels):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def define_model(self):
        raise NotImplementedError("define_model method must be implemented in child classes")

    def compile_model(self):
        raise NotImplementedError("compile_model method must be implemented in child classes")

    def train_model(self, X_train, y_train):
        raise NotImplementedError("train_model method must be implemented in child classes")


class CNNModel(BaseModel):
    def define_model(self):
        self.model = models.Sequential([
            layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=64,
                             input_length=self.max_seq_length),
            layers.Conv1D(128, 5, activation='relu'),
            layers.GlobalMaxPooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(len(self.label_to_index), activation='softmax')
        ])

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train):
        history = self.model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
        return history


class RNNModel(BaseModel):
    def define_model(self):
        self.model = models.Sequential([
            layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=64,
                             input_length=self.max_seq_length),
            layers.SimpleRNN(128),
            layers.Dense(len(self.label_to_index), activation='softmax')
        ])

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train):
        history = self.model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
        return history


class LSTMModel(BaseModel):
    def define_model(self):
        self.model = models.Sequential([
            layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=64,
                             input_length=self.max_seq_length),
            layers.LSTM(128),
            layers.Dense(len(self.label_to_index), activation='softmax')
        ])

    def compile_model(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, X_train, y_train):
        history = self.model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)
        return history


def train_and_evaluate_model(model_type, train_folder, model_save_path, test_folder):
    if model_type == 'CNN':
        model = CNNModel()
    elif model_type == 'LSTM':
        model = LSTMModel()
    elif model_type == 'RNNModel':
        model = RNNModel()
    else:
        raise ValueError("Invalid model type. Please choose 'CNN' or 'LSTM'.")

    # Train and save model
    history = model.train_and_save_model(train_folder, model_save_path)

    # Load the saved model
    loaded_model = keras.models.load_model(model_save_path)

    # Load test data
    X_test, y_test = model.load_data(test_folder)
    y_test = [model.label_to_index[label] for label in y_test]
    X_test = model.tokenizer.texts_to_sequences(X_test)
    X_test = tf.keras.preprocessing.sequence.pad_sequences(X_test, maxlen=model.max_seq_length)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Evaluate the model
    model.evaluate_model(loaded_model, X_test, y_test, history=history)


# Path to train and test folders
train_folder = "Train_Data"
test_folder = "Test_Data"

# Train and evaluate CNN model
train_and_evaluate_model('CNN', train_folder, "text_classifier_model.h5", test_folder)

# Train and evaluate LSTM model
train_and_evaluate_model('LSTM', train_folder, "text_classifier_model_lstm.h5", test_folder)

# Train and evaluate RNNModel model
train_and_evaluate_model('RNNModel', train_folder, "text_classifier_model_RNNModel.h5", test_folder)
