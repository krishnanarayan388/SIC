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
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping




class TextPreprocessor:
    def preprocess(self, text):
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

class TextClassifier:
    def __init__(self, model_type='CNN'):
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.model_type = model_type
        self.max_seq_length = 500  # Maximum sequence length

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

    def train_and_evaluate_model(self, train_folder, test_folder, model_save_path, tokenizer_save_path):
        # Load data
        X_train, y_train = self.load_data(train_folder)
        X_test, y_test = self.load_data(test_folder)

        # Ensure all documents are strings
        X_train = [" ".join(doc) if isinstance(doc, list) else doc for doc in X_train]
        X_test = [" ".join(doc) if isinstance(doc, list) else doc for doc in X_test]

        # Label encoding
        y_train = self.label_encoder.fit_transform(y_train)
        y_test = self.label_encoder.transform(y_test)

        # Tokenization and padding
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(X_train)
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_seq_length)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_seq_length)

        # Choose model
        if self.model_type == 'CNN':
            self.model = self.create_cnn_model()
        elif self.model_type == 'RNN':
            self.model = self.create_rnn_model()
        elif self.model_type == 'LSTM':
            self.model = self.create_lstm_model()
        else:
            raise ValueError("Invalid model type. Please choose 'CNN', 'RNN', or 'LSTM'.")

        # Compile model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        self.model.fit(X_train_pad, y_train, validation_data=(X_test_pad, y_test), epochs=10, batch_size=32, callbacks=[early_stopping])

        # Save model and tokenizer
        self.model.save(model_save_path)
        joblib.dump(self.tokenizer, tokenizer_save_path)

        # Evaluate model
        y_pred = self.model.predict(X_test_pad)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred_classes, target_names=self.label_encoder.classes_))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        print("Confusion Matrix:")
        print(cm)

        # Plot heatmap for confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix Heatmap')
        plt.show()

        # AUC ROC plot
        plt.figure(figsize=(6, 6))
        for i, label in enumerate(self.label_encoder.classes_):
            fpr, tpr, _ = roc_curve(y_test == i, y_pred[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def create_cnn_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=128, input_length=self.max_seq_length))
        model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))
        return model

    def create_rnn_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=128, input_length=self.max_seq_length))
        model.add(SimpleRNN(128))
        model.add(Dense(10, activation='softmax'))
        return model

    def create_lstm_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=5000, output_dim=128, input_length=self.max_seq_length))
        model.add(LSTM(128))
        model.add(Dense(10, activation='softmax'))
        return model

    def classify_document(self, file_name, model_path, tokenizer_path):
        text = self.extract_text_from_file(file_name)
        if text:
            # Load the trained model and tokenizer
            if not self.model:
                self.model = tf.keras.models.load_model(model_path)
            if not self.tokenizer:
                self.tokenizer = joblib.load(tokenizer_path)

            sequence = self.tokenizer.texts_to_sequences([text])
            padded_sequence = pad_sequences(sequence, maxlen=self.max_seq_length)
            predicted_class_indices = np.argmax(self.model.predict(padded_sequence), axis=1)
            return self.label_encoder.inverse_transform(predicted_class_indices)[0]
        else:
            return "Unable to extract text from document"

# Usage
# Path to train and test folders
train_folder = "Train_Data"
test_folder = "Test_Data"

# Train and evaluate CNN model
#cnn_classifier = TextClassifier(model_type='CNN')
#cnn_classifier.train_and_evaluate_model(train_folder, test_folder, "cnn_model.h5", "tokenizer_cnn.joblib")

# Train and evaluate RNN model
rnn_classifier = TextClassifier(model_type='RNN')
rnn_classifier.train_and_evaluate_model(train_folder, test_folder, "rnn_model.h5", "tokenizer_rnn.joblib")

# Train and evaluate LSTM model
lstm_classifier = TextClassifier(model_type='LSTM')
lstm_classifier.train_and_evaluate_model(train_folder, test_folder, "lstm_model.h5", "tokenizer_lstm.joblib")

# Example of classifying a document
file_name = "path_to_document"
model_path = "cnn_model.h5"
tokenizer_path = "tokenizer_cnn.joblib"
print(cnn_classifier.classify_document(file_name, model_path, tokenizer_path))
