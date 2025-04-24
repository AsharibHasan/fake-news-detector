import pandas as pd
import numpy as np
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load dataset
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Label the data
fake["label"] = 0  # Fake = 0
real["label"] = 1  # Real = 1

# Combine and shuffle
data = pd.concat([fake, real], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

print("Data shape:", data.shape)
print(data["label"].value_counts())

# Text Preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Combine title + text, then preprocess
data["text"] = data["title"] + " " + data["text"]
data["text"] = data["text"].apply(preprocess)

# Train-test split
X = data["text"]
y = data["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test_tfidf)

print("\n=== Model Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from lime.lime_text import LimeTextExplainer

# Define class names
class_names = ['Fake', 'Real']

# Create a LIME explainer
explainer = LimeTextExplainer(class_names=class_names)

# Define prediction function for raw text
def predict_proba(texts):
    tfidf = vectorizer.transform(texts)
    return model.predict_proba(tfidf)

# Choose an article from test set
i = 15  # You can try different values
text_sample = X_test.iloc[i]
label = y_test.iloc[i]

print("Text:\n", text_sample)
print("Actual Label:", class_names[label])

# Run LIME explainer
exp = explainer.explain_instance(text_sample, predict_proba, num_features=10)

# Show explanation in browser
exp.show_in_notebook()
# OR open in browser if notebook not used
exp.save_to_file('lime_explanation.html')

import pickle

# Save the trained model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the fitted vectorizer
with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)