import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import fitz  # PyMuPDF for PDF reading

# ✅ Streamlit app setup (must be first Streamlit command)
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("🧠 Fake News Detector")
st.markdown("Paste a news article or headline below to find out if it's *Real* or *Fake*.")
st.markdown("---")

# Input from textarea or file
st.subheader("Input Options:")

# Text area input
news = st.text_area("Paste news text here (or upload a file):", height=200)

# File uploader
uploaded_file = st.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])

if uploaded_file is not None:
    if uploaded_file.type == "text/plain":
        # Read text from .txt
        news = uploaded_file.read().decode("utf-8")
        st.success("Text file uploaded successfully!")
    elif uploaded_file.type == "application/pdf":
        # Read text from .pdf using PyMuPDF
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        pdf_text = ""
        for page in doc:
            pdf_text += page.get_text()
        news = pdf_text
        st.success("PDF file uploaded successfully!")

    # Show preview
    if news:
        st.write("Preview:")
        st.code(news[:300] + "..." if len(news) > 300 else news)

# Load the model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Analyze button
if st.button("Analyze"):
    if news.strip() == "":
        st.warning("Please enter some news text.")
    else:
        vector_input = vectorizer.transform([news])
        prediction = model.predict(vector_input)[0]
        proba = model.predict_proba(vector_input)[0]
        confidence = round(np.max(proba) * 100, 2)

        if prediction == 1:
            st.success(f"*This news article is likely Real.*")
            st.info(f"Confidence: {confidence}%")
        else:
            st.error(f"*This news article is likely Fake.*")
            st.info(f"Confidence: {confidence}%")

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Word Cloud Section
st.subheader("Word Cloud from Uploaded/Typed Text")

if news.strip():
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(news)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Created by *Syed Asharib* | Powered by Machine Learning")

