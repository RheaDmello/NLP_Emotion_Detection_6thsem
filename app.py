import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download stopwords
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model and vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
lr_model = pickle.load(open("model.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Function to clean input text
def clean_text_streamlit(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = text.split()  # Tokenize
    lemmatizer = WordNetLemmatizer()
    text = [lemmatizer.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    return ' '.join(text)

# Streamlit App
st.title("Emotion Detection from Text")
st.write("Enter a sentence to detect its emotion!")

user_input = st.text_area("Your Text", "I am feeling happy today!")
if st.button("Predict Emotion"):
    cleaned_input = clean_text_streamlit(user_input)
    vectorized_input = vectorizer.transform([cleaned_input]).toarray()
    prediction = label_encoder.inverse_transform(lr_model.predict(vectorized_input))
    st.write(f"Predicted Emotion: {prediction[0]}")
