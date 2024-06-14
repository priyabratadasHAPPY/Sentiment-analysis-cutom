# !pip install streamlit transformers

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModelForSeq2SeqLM
import numpy as np
from scipy.special import softmax

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Load the model and tokenizer
MODEL = "priyabrat/sentiment_analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Define the scoring function
def score(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=50, add_special_tokens=True)
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    result = []
    for i in range(scores.shape[0]):
        l = config.id2label[ranking[i]]
        s = scores[ranking[i]] * 100  # Convert to percentage
        result.append(f"{l}: {s:.2f}%")  # Format the score as a percentage
    return result

# Streamlit app
st.title("Blog Sentiment Analysis")

# Text area for user input
text = st.text_area("Enter the blog content here:")

if st.button('Analyze Sentiment'):
    # Perform sentiment analysis
    sentiment_scores = score(text)
    st.write("Sentiment Analysis Results:")
    for sentiment in sentiment_scores:
        st.write(sentiment)
