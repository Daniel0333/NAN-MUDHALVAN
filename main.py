import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

sentiment_map = {
    0: ("Negative", "Angry/Sad"),
    1: ("Positive", "Happy/Excited")
}

def analyze_sentiment(text):
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        output = model(**encoded_input)
        logits = output.logits
        scores_tensor = F.softmax(logits, dim=1)[0]
        scores = scores_tensor.tolist()
        max_index = int(torch.argmax(scores_tensor))
    sentiment, emotion = sentiment_map[max_index]
    confidence = round(scores[max_index], 3)
    return sentiment, emotion, confidence, scores

# Streamlit UI
st.title("Sentiment & Emotion Analysis Demo")
st.markdown("---")

user_input = st.text_input("Enter your social media text here:")

if user_input:
    if len(user_input.strip()) < 3:
        st.warning("Please enter a more meaningful input.")
    else:
        sentiment, emotion, confidence, scores = analyze_sentiment(user_input)
        st.write(f"**Sentiment:** {sentiment}  (Confidence: {confidence})")
        st.write(f"**Emotion:** {emotion}")
        st.write("**Class Scores:**")
        st.json({
            'Negative': round(scores[0], 3),
            'Positive': round(scores[1], 3)
        })
