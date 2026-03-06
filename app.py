import streamlit as st
import torch
import pandas as pd
import plotly.graph_objects as go
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re

st.set_page_config(
    page_title="Amazon Review Sentiment Dashboard",
    layout="wide"
)

st.title("Amazon Review Sentiment Dashboard")

st.write("Sentiment analysis using DistilBERT")

# ----------------------
# Load model
# ----------------------

@st.cache_resource
def load_model():

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()

# ----------------------
# Prediction function
# ----------------------

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

    neg = float(probs[0])
    pos = float(probs[1])

    sentiment = "Positive 😊" if pos > neg else "Negative 😞"

    return sentiment, pos, neg


# ----------------------
# Clean text
# ----------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-z ]", "", text)

    return text.split()


# ----------------------
# Sidebar
# ----------------------

mode = st.sidebar.radio(
    "Choose Mode",
    ["Single Review", "CSV Analysis"]
)

# ----------------------
# Single Review
# ----------------------

if mode == "Single Review":

    review = st.text_area("Enter review")

    if st.button("Analyze"):

        if review.strip() == "":
            st.warning("Enter a review")

        else:

            sentiment, pos, neg = predict_sentiment(review)

            st.success(f"Prediction: {sentiment}")

            fig = go.Figure(data=[
                go.Bar(name="Positive", x=["Sentiment"], y=[pos]),
                go.Bar(name="Negative", x=["Sentiment"], y=[neg])
            ])

            fig.update_layout(title="Sentiment Probability")

            st.plotly_chart(fig)

            confidence = max(pos, neg)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence*100,
                title={'text': "Confidence"},
                gauge={'axis': {'range': [0, 100]}}
            ))

            st.plotly_chart(gauge)

            words = clean_text(review)

            wc = WordCloud(width=800, height=400).generate(" ".join(words))

            plt.imshow(wc)

            plt.axis("off")

            st.pyplot(plt)

            freq = Counter(words).most_common(10)

            st.subheader("Top Keywords")

            for word, count in freq:

                st.write(word, ":", count)

# ----------------------
# CSV Analysis
# ----------------------

if mode == "CSV Analysis":

    file = st.file_uploader("Upload CSV with column 'Text'")

    if file:

        df = pd.read_csv(file)

        if "Text" not in df.columns:

            st.error("CSV must contain 'Text' column")

        else:

            sentiments = []

            for review in df["Text"]:

                sentiment, _, _ = predict_sentiment(str(review))

                sentiments.append(sentiment)

            df["Sentiment"] = sentiments

            st.dataframe(df.head())

            counts = df["Sentiment"].value_counts()

            fig = go.Figure([
                go.Bar(x=counts.index, y=counts.values)
            ])

            fig.update_layout(title="Sentiment Distribution")

            st.plotly_chart(fig)

            csv = df.to_csv(index=False).encode()

            st.download_button(
                "Download Results",
                csv,
                "sentiment_results.csv",
                "text/csv"
            )
