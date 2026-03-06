import streamlit as st
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import plotly.graph_objects as go
import time

# ----------------------------
# PAGE CONFIG
# ----------------------------

st.set_page_config(
    page_title="Amazon Sentiment AI",
    layout="wide",
)

# ----------------------------
# LOAD MODEL
# ----------------------------

@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizer.from_pretrained("sentiment_model")
    model = DistilBertForSequenceClassification.from_pretrained("sentiment_model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ----------------------------
# SENTIMENT FUNCTION
# ----------------------------

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

    negative_prob = float(probs[0])
    positive_prob = float(probs[1])

    prediction = torch.argmax(probs).item()

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return sentiment, positive_prob, negative_prob


# ----------------------------
# MODERN CSS
# ----------------------------

st.markdown("""
<style>

body {
background-color:#0E1117;
color:white;
}

.big-title {
font-size:55px;
font-weight:800;
text-align:center;
background: linear-gradient(90deg,#00DBDE,#FC00FF);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
animation: glow 2s infinite alternate;
}

@keyframes glow {
from {opacity:0.7;}
to {opacity:1;}
}

.card {
background: rgba(255,255,255,0.05);
padding:25px;
border-radius:18px;
backdrop-filter: blur(10px);
margin-bottom:20px;
box-shadow:0px 0px 15px rgba(0,0,0,0.4);
}

.stButton>button {
background: linear-gradient(90deg,#00DBDE,#FC00FF);
color:white;
font-size:18px;
height:50px;
width:220px;
border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# TITLE
# ----------------------------

st.markdown('<div class="big-title">Amazon Review Sentiment AI</div>', unsafe_allow_html=True)

st.write("### Transformer-based sentiment analysis using DistilBERT")

# ----------------------------
# SIDEBAR
# ----------------------------

st.sidebar.title("AI Model Info")

st.sidebar.markdown("""
Model : DistilBERT  
Training Samples : 5000  
Environment : CPU  
Task : Sentiment Classification  
""")

# ----------------------------
# INPUT SECTION
# ----------------------------

st.markdown('<div class="card">', unsafe_allow_html=True)

review = st.text_area("Write your product review here", height=150)

analyze = st.button("Analyze Sentiment")

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------
# ANALYSIS
# ----------------------------

if analyze:

    if review.strip() == "":
        st.warning("Please enter a review")
    else:

        with st.spinner("Analyzing sentiment with AI..."):
            time.sleep(1.5)

        sentiment, pos_prob, neg_prob = predict_sentiment(review)

        st.success(f"### Prediction : {sentiment}")

        # ----------------------------
        # PROBABILITY BAR
        # ----------------------------

        fig = go.Figure(data=[
            go.Bar(name="Positive", x=["Sentiment"], y=[pos_prob]),
            go.Bar(name="Negative", x=["Sentiment"], y=[neg_prob])
        ])

        fig.update_layout(
            template="plotly_dark",
            title="Sentiment Probability",
            barmode="group"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------
        # CONFIDENCE GAUGE
        # ----------------------------

        confidence = max(pos_prob, neg_prob)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence*100,
            title={'text': "Confidence Level"},
            gauge={
                'axis': {'range': [0,100]},
                'bar': {'color': "#00DBDE"},
                'steps': [
                    {'range':[0,50],'color':"#8B0000"},
                    {'range':[50,75],'color':"#FFA500"},
                    {'range':[75,100],'color':"#00FF7F"},
                ]
            }
        ))

        gauge.update_layout(template="plotly_dark")

        st.plotly_chart(gauge, use_container_width=True)

        # ----------------------------
        # STATS
        # ----------------------------

        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("Review Statistics")

        words = len(review.split())
        chars = len(review)

        col1, col2 = st.columns(2)

        col1.metric("Word Count", words)
        col2.metric("Character Count", chars)

        st.markdown('</div>', unsafe_allow_html=True)

        # ----------------------------
        # PROGRESS ANIMATION
        # ----------------------------

        st.subheader("Confidence Progress")

        progress = st.progress(0)

        for i in range(int(confidence*100)):
            progress.progress(i+1)
            time.sleep(0.01)

        # ----------------------------
        # INSIGHT
        # ----------------------------

        if confidence > 0.85:
            st.info("🔥 High confidence prediction")
        elif confidence > 0.65:
            st.info("⚡ Moderate confidence prediction")
        else:
            st.warning("⚠ Low confidence prediction")