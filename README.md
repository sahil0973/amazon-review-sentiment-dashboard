# Amazon Review Sentiment Dashboard

An interactive **AI-powered dashboard** that analyzes Amazon product reviews and classifies them into **positive or negative sentiment** using a **DistilBERT transformer model**.
The project combines **Natural Language Processing (NLP)** with an intuitive **Streamlit dashboard** to provide real-time insights from customer reviews.

---

## Project Overview

Customer reviews contain valuable insights about products and services.
This project builds a **sentiment analysis system** using deep learning and deploys it through an interactive dashboard.

Users can:

* Analyze individual reviews in real time
* Upload CSV files for batch sentiment analysis
* Visualize sentiment probabilities
* View confidence scores and statistics
* Generate word clouds from reviews

---

## Key Features

* Transformer-based sentiment classification using **DistilBERT**
* Real-time prediction through **Streamlit dashboard**
* Sentiment probability visualization
* Confidence gauge meter
* Word cloud generation
* Top keyword extraction
* CSV upload for bulk review analysis
* Downloadable prediction results

---

## Project Architecture

```text
User Input / CSV Upload
        ↓
Text Preprocessing
        ↓
DistilBERT Tokenizer
        ↓
DistilBERT Model
        ↓
Sentiment Prediction
        ↓
Interactive Streamlit Dashboard
```

---

## Dataset

Dataset used: **Amazon Fine Food Reviews**

Download from Kaggle:

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

Dataset contains:

* 568,000+ product reviews
* Product ratings
* Review summaries
* Review text

Sentiment labels are created from ratings:

```text
Ratings 1–3 → Negative Sentiment  
Ratings 4–5 → Positive Sentiment
```

---

## Project Structure

```text
amazon-review-sentiment-dashboard
│
├── notebook.ipynb
│   Model training notebook
│
├── app.py
│   Streamlit dashboard
│
├── Reviews.csv
│   Dataset file
│
├── sentiment_model/
│   Saved DistilBERT model
│
├── requirements.txt
│   Python dependencies
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/amazon-review-sentiment-dashboard.git
cd amazon-review-sentiment-dashboard
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Model Training

Open the notebook:

```text
notebook.ipynb
```

Steps inside the notebook:

1. Load dataset
2. Clean review text
3. Convert ratings to sentiment labels
4. Tokenize text using DistilBERT tokenizer
5. Train transformer model
6. Evaluate model performance
7. Save trained model

The trained model is saved in:

```text
sentiment_model/
```

---

## Running the Dashboard

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Open in your browser:

```text
http://localhost:8501
```

---

## Dashboard Capabilities

The dashboard allows users to:

### Single Review Analysis

* Enter a product review
* Predict sentiment
* View probability chart
* Check confidence score
* Generate word cloud
* View review statistics

### Bulk Review Analysis

* Upload a CSV file containing reviews
* Analyze thousands of reviews
* View sentiment distribution chart
* Generate dataset word cloud
* Download results as CSV

---

## Example Prediction

Input review:

```text
This product tastes amazing and I would definitely buy it again.
```

Output:

```text
Sentiment: Positive
Confidence: 91%
```

---

## Technologies Used

* Python
* PyTorch
* HuggingFace Transformers
* DistilBERT
* Streamlit
* Pandas
* Scikit-learn
* Plotly
* WordCloud

---

## Applications

This system can be used for:

* Customer feedback analysis
* Product review monitoring
* E-commerce sentiment tracking
* Brand reputation management
* Market research insights

---

## Future Improvements

Possible extensions:

* Multiclass sentiment classification
* Aspect-based sentiment analysis
* Real-time review scraping
* Cloud deployment
* Mobile-friendly dashboard

---

## Author

Sahil 

---

## License

This project is licensed under the MIT License.
