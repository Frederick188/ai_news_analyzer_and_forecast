# AI News Analyzer & Forecast

A **Streamlit-based web application** that fetches live news, analyzes sentiment, detects anomalies, and forecasts sentiment trends using AI and Chronos. Perfect for monitoring the pulse of topics like Artificial Intelligence, Machine Learning, and more.

---

## **Features**

### 1. Live News Analysis
- Fetch latest news articles from [NewsAPI](https://newsapi.org/) based on your keywords.
- Clean and preprocess text automatically.
- Perform sentiment analysis (Positive, Negative, Neutral) using TextBlob.
- Visualize sentiment distribution with interactive bar charts.
- Download the analyzed data as CSV.

### 2. Forecast Dashboard
- Upload your historical sentiment CSV file.
- Filter data by keyword and date range.
- Forecast future sentiment trends (3–10 days) using Chronos T5 model.
- Detect anomalies in sentiment scores using Z-score method.
- Visualize actual, forecasted, and anomalous sentiment in line charts.
- View tables of forecasted values and detected anomalies.

---
