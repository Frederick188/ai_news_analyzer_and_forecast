import re
import pandas as pd
import unicodedata
import ftfy
from newsapi import NewsApiClient
from textblob import TextBlob
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from chronos import ChronosPipeline

# Streamlit Config 
st.set_page_config(page_title="AI News Analyzer & Forecast", layout="wide")

# Text Cleaning
def clean_text(text):
    if not text:
        return ""
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Sentiment Analysis 
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        sentiment = "Positive"
    elif polarity < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    return sentiment, polarity

# Fetch News 
@st.cache_data(ttl=3600)
def fetch_news(api_key, keywords, limit=20, pages=2):
    newsapi = NewsApiClient(api_key=api_key)
    all_articles = []

    for kw in keywords:
        for page in range(1, pages + 1):
            articles = newsapi.get_everything(
                q=kw,
                language="en",
                sort_by="publishedAt",
                page_size=limit,
                page=page
            ).get("articles", [])

            for a in articles:
                text = clean_text((a.get("title") or "") + " " + (a.get("description") or ""))
                sentiment, score = analyze_sentiment(text)

                all_articles.append({
                    "keyword": kw,
                    "title": a.get("title", ""),
                    "source": a.get("source", {}).get("name", "unknown"),
                    "text": text,
                    "date": a.get("publishedAt", ""),
                    "sentiment": sentiment,
                    "score": score
                })

    df = pd.DataFrame(all_articles)
    return df

# Forecast Setup 
@st.cache_resource
def load_chronos():
    return ChronosPipeline.from_pretrained("amazon/chronos-t5-tiny")

chronos = load_chronos()

def forecast_scores(daily, days=5):
    if len(daily) < 3:
        return []
    context = torch.tensor(daily["y"].values, dtype=torch.float32)
    preds = chronos.predict(context=context, prediction_length=days)
    preds_np = preds.detach().cpu().numpy() if isinstance(preds, torch.Tensor) else np.array(preds)
    forecast_mean = [float(preds_np[:, i].mean()) for i in range(days)]
    return forecast_mean

# Anomaly Detection 
def detect_anomalies(daily_score, z_threshold=2):
    y = daily_score["y"]
    mean, std = np.mean(y), np.std(y)
    anomalies = []
    for i, val in enumerate(y):
        z_score = (val - mean) / std if std > 0 else 0
        if abs(z_score) > z_threshold:
            anomalies.append((daily_score["ds"].iloc[i], val))
    return anomalies

# Tabs 
tab1, tab2 = st.tabs(["📰 Live News", "📊 Forecast Dashboard"])

# Tab 1: Live News 
with tab1:
    st.header("Fetch Latest News & Sentiment")

    api_key_input = st.text_input("🔑 Enter your NewsAPI Key", type="password")
    keywords_input = st.text_area(
        "Enter keywords (comma-separated):",
        "Artificial Intelligence, Machine Learning, Deep Learning, Neural Networks, NLP, Generative AI, Computer Vision, AI Ethics, Chatbots, Robotics"
    )
    keywords = [kw.strip() for kw in keywords_input.split(",") if kw.strip()]

    limit = st.slider("Articles per page", 5, 50, 20)
    pages = st.slider("Pages to fetch", 1, 5, 2)
    fetch_btn = st.button("🚀 Fetch News")

    if fetch_btn:
        if not api_key_input:
            st.error("⚠️ Please enter your NewsAPI key.")
        else:
            with st.spinner("Fetching and analyzing news..."):
                df_news = fetch_news(api_key_input, keywords, limit=limit, pages=pages)

            if not df_news.empty:
                st.success(f"✅ Fetched {len(df_news)} articles.")
                st.dataframe(df_news, use_container_width=True)

                st.subheader("📊 Sentiment Distribution")
                st.bar_chart(df_news["sentiment"].value_counts())

                csv = df_news.to_csv(index=False, encoding="utf-8-sig")
                st.download_button("💾 Download CSV", data=csv, file_name="ai_news_sentiment.csv", mime="text/csv")
            else:
                st.warning("⚠️ No articles found. Try different keywords.")

# Tab 2: Forecast Dashboard 
with tab2:
    st.header("Forecast Dashboard from CSV")
    uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"], key="forecast_upload")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        st.sidebar.header("Filters")
        topics = st.sidebar.multiselect("Select Keywords", df["keyword"].unique(), default=df["keyword"].unique()[:3])
        date_range = st.sidebar.date_input("Select Date Range", [df["date"].min(), df["date"].max()])
        forecast_days = st.sidebar.slider("Forecast Horizon (days)", 3, 10, 5)

        start, end = date_range if len(date_range) == 2 else (df["date"].min(), df["date"].max())
        filtered = df[df["keyword"].isin(topics) & df["date"].between(start, end)]

        if not filtered.empty:
            subtab_trends, subtab_benchmarks, subtab_alerts, subtab_forecast = st.tabs(
                ["📈 Trends", "📊 Benchmarks", "🚨 Alerts", "🔮 Forecast"]
            )

            # Trends 
            with subtab_trends:
                st.subheader("Sentiment Trend Over Time")
                plt.figure(figsize=(10,5))
                sns.lineplot(data=filtered, x="date", y="score", hue="keyword", marker="o")
                plt.xticks(rotation=45)
                st.pyplot(plt)

            # Benchmarks 
            with subtab_benchmarks:
                st.subheader("Keyword Benchmarks")
                avg_sentiment = filtered.groupby("keyword")["score"].mean().sort_values(ascending=False)
                mention_counts = filtered["keyword"].value_counts()

                plt.figure(figsize=(8,4))
                avg_sentiment.plot(kind="bar", color="skyblue")
                plt.ylabel("Avg Sentiment Score")
                plt.xticks(rotation=45)
                st.pyplot(plt)

                plt.figure(figsize=(8,4))
                mention_counts.plot(kind="bar", color="orange")
                plt.ylabel("Mentions Count")
                plt.xticks(rotation=45)
                st.pyplot(plt)

            # Alerts 
            with subtab_alerts:
                st.subheader("Alerts (Negative Sentiment)")
                neg_alerts = filtered[filtered["score"] < -0.2][["date", "keyword", "title"]]
                st.table(neg_alerts if not neg_alerts.empty else pd.DataFrame([{"Alert": "No major alerts ✅"}]))

            # Forecast 
            with subtab_forecast:
                st.subheader(f"Sentiment Forecast ({forecast_days} days)")
                daily_score = filtered.groupby("date")["score"].mean().reset_index()
                daily_score.columns = ["ds", "y"]

                if len(daily_score) < 3:
                    st.warning("⚠️ Not enough data points to forecast. Please provide at least 3 days of sentiment data.")
                else:
                    forecast_vals = forecast_scores(daily_score, days=forecast_days)
                    if forecast_vals:
                        future_dates = pd.date_range(
                            start=pd.to_datetime(daily_score["ds"].iloc[-1]) + pd.Timedelta(days=1),
                            periods=forecast_days
                        )

                        # Combine historical and forecast for anomaly detection
                        all_dates = list(daily_score["ds"]) + list(future_dates.date)
                        all_vals = list(daily_score["y"]) + list(forecast_vals)
                        combined_df = pd.DataFrame({"ds": all_dates, "y": all_vals})

                        # Detect anomalies in combined data
                        anomalies = detect_anomalies(combined_df, z_threshold=2)

                        # Plot
                        plt.figure(figsize=(10,5))
                        plt.plot(daily_score["ds"], daily_score["y"], color="blue", marker="o", label="Actual")
                        plt.plot(future_dates, forecast_vals, color="orange", marker="o", label="Forecast")
                        if anomalies:
                            anomaly_dates, anomaly_vals = zip(*anomalies)
                            plt.scatter(anomaly_dates, anomaly_vals, color="red", s=90, label="Anomalies")
                        plt.xlabel("Date")
                        plt.ylabel("Sentiment Score")
                        plt.legend()
                        plt.xticks(rotation=45)
                        st.pyplot(plt)

                        # Show anomalies table
                        if anomalies:
                            st.subheader("Anomalies Detected")
                            st.table(pd.DataFrame(anomalies, columns=["Date", "Sentiment Score"]))

                        # Show forecast table
                        st.subheader("Forecasted Sentiment Values")
                        st.dataframe(pd.DataFrame({"date": future_dates.date, "forecast": forecast_vals}))

    else:
        st.info("👈 Please upload a CSV file to start analyzing and forecasting.")


