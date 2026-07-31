# Real-Time Industry Insight & Strategic Intelligence System

## Overview

Real-Time Industry Insight & Strategic Intelligence System is an end-to-end AI-powered application that collects the latest Artificial Intelligence news using NewsAPI, performs sentiment analysis using Natural Language Processing (NLP), and forecasts future sentiment trends using Amazon Chronos, a transformer-based time-series forecasting model.

The application provides an interactive Streamlit dashboard where users can fetch live AI news, analyze sentiment across multiple AI topics, visualize historical trends, identify negative sentiment alerts, and forecast future sentiment scores.

The project combines real-time data collection, text preprocessing, sentiment analysis, forecasting, anomaly detection, and interactive visualization into a single platform.

---

# Features

## Live News Collection

- Fetches real-time AI-related news using NewsAPI.
- Supports multiple AI-related keywords.
- User-configurable number of articles and pages.
- Displays source, publication date, sentiment, and polarity score.
- Download analyzed news as CSV.

---

## Text Preprocessing

Before sentiment analysis, the application cleans each news article by:

- Fixing text encoding using ftfy.
- Unicode normalization.
- Removing URLs.
- Removing HTML tags.
- Removing email addresses.
- Removing hashtags and mentions.
- Removing special characters.
- Converting text to lowercase.
- Removing extra whitespaces.

This preprocessing improves sentiment analysis quality.

---

## Sentiment Analysis

Sentiment analysis is performed using TextBlob.

Each article receives:

- Sentiment Label
  - Positive
  - Neutral
  - Negative

- Sentiment Polarity Score
  - Range: -1 to +1

Classification rules:

| Polarity | Sentiment |
|----------|-----------|
| > 0.05 | Positive |
| < -0.05 | Negative |
| Otherwise | Neutral |

---

## Forecasting

The project uses Amazon Chronos (Chronos-T5 Tiny), a transformer-based time-series forecasting model.

Historical daily sentiment scores are aggregated and used as input to predict future sentiment values.

Users can forecast sentiment for 3 to 10 days.

---

## Anomaly Detection

The application detects abnormal sentiment changes using the Z-score statistical method.

A sentiment value is considered an anomaly if:

|Z-score| > 2

Detected anomalies are:

- Highlighted on the forecast graph.
- Displayed in a separate table.

---

## Dashboard Modules

The application contains two main modules.

### 1. Live News

Users can:

- Enter their NewsAPI key.
- Choose AI-related keywords.
- Specify the number of articles.
- Fetch live news.
- View sentiment analysis.
- Download the dataset as CSV.

### 2. Forecast Dashboard

Users can upload a previously generated CSV file and perform:

- Sentiment trend analysis
- Keyword benchmarking
- Negative sentiment alerts
- Future sentiment forecasting
- Anomaly detection

---

# Workflow

The project follows the workflow below:

User Input

↓

NewsAPI

↓

Fetch News Articles

↓

Text Cleaning

↓

Sentiment Analysis (TextBlob)

↓

CSV Dataset Generation

↓

Upload Dataset

↓

Historical Trend Analysis

↓

Amazon Chronos Forecasting

↓

Anomaly Detection

↓

Interactive Streamlit Dashboard

---

# Technologies Used

## Programming Language

- Python

## Framework

- Streamlit

## Libraries

- Pandas
- NumPy
- TextBlob
- Matplotlib
- Seaborn
- PyTorch
- Chronos
- ftfy
- Regex
- unicodedata

## API

- NewsAPI

---

# Project Structure

AI-News-Analyzer/

│

├── app.py

├── README.md

├── requirements.txt

├── ai_news_sentiment.csv

└── assets/

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/AI-News-Analyzer.git

cd AI-News-Analyzer
