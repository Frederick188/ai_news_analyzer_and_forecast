from newsapi import NewsApiClient
import pandas as pd
from textblob import TextBlob

# Initialize News API Client
newsapi = NewsApiClient(api_key="your_api_key")

# Fetch AI-related news
ai_news = newsapi.get_everything(
    q="artificial intelligence",
    language="en",
    sort_by="publishedAt",
    page_size=100
)

# Extract articles
articles = ai_news.get("articles", [])
df = pd.DataFrame(articles)

# Function to classify sentiment
def get_sentiment(text):
    analysis = TextBlob(str(text))
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply sentiment on title (you can also use description/content)
df["sentiment"] = df["title"].apply(get_sentiment)

print("\nArtificial Intelligence News Headlines with Sentiment\n")
for i, row in enumerate(df[["title", "sentiment"]].values, start=1):
    print(f"{i}. {row[0]}  --->  {row[1]}")

# Save to CSV
df.to_csv("ai_news_with_sentiment.csv", index=False)
