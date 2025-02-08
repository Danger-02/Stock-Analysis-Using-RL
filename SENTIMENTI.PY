import requests
from textblob import TextBlob
from collections import Counter

# NewsAPI credentials
NEWSAPI_KEY = '9c8a4deaa04449caa3dfd14ab1ee39db'

# Fetch news articles related to a stock
def fetch_news(query, count=10):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_KEY}&pageSize={count}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            articles = response.json().get('articles', [])
            # Handle cases where title or description is None
            return [f"{article['title'] or ''} {article['description'] or ''}" for article in articles]
        else:
            print("Failed to fetch news:", response.status_code)
            return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char.isalnum() or char == ' '])  # Remove special characters
    return text

# Analyze sentiment using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity < 0:
        return 'negative'
    else:
        return 'neutral'

# Map sentiment to action
def sentiment_to_action(sentiment):
    if sentiment == 'positive':
        return 'buy'
    elif sentiment == 'negative':
        return 'sell'
    else:
        return 'hold'

# Analyze stock sentiment
def analyze_stock_sentiment(stock_symbol):
    # Fetch news articles
    news_articles = fetch_news(stock_symbol, count=10)

    # Preprocess and analyze sentiment
    data = [preprocess_text(text) for text in news_articles]
    sentiments = [get_sentiment(text) for text in data]

    # Map sentiment to action
    actions = [sentiment_to_action(sentiment) for sentiment in sentiments]
    action_counts = Counter(actions)

    # Check if there are any actions
    if action_counts:
        final_action = action_counts.most_common(1)[0][0]
    else:
        final_action = "hold"  # Default to "hold" if no data is available

    return final_action, action_counts

# Main function
if __name__ == "__main__":
    # Input stock symbol
    stock_symbol = input("Enter the stock symbol (e.g., AAPL): ").strip().upper()

    # Analyze sentiment and get recommendation
    recommendation, counts = analyze_stock_sentiment(stock_symbol)
    print(f"\nRecommendation for {stock_symbol}: {recommendation}")
    print("Action Counts:", counts)