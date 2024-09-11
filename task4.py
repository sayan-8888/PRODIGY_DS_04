import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Download the dataset from Kaggle (replace with your download path)
data_path = "task4.csv"  # Replace with your downloaded file path
df = pd.read_csv(data_path)

def sentiment(text):
  # Create TextBlob object
  blob = TextBlob(text)
  # Calculate sentiment polarity (negative: -1, positive: 1)
  return blob.sentiment.polarity

df['sentiment'] = df['text'].apply(sentiment)

# Analyze sentiment distribution across all airlines
plt.hist(df['sentiment'])
plt.xlabel('Sentiment Polarity')
plt.ylabel('Number of Tweets')
plt.title('Sentiment Distribution of Airline Tweets')
plt.show()

# Analyze sentiment distribution by airline
df_grouped = df.groupby('airline')['sentiment'].mean()
df_grouped.plot(kind='bar')
plt.xlabel('Airline')
plt.ylabel('Average Sentiment Polarity')
plt.title('Average Sentiment by Airline')
plt.show()

def categorize_sentiment(polarity):
  if polarity > 0:
    return 'positive'
  elif polarity < 0:
    return 'negative'
  else:
    return 'neutral'

df['sentiment_category'] = df['sentiment'].apply(categorize_sentiment)

# Analyze sentiment distribution by category
df['sentiment_category'].value_counts().plot(kind='bar')
plt.xlabel('Sentiment Category')
plt.ylabel('Number of Tweets')
plt.title('Sentiment Distribution by Category')
plt.show()

keywords = ['good', 'great', 'excellent', 'bad', 'terrible', 'worst']

# Create a new column for keyword-based sentiment
df['keyword_sentiment'] = df['text'].apply(lambda text: any(keyword in text.lower() for keyword in keywords))

# Analyze sentiment based on keywords
df.groupby('keyword_sentiment')['sentiment'].mean().plot(kind='bar')
plt.xlabel('Keyword-Based Sentiment')
plt.ylabel('Average Sentiment Polarity')
plt.title('Average Sentiment by Keyword')
plt.show()

correlation = df['sentiment'].corr(df['retweet_count'])
print("Correlation between sentiment and retweet count:", correlation)

# Visualize the correlation
plt.scatter(df['sentiment'], df['retweet_count'])
plt.xlabel('Sentiment Polarity')
plt.ylabel('Retweet Count')
plt.title('Correlation between Sentiment and Retweet Count')
plt.show()