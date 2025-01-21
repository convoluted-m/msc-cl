# Import required packages
import re
from collections import Counter
import matplotlib.pyplot as plt

# Import data & Put into separate lists
reviews=[]
sentiment_ratings=[]
product_types=[]
helpfulness_ratings=[]

with open("Compiled_Reviews.txt") as f:
   for line in f.readlines()[1:]:
        fields = line.rstrip().split('\t')
        reviews.append(fields[0])
        sentiment_ratings.append(fields[1])
        product_types.append(fields[2])
        helpfulness_ratings.append(fields[3])

## Tokenise the text
# Define a token (word, space-based)
token_definition = re.compile("[^ ]+")
# Tokenise the reviews
tokenised_reviews = [token_definition.findall(txt) for txt in reviews]

# Put all tokens into one list
tokens=[]
for token in tokenised_reviews:
      tokens.extend(token)

# A list of tuples with tokens and their counts
token_counts=Counter(tokens)

### Explore the data 

## Explore sentiment ratings

# Count sentiment with Counter()
sentiment_ratings_counter = Counter(sentiment_ratings)
print(sentiment_ratings_counter)

# Plot positive vs negative sentiment using matplotlib
plt.hist(sentiment_ratings, bins=2, align='left', rwidth=0.8)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks([0, 0.5], ['Positive', 'Negative'])
plt.show()

# NOT needed?
# Categorise sentiments
categorised_sentiment = []
for sentiment in sentiment_ratings:
  if sentiment == 'positive':
    categorised_sentiment.append('Positive')
  else:
    categorised_sentiment.append('Negative')

print(len(categorised_sentiment))

# Count positive vs negative sentiments
positive_count = categorised_sentiment.count("Positive")
print(f"Positive count: {positive_count}")
negative_count = categorised_sentiment.count("Negative")
print(f"Negative count: {negative_count}")

# Plot positive vs negative sentiment using matplotlib
plt.figure(figsize=(5, 4))
plt.hist(categorised_sentiment, bins=2, align='left', rwidth=0.8)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks([0, 0.5], ['Positive', 'Negative'])
plt.show()

## Explore product types
# Get count per profuct type - returns a dict
product_type_count = Counter(product_types)
product_type_count

# Plot product types with counts
plt.figure(figsize=(10, 8))
product_type_count = Counter(product_types)
plt.barh(list(product_type_count.keys()), list(product_type_count.values()))
plt.xlabel('Count')
plt.ylabel('Product Type')
plt.show()

## Explore review Helpfulness
helpfulness_ratings_count = Counter(helpfulness_ratings)
helpfulness_ratings_count

plt.hist(helpfulness_ratings, bins=3, align='left', rwidth=0.8)
plt.title('Helpfulness Distribution')
plt.xlabel('Helpfulness')
plt.ylabel('Count')
plt.xticks([0, 0.65, 1.35], ['neutral', 'helpful', 'unhelpful'])
plt.show()