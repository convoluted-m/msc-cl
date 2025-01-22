## Import required packages
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# Download off-the-shelf w2v embeddings
import gensim.downloader as api
w = api.load('word2vec-google-news-300') # off the shelf embeddings
vocab=[x for x in w.key_to_index.keys()] # a list of words from the pre-trained embeddings

# Import data & create separate lists for reviews and labels
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

### EXPLORE THE DATA

## Explore sentiment ratings
#Count sentiment with Counter()
sentiment_ratings_counter = Counter(sentiment_ratings)
#print(sentiment_ratings_counter)

# # Plot positive vs negative sentiment using matplotlib
# plt.hist(sentiment_ratings, bins=2, align='left', rwidth=0.8)
# plt.xlabel('Sentiment')
# plt.ylabel('Count')
# plt.xticks([0, 0.5], ['Positive', 'Negative'])
# #plt.show()

# ## NOT needed?
# # Categorise sentiments
# categorised_sentiment = []
# for sentiment in sentiment_ratings:
#   if sentiment == 'positive':
#     categorised_sentiment.append('Positive')
#   else:
#     categorised_sentiment.append('Negative')
# #print(len(categorised_sentiment))
# # Count positive vs negative sentiments
# positive_count = categorised_sentiment.count("Positive")
# #print(f"Positive count: {positive_count}")
# negative_count = categorised_sentiment.count("Negative")
# #print(f"Negative count: {negative_count}")

# # Plot positive vs negative sentiment using matplotlib
# plt.figure(figsize=(5, 4))
# plt.hist(categorised_sentiment, bins=2, align='left', rwidth=0.8)
# plt.xlabel('Sentiment')
# plt.ylabel('Count')
# plt.xticks([0, 0.5], ['Positive', 'Negative'])
# #plt.show()

# ## Explore product types
# # Get count per profuct type - returns a dict
# product_type_count = Counter(product_types)
# product_type_count

# # Plot product types with counts
# plt.figure(figsize=(10, 8))
# product_type_count = Counter(product_types)
# plt.barh(list(product_type_count.keys()), list(product_type_count.values()))
# plt.xlabel('Count')
# plt.ylabel('Product Type')
# #plt.show()

# ## Explore review Helpfulness
# helpfulness_ratings_count = Counter(helpfulness_ratings)
# helpfulness_ratings_count

# plt.hist(helpfulness_ratings, bins=3, align='left', rwidth=0.8)
# plt.xlabel('Helpfulness')
# plt.ylabel('Count')
# plt.xticks([0, 0.65, 1.35], ['neutral', 'helpful', 'unhelpful'])
# #plt.show()

###  SENTIMENT CLASSIFIER

## PREPARE THE DATA

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
#token_counts=Counter(tokens)

# create a unique set of tokens from the Amazon dataset
types=set(tokens)

## ENCODE TEXT using embeddings
# Map my tokens(types) to embeddings (vocab)
indices=[vocab.index(x) for x in types if x in vocab]
# filter tokens that are both in my dataset and the pre-trained embeddings vocab
types_inc=[x for x in types if x in vocab] 

# Create an embeddings matrix for the included tokens
M=w[indices]
#print(M.shape)

# Create embeddings for my classification
# create an empty list to store embeddings
embeddings=[]

# For each review, tokenise it, create a vector of size 300
for i, review in enumerate(reviews):
    tokens = re.findall("[^ ]+",review)
    this_vec = np.zeros((1, 300))
    #for each token in a review, if token in the types, add its embedding to the vector & append vector to vectors' list
    for token in tokens:
        if token in types_inc:
            this_vec = this_vec + M[types_inc.index(t)]
    embeddings.append(this_vec)
    
# convert the list into an array and squeeze to remove extra dimensions    
embeddings=np.array(embeddings).squeeze()

## SPLIT THE DATA
# Split the data into train, test and dev sets
train_set=np.random.choice(len(embeddings),int(len(embeddings)*0.8),replace=False)
remaining_set=list(set(range(0,len(embeddings))) - set(train_set))
test_set=np.random.choice(len(remaining_set),int(len(remaining_set)*0.5),replace=False)
dev_set=list(set(range(0,len(remaining_set))) - set(test_set))

# Prepare train, test, dev embeddings
M_train_emb = embeddings[train_set,]
M_test_emb = embeddings[test_set,]
M_test_emb = embeddings[dev_set,]

# Prepare train, test, dev labels
labels_train = [sentiment_ratings[i] for i in train_set]
labels_test = [sentiment_ratings[i] for i in test_set]
labels_dev = [sentiment_ratings[i] for i in dev_set]

# training labels
labels_train = [sentiment_ratings[i] for i in train_set]
# test labels
labels_test = [sentiment_ratings[i] for i in test_set]

# Prepare batches for training
batches = 10
# Create array of all indices in training data
a=np.arange(M_train_emb.shape[0])
# randomly shuffle indices to prevent uneven class distrubution
np.random.shuffle(a)
# Split indices into k equal batches
batches=np.array(np.split(a, batches))

batches.shape

## FIT A LOGISTIC REGRESSION MODEL
num_features=300
weights = np.random.rand(num_features)
bias=np.random.rand(1)

# Train - change code for batches
n_iters = 2500
lr=0.005
logistic_loss=[]
num_samples=len(y)

for i in range(n_iters):
    for k in range(batches.shape):
        