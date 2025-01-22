## Import required packages
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

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

# Count the tokens - returns a list of tuples with tokens and their counts
token_counts=Counter(tokens)

# sort tuples to put most frequent first
sorted_token_counts=sorted(token_counts.items(), key=lambda item: item[1], reverse=True)
# select just tokens
sorted_tokens=list(zip(*sorted_token_counts))[0]

## ENCODE TEXT 
# Select the first 5000 words
tokens_5000 = sorted_tokens[0:5000]

# Create a 10000 x 5000 matrix of zeros for one-hot encoding
one_hot_matrix = np.zeros((len(reviews), len(tokens_5000)))
#iterate over the reviews
for i, rev in enumerate(reviews):
    # Tokenise the current review
    tokens = token_definition.findall(rev)
    # iterate over the set of 5000 words
    for word,t in enumerate(tokens_5000):
        # if the current word j occurs in the current review i then set the matrix element at i,j to be one. Otherwise leave as zero.
        if t in tokens:
             one_hot_matrix[i,word] = 1

train_set=np.random.choice(len(reviews),int(len(reviews)*0.8),replace=False)
test_set=list(set(range(0,len(reviews))) - set(train_set))

train_matrix = one_hot_matrix[train_set,]
test_matrix = one_hot_matrix[test_set,]

sentiment_labels_train = [sentiment_ratings[i] for i in train_set]
sentiment_labels_test = [sentiment_ratings[i] for i in test_set]

## FIT A LOGISTIC REGRESSION MODEL
num_features=5000
y=[int(l == "positive") for l in sentiment_labels_train]
weights = np.random.rand(num_features)
bias=np.random.rand(1)
n_iters = 2500
lr=0.1
logistic_loss=[]
num_samples=len(y)
for i in range(n_iters):
  z = train_matrix.dot(weights)+bias
  q = 1/(1+np.exp(-z))
  eps=0.00001
  loss = -sum((y*np.log2(q+eps)+(np.ones(len(y))-y)*np.log2(np.ones(len(y))-q+eps)))
  logistic_loss.append(loss)

  dw = ((q-y).dot(train_matrix) * (1/num_samples))
  db = sum((q-y))/num_samples
  weights = weights - lr*dw
  bias = bias - lr*db

plt.plot(range(1,n_iters),logistic_loss[1:])
plt.xlabel("number of epochs")
plt.ylabel("loss")

# calculate predicted class - a vector of predicted values
z = test_matrix.dot(weights)+bias
# turn z into probability
q = 1/(1+np.exp(-z))
#create an empty list for predicted labels
y_test_pred = []

# iterate over probability score q and add labels 1 and 0 to the labels list
#for i in q:
#  if i > 0.5:
#    y_test_pred.append(1)
#  else:
#    y_test_pred.append(0)

# could be done with list comprehension as well
y_test_pred = [int(i>0.5) for i in q]

print(y_test_pred)

# Calculate accuracy
y_test=[int(l == "positive") for l in sentiment_labels_test] # int returns 1 or 0
acc_test=[int(yp == y_test[s]) for s,yp in enumerate(y_test_pred)] # compares same index in both lists (gold vs test labels)
print(sum(acc_test)/len(acc_test))

# precision , recall
labels_test_pred=["positive" if s == 1 else "negative" for s in y_test_pred]
true_positives= sum([int(yp == "positive" and sentiment_labels_test[s] == "positive") for s,yp in enumerate(labels_test_pred)])
false_positives = sum([int(yp=="positive" and sentiment_labels_test[s] == "negative") for s,yp in enumerate(labels_test_pred)])
false_negatives = sum([int(yp=="negative" and sentiment_labels_test[s] == "positive") for s,yp in enumerate(labels_test_pred)])

prec_test = true_positives/(true_positives + false_positives)
print(prec_test)

recal_test = true_positives/(true_positives + false_negatives)
print(recal_test)