
# coding: utf-8

# # Text data analysis with Knowledge-based system

# # Data preperation
# 
# We will use a dataset consisting of baby product reviews on Amazon.com.

# In[1]:

NUMBER_OF_REVIEWS_TO_ANALYZE = 100000


# In[2]:

import pandas as pd


# In[3]:

products = pd.read_csv("../valt_sa_data/amazon_baby.csv")[['review', 'rating']]


# In[4]:

products = products[0:NUMBER_OF_REVIEWS_TO_ANALYZE]


# In[5]:

products


# ## Build the word count vector for each review
# 
# Let us explore a specific example of a baby product.

# In[6]:

products.iloc[9]


# Now, we will perform 2 simple data transformations:
# 
# 1. Remove punctuation using Python's built-in string functionality.
# 2. Transform the reviews into word-counts of positive and negative word.
# 3. Finally made prediction based on positive/negative words ratio.

# In[7]:

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

review_without_puctuation = products['review'].apply(str).apply(remove_punctuation)


# In[8]:

significant_words = pd.read_csv('../valt_sa_data/positive-negative-words.csv', header=None)[0].tolist()

positive_words = pd.read_csv('../valt_sa_data/positive-words.csv', header=None)[0].tolist()
negative_words = pd.read_csv('../valt_sa_data/negative-words.csv', header=None)[0].tolist()
        
def count_number_of_significant_words(text):
    prediction = 3
    words = text['review'].split()
    word_dict = {}
    for word in significant_words:
        word_dict[word] = 0
    for word in words:
        if word in significant_words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] = word_dict[word] + 1
    positive = 0
    negative = 0
    for positive_word in positive_words:
        if positive_word in word_dict:
            positive += word_dict[positive_word]
    
    for negative_word in negative_words:
        if negative_word in word_dict:
            negative += word_dict[negative_word]
                        
    n = positive + negative
    if n > 0:
        prediction = 1 + int(round(float(positive) / n * 4))
     
    return pd.Series(prediction)

predictions_df = pd.DataFrame(review_without_puctuation).apply(count_number_of_significant_words, axis=1)
predictions_df.columns = ['prediction']

products_with_words = products.join(predictions_df)


# Now, let us see what rating and predictions look like.

# In[9]:

products_with_words


# ## Evaluate the model

# In[10]:

y_true = products_with_words['rating']
y_predicted = products_with_words['prediction']

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_predicted)

print 'Confusion matrix:'
print cm

from sklearn.metrics import classification_report

print 'Classification report:'
print classification_report(y_true, y_predicted)

