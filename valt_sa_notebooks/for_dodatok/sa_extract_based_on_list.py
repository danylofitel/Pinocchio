
# coding: utf-8

# # Text data feature extraction

# Here are parameters of the program user can easily change and estimate their impact on the performance.

# In[1]:

# If set to false, number of occurances of words is calculated
USE_BOOLEAN_REPRESENTATION = True
NUMBER_OF_REVIEWS_TO_ANALYZE = 100000


# # Data preperation
# 
# We will use a dataset consisting of baby product reviews on Amazon.com.

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
# 2. Transform the reviews into word-counts.

# In[7]:

def remove_punctuation(text):
    import string
    return text.translate(None, string.punctuation) 

review_without_puctuation = products['review'].apply(str).apply(remove_punctuation)


# In[8]:

significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
                         'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
                         'work', 'product', 'money', 'would', 'return']

def count_number_of_significant_words(text):
    words = [word.lower() for word in text['review'].split()]
    word_dict = {}
    for word in significant_words:
        word_dict[word] = 0
    for word in words:
        if word in significant_words:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                if USE_BOOLEAN_REPRESENTATION:
                    word_dict[word] = 1
                else:
                    word_dict[word] = word_dict[word] + 1
    significant_words_counts = []
    for word in significant_words:
        significant_words_counts.append(word_dict[word]) 
    return pd.Series(significant_words_counts, index=significant_words)

lambdafunc = lambda x: pd.Series(significant_words)

word_counts_df = pd.DataFrame(review_without_puctuation).apply(count_number_of_significant_words, axis=1)
word_counts_df.columns = significant_words

products_with_words = products.join(word_counts_df)


# Now, let us explore what the sample example above looks like after these 2 transformations.

# In[9]:

products_with_words.iloc[9]


# ## Save prepared data into a file

# In[10]:

X = products_with_words[significant_words]
y = products_with_words['rating']
X.to_csv('../valt_sa_data/x_m.csv', index=False)
y.to_csv('../valt_sa_data/y_m.csv', index=False)

