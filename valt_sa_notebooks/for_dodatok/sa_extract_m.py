
# coding: utf-8

# # Text data feature extraction

# Here are parameters of the program user can easily change and estimate their impact on the performance.

# In[109]:

USE_MY_METHOD = True
USE_STOP_WORDS = False
USE_EMOTICONS = False
USE_NEGATION = True

# If set to false, number of occurrences of words is calculated
USE_BOOLEAN_REPRESENTATION = True

NUMBER_OF_REVIEWS_TO_ANALYZE = 10000
NUMBER_OF_POPULAR_WORDS_TO_USE = 1000


# We will use a dataset consisting of baby product reviews on Amazon.com.

# In[110]:

import pandas as pd


# In[111]:

products_raw = pd.read_csv("../valt_sa_data/amazon_baby.csv")
products = products_raw[['review', 'rating']][0:NUMBER_OF_REVIEWS_TO_ANALYZE]


# Let us see how the data looks like:

# In[112]:

products


# Let us explore a specific example of a baby product.

# In[113]:

products.iloc[9]


# Let us define an emoticons extraction function.

# In[114]:

emoticons = [
    ':)', ':))', ':)))', ':(', ':((',
    ':(((', '=)', '=(', '=))', '=(('
]

def extract_emoticons(text):
    emoticons_in_text = []
    for emoticon in emoticons:
        i = text.find(emoticon)
        if i > -1:
            emoticons_in_text.append(emoticon)
    return emoticons_in_text


# The helper functions below are also useful.

# In[115]:

punctuation_to_remove = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~'

def remove_punctuation(text):
    return text.translate(None, punctuation_to_remove)

pos_dict = {
    'NN': 'n', 'VB': 'v', 'VBD': 'v', 'VBG': 'v', 
    'VBN': 'v', 'VBP': 'v', 'VBZ': 'v',
    'JJ': 'a', 'JJR': 'a', 'JJS': 'a', 'JJT': 'a'
}

def get_pos_for_lemmatirzer(brown_post):
    if not brown_post in pos_dict:
        return 'n'
    else:
        return pos_dict[brown_post]


# Now let us define a more sophisticated function for review analysis.
# 
# First the punctuation is removed.
# 
# Then every word is pos tagged to prepare for lemmatization.
# 
# After that lemmatization is performed to find the root form of each word.
# 
# All the stop words are removed if the corresponding program parameter is set.
# Also if set, emoticons are extracted and processed.

# In[116]:

import nltk
from nltk.stem import WordNetLemmatizer

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 
 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 
 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 
 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 
 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 
 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 
 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 
 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 
 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
 'nor', 'not', "n't", 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
 'will', 'just', 'don', 'should', 'now']

def analyze_review(text):
    if USE_EMOTICONS:
        emoticons_features = extract_emoticons(text)
    else:
        emoticons_features = []

    text_without_punctuation = remove_punctuation(text)
    tokens = nltk.word_tokenize(text_without_punctuation)
    tagged_tokens = nltk.pos_tag(tokens)
    tokens_prepared_for_lemmatization = [(t[0], get_pos_for_lemmatirzer(t[1])) for t in tagged_tokens]
    
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    
    not_count = 0
    words_after_not_count = 0
    for tpl in tokens_prepared_for_lemmatization:
        current_word = lemmatizer.lemmatize(tpl[0], tpl[1]).lower()
        if words_after_not_count > 2:
            not_count = 0
            words_after_not_count = 0
        if current_word == 'not' or current_word == "n't":
                not_count += 1
        elif (not USE_STOP_WORDS) or (not current_word in stop_words):
            if USE_NEGATION and not_count % 2 == 1:
                current_word = 'NOT_' + current_word
                words_after_not_count += 1
            lemmas.append('F_' + current_word) # F - meaning feature
    
    review_words = lemmas + emoticons_features
    return review_words


# Now, we will perform text analysis.
# We will also find and print most common words and total number of words in the dictionary.

# In[117]:

analyzed_reviews = products['review'].apply(str).apply(analyze_review)

review_words_list = [] # conaints duplicates, so that count of each word can be calculated
review_dictionary = set()

for w_l in analyzed_reviews:
    for word in w_l:
        review_words_list.append(word)
        review_dictionary.add(word)

from collections import Counter

review_counter = Counter(review_words_list)
most_common_words = map(lambda x: x[0], review_counter.most_common(NUMBER_OF_POPULAR_WORDS_TO_USE))
print most_common_words
print len(review_dictionary)


# We perform feature extraction on the analyzed text. The matrix for machine learning is formed. Only features stored in the variable `significant_words` are included.

# In[118]:

if USE_MY_METHOD:
    if USE_EMOTICONS:
        significant_words = most_common_words + emoticons
    else:
        significant_words = most_common_words
else:
    significant_words = ['love', 'great', 'easy', 'old', 'little', 'perfect', 'loves', 
                         'well', 'able', 'car', 'broke', 'less', 'even', 'waste', 'disappointed', 
                         'work', 'product', 'money', 'would', 'return']
        
def count_number_of_significant_words(text):
    words = text['review']
    word_dict = {}
    for word in significant_words:
        word_dict[word] = 0
    for word in words:
        if word in significant_words:
            if not word in word_dict:
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

word_counts_df = pd.DataFrame(analyzed_reviews).apply(count_number_of_significant_words, axis=1)
word_counts_df.columns = significant_words

products_with_words = products.join(word_counts_df)


# Now, let us explore what the sample looks like after all the transformations.
# 
# The resulting matrix is very sparse, as was expected.

# In[119]:

products_with_words.iloc[9]


# ## Save prepared data into a file

# In[120]:

X = products_with_words[significant_words]
y = products_with_words['rating']
X.to_csv('../valt_sa_data/x_m.csv', index=False)
y.to_csv('../valt_sa_data/y_m.csv', index=False)

