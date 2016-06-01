
# coding: utf-8

# ## Load already prepared data

# In[1]:

TEST_SIZE=0.2


# In[2]:

import pandas as pd


# In[3]:

X = pd.read_csv('../valt_sa_data/x_m.csv')
y = pd.read_csv('../valt_sa_data/y_m.csv', header=None)[0]


# ## Split data into training and test sets

# Let's perform a train/test split with 80% of the data in the training set and 20% of the data in the test set. We use `random_state=0` so that every execution yields the same result.

# In[4]:

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X.as_matrix(),
                                                    y.as_matrix(),
                                                    test_size=TEST_SIZE,
                                                    random_state=0)


# # Train a sentiment classifier with logistic regression
# 
# We will now use logistic regression to create a sentiment classifier on the training data.
# 
# **Note:** This line may take a few minutes.

# In[5]:

from sklearn import linear_model

#logreg = linear_model.LogisticRegression(C=1e5)
logreg = linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')

model = logreg.fit(X_train, y_train)


# # Evaluate the trained model
# 
# We will now use the cross-validation set to evaluate our model.

# In[6]:

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, model.predict(X_test))

print 'Confusion matrix:'
print cm

from sklearn.metrics import classification_report

print 'Classification report:'
print classification_report(y_test, model.predict(X_test))

