#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from sqlalchemy import create_engine

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics  import precision_score
from sklearn.metrics import classification_report


# In[2]:


# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table(table_name='gal_project_df',con=engine)
X = df["message"]
Y = df.iloc[:, 4:]


# ### 2. Write a tokenization function to process your text data

# In[3]:


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[4]:


pipeline_random_forest = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, Y)
pipeline_random_forest.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[6]:


# in the next code i take reference from -https://github.com/Mcamin/Disaster-Response-Pipeline/blob/master/ML%20Pipeline%20Preparation.ipynb

def plot_scores(y_test, y_pred):
    #Testing the model
    # Printing the classification report for each label
    i = 0
    for col in y_test:
        print('Feature {}: {}'.format(i+1, col))
        print(classification_report(y_test[col], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))
    
y_pred = pipeline_random_forest.predict(X_test)
plot_scores(y_test, y_pred)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

## Try other ML alogrothm
#KNeighborsClassifier
pipeline_k_nearest = Pipeline([
   ('vect',CountVectorizer(tokenizer=tokenize)),
   ('tfidf', TfidfTransformer()),
   ('clf', MultiOutputClassifier(KNeighborsClassifier()))
])
pipeline_k_nearest.fit(X_train, y_train)
y_pred = pipeline_k_nearest.predict(X_test)
plot_scores(y_test, y_pred)


# In[ ]:


from sklearn.naive_bayes import MultinomialNB

## Try other ML alogrothm
#KNeighborsClassifier
pipeline_naive = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(MultinomialNB()))
])
pipeline_naive.fit(X_train, y_train)
y_pred = pipeline_naive.predict(X_test)
plot_scores(y_test, y_pred)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[ ]:


##Meanwhile the best algorithm  is random forest


# In[ ]:


parameters = {
    'clf__estimator__bootstrap': [True],
    'clf__estimator__max_depth': [10,20, None]
}
#  'max_features': ['auto', 'sqrt'],
#  'min_samples_leaf': [1, 2, 4],
#  'min_samples_split': [2, 5, 10],
#  'n_estimators': [200, 1000,  2000]}

cv = GridSearchCV(pipeline_random_forest,param_grid=parameters)
cv.fit(X_train, y_train)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[ ]:


y_pred = cv.predict(X_test)
plot_scores(y_test, y_pred)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[ ]:


## I tried 3 ML algorithms


# ### 9. Export your model as a pickle file

# In[ ]:


# Create a pickle file for the model
import pickle
file_name = 'Gal_project_model.pkl'
pickle.dump(pipeline_random_forest, open(file_name, 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




