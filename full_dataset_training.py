import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split, RandomizedSearchCV

import pickle
import joblib

from pprint import pprint

if __name__=="__main__":

# load df
with open('./models/training_df_pickle_5.pkl', 'rb') as f:
    bow_data = pickle.load(f)

# define documents
docs_cleaned = bow_data['cleaned_bow']

# count vectorizer
num_features = 1000
ngram_range=(1,3)

vec = CountVectorizer(max_df=0.85, 
                      min_df=10,
                      ngram_range=ngram_range,
                      max_features=num_features)

tf = vec.fit_transform(docs_cleaned)

# random grid search parameters
num_topics = [100]
learning_method=['online']
learning_offset = [10, 50, 90]
doc_topic_prior = [None, 0.1, 0.9]
topic_word_prior = [None, 0.1, 0.9]
learning_decay = [0.5, 0.7, 0.9]
batch_size = [64, 128]
n_jobs= [-1]

random_grid = {'n_components': num_topics,
               'learning_method':learning_method,
               'learning_offset': learning_offset,
               'doc_topic_prior': doc_topic_prior,
               'topic_word_prior': topic_word_prior,
               'learning_decay': learning_decay,
               'batch_size': batch_size,
               'n_jobs': n_jobs}

pprint(random_grid)

# train, test, split
tf_train, tf_test = train_test_split(tf, test_size=0.25)

# model training
lda = LatentDirichletAllocation()
lda_grid = RandomizedSearchCV(estimator=lda, 
                              param_distributions=random_grid,
                              cv=5,
                              n_iter=10,
                              n_jobs=-1, 
                              verbose=1)

lda_grid.fit(tf_train)

# getting best model
best_lda_model = lda_grid.best_estimator_

# persisting best model
joblib.dump(best_lda_model, './models/lda_model_full_tid_pickle4.joblib', protocol=4)
joblib.dump(best_lda_model, './models/lda_model_full_tid_pickle5.joblib', protocol=5)
joblib.dump(vec, './models/vec_full_tid_pickle4.joblib', protocol=4)
joblib.dump(vec, './models/vec_full_tid_pickle5.joblib', protocol=5)

# printing scores on test set
print(pd.DataFrame.from_dict(lda_grid.cv_results_))
print('Test Score:', lda_grid.score(tf_test))
print('Perplexity:', lda_grid.perplexity(tf_test))

# printing best parameters and scores
print("Best Model's Params: ", lda_grid.best_params_)
print("Best Model's Log Likelihood Score: ", lda_grid.best_score_)
print("Best Model's Perplexity: ", best_lda_model.perplexity(tf))