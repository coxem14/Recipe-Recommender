from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import joblib


app = Flask(__name__)

# load df
with open('df.pkl', 'rb') as f:
    df = pickle.load(f)

# load vectorizer
count_vec = joblib.load('../models/vec_6_tid_pickle4.joblib')

# load model
lda = joblib.load('../models/lda_model_6_tid_pickle4.joblib')

# define docs
docs = df['cleaned_bow']

# vectorize docs
tf = count_vec.fit_transform(docs)

# get probability matrix
probs = lda.transform(tf)

# define recipes
recipes = df['title']

# define links
links = df['link']

# define indices
idx_arr = np.array(recipes.index)

def get_keyword_idxs(keyword, idx_arr, recipes):
    keyword_recipes = recipes[recipes.str.contains(keyword, case=False, regex=False)]
    keyword_samples = np.random.choice(keyword_recipes.index, size=min(len(keyword_recipes), 50), replace=False)
    keyword_idxs = []
    for sample_idx in keyword_samples:
        keyword_idx = int(np.where(idx_arr == sample_idx)[0])
        keyword_idxs.append(keyword_idx)
    return keyword_idxs


def closest_recipes(keyword, idx_arr, recipes, probs, n_recipes=10):
    keyword_idxs = get_keyword_idxs(keyword, idx_arr, recipes)
    
    d={}
    for idx in keyword_idxs:
        sims = cosine_distances(probs[idx].reshape(1, -1), probs).argsort()[0]
        for sim in sims[1:n_recipes+1]:
            if sim not in d:
                d[sim] = 1
            else:
                d[sim] += 1
                
    rec_idxs = [k for k, v in sorted(d.items(), key=lambda item: item[1])][:-n_recipes:-1]
    recipe_recs = np.array(recipes)[rec_idxs]
    reference_recipes = np.array(recipes)[keyword_idxs]
    
    return recipe_recs, rec_idxs, reference_recipes, keyword_idxs

def get_results_df(recipe_recs, links, rec_idxs):
    rec_links = np.array(links)[rec_idxs]
    results = {'Recipe': recipe_recs, 'Link': rec_links}
    results_df = pd.DataFrame(results)
    return results_df

@app.route('/')
@app.route('/home', methods=['GET','POST'])
def home(title=None):
    return render_template("home.html", title=title)

@app.route('/explore', methods=['GET', 'POST'])
def explore():
    return render_template("explore.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/results', methods=['GET', 'POST'])
def results():
    keyword = request.form['keyword']
    if keyword == '':
        return 'You must enter a keyword.'
    recipe_recs, rec_idxs, reference_recipes, keyword_idxs = closest_recipes(keyword, idx_arr, recipes, probs)
    if len(recipe_recs) < 1:
        return 'Keyword not found.'
    results_df = get_results_df(recipe_recs, links, rec_idxs)
    return render_template("results.html", results_df=results_df)



if __name__=="__main__":
    app.run(debug=True)




























