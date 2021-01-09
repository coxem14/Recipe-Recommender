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

# define links and convert to urls
links = df['link']

def parse_links(link):
    if link.startswith('http://') or link.startswith('https://'):
        return link
    else:
        url = 'http://' + link
        return url
    
def series_link_to_url(links):
    urls = links.apply(parse_links)
    return urls

urls = series_link_to_url(links)

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

def get_results_df(recipe_recs, urls, rec_idxs):
    rec_urls = np.array(urls)[rec_idxs]
    results = {'Recipe': recipe_recs, 'Link': rec_urls}
    results_df = pd.DataFrame(results)
    return results_df

def make_title_from_keyword(keyword):
    keyword = keyword.lower()
    if keyword in title_d:
        return title_d[keyword]
    else:
        title = np.random.choice(generic_titles, size=1)[0]
        return title.format(keyword)

title_d = {'guacamole': 'Holy Guacamole!',
           'apple': 'Are any of these the apple of your eye?',
           'apple pie': 'Apple pie of your eye?',
           'buns': 'Want buns in the oven?',
           'bun': 'Want a bun in the oven?',
           'cinnamon bun': 'Want a cinnamon bun in the oven?',
           'cheese': 'Sweet dreams are made of cheese...',
           'cheesy': "A little cheesy, but still grate:",
           'bread': 'Best thing since sliced bread:',
           'bacon': 'Bring home the bacon!',
           'butter': 'You butter believe it!',
           'cake': 'You can have your cake and eat it, too.',
           'chocolate cake': 'You can have your chocolate cake and eat it, too.',
           'tea': 'Any of these your cup of tea?',
           'enchilada': 'Recipes, links, the whole enchilada...',
           'enchiladas': 'Recipes, links, the whole enchilada...',
           'egg': 'Check out these egg-cellent recipes:',
           'eggs': 'Check out these eggs-quisite recipes:',
           'gravy': 'All aboard the gravy train!',
           'spice': 'Spice things up!',
           'pie': 'Finding new recipes is easy as pie!',
           'coffee': 'Wake up and smell the coffee!',
           'potatoes': 'Po-tay-toes! Boil em, mash em, stick em in a stew...',
           'omelet': "You can't make an omelet without breaking some eggs...",
           'omelette': "You can't make an omelette without breaking some eggs...",
           'pizza': 'You wanna pizza these?',
           'pea': "Don't worry, be hap-pea",
           'thyme': 'Well, well, well. Would you look at the thyme?',
           'noodle': 'Here are some recipes to noodle over:'}

generic_titles = ['Curious about "{}"? Maybe these will hit the spot:',
                  "Curious about '{}'? Here's some food for thought:",
                  'Curious about "{}"? Dig into these:',
                  'Curious about "{}"? Chew on these:']

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
    results_df = get_results_df(recipe_recs, urls, rec_idxs)
    title = make_title_from_keyword(keyword)
    return render_template("results.html", results_df=results_df, title=title)



if __name__=="__main__":
    app.run(debug=True)




























