from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


app = Flask(__name__)







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
    else:
        return render_template("results.html", keyword=keyword)

    # filename = 'lda.pkl'
    # model = pickle.load(open(filename, 'rb'))





if __name__=="__main__":
    app.run(debug=True)




























