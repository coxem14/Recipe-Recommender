**********************************************
# Content-Based Recipe Recommender
**********************************************

#### Erin Cox
#### https://github.com/coxem14/Recipe-Recommender
*Last update: 1/14/2021*
***

<p align = 'center'>
    <img src = ''>
</p>

## Table of Contents
  - [Table of Contents](#table-of-contents)
  - [Flask App](#flask-app)
  - [Background](#background)
  - [Data](#data)
  - [EDA](#eda)


## Flask App

#### Home Page
<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-11%20at%2012.38.32%20PM.png'>
</p>

#### Explore Page
<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-11%20at%2012.40.47%20PM.png'>
</p>

#### Results Page
<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-11%20at%2012.41.22%20PM.png'>
</p>

#### Webpage for Recipe
<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-11%20at%2012.42.57%20PM.png'>
</p>

## Background 
I've always loved food, and one of my hobbies is to try to recreate my favorite restaurant dishes at home. Since COVID began, my partner and I haven't been going to restaurants, and we've been cooking a ton. We have expanded *(and exhausted)* our recreated restaurant dishes portfolio. Now we are in the exploratory phase to add new recipes to our cook book.

## Data

I stumbled upon an awesome open source instant search [website](https://recipe-search.typesense.org/) which had over 2 million recipes. I tracked down the original dataset and downloaded it from [RecipeNLG](https://recipenlg.cs.put.poznan.pl/). The dataset was originally created to address challenges with semi-structured text generation.

The dataset contains 2,231,142 recipes. Approximately 1.6M of the recipes were formatted to have better quality (source = Gathered). I used 250K subset of these recipes in my project. The dataset included 6 feature columns containing strings of lists and text.

### Before cleaning:
<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-10%20at%2010.18.47%20AM.png'>
</p>

### The ingredients, directions, and NER columns were strings of lists of strings.

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-10%20at%2010.19.24%20AM.png'>
</p>

### After cleaning:
<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-10%20at%2010.24.39%20PM.png'>
</p>

## EDA

There were a total of 11 different sites that the recipes came from. When creating my subset, I split the data stratified by site to preserve the ratios of the websites in my subset.

### Recipe Website Distribution:
<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-10%20at%2010.55.10%20PM.png'>
</p>

## Topic Modeling with KMeans Clustering

I used Scikit-Learn's KMeans Clustering over a range of k values to get an idea of how many topics were present in the dataset. I found 100 clusters (topics) resulted in the best silhouette scores.

```
vectorizer = TfidfVectorizer(max_df=0.85,
                             min_df=10,
                             ngram_range=(1,3),
                             max_features=1000)

docs_vec = vectorizer.fit_transform(cleaned_bow)
```

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-11%20at%205.09.54%20PM.png'>
</p>

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-11%20at%204.53.17%20PM.png'>
</p>

I looked at the recipe count per cluster and the top words in each cluster to get an idea for how the recipes were distributed.

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-10%20at%209.59.45%20AM.png'>
</p>

I also looked at samples of recipes assigned to each cluster:

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-10%20at%209.50.04%20AM.png'>
</p>

## Topic Modeling with Latent Dirichlet Allocation (LDA)

I used Scikit-Learn's Latent Dirichlet Allocation model as the basis for my recommender. I vectorized the recipe bag-of-words using Scikit-Learn's CountVectorizer(). I experimented with the document frequencies, ngram range, and max features parameters.

```
vec = CountVectorizer(max_df=0.85, 
                      min_df=2,
                      ngram_range=(1,2),
                      max_features=1000)

term_frequency = vec.fit_transform(cleaned_bow)
```

For the LDA model itself, I experimented with the learning offset, document-topic and topic-word priors, and batch size.

```
lda = LatentDirichletAllocation(n_components=100,
                                learning_method='online',
                                learning_offset=10,
                                doc_topic_prior=0.01,
                                topic_word_prior=0.01,
                                batch_size=128)
lda.fit(term_frequency)
```
While I was tuning the parameters of my models and vectorizers, I compared the log-likelihood scores as well as perplexity to see if I was making improvements. I wanted to maximize the log-likelihood, while minimizing perplexity.

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-13%20at%209.12.54%20PM.png'>
</p>

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-13%20at%209.13.04%20PM.png'>
</p>

However, recent studies have shown that predictive likelihood (or equivalently, perplexity) and human judgment are often not correlated, and even sometimes slightly anti-correlated. [source](https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d)
I definitely found this to be the case when I was analyzing the recommendations of my models.

In this example, the model with the highest perplexity, actually performed better when recommending recipes similar to the reference recipe.

<p align = 'center'>
    <img src = 'https://github.com/coxem14/Recipe-Recommender/blob/main/images/Capstone_3_ScreenShots/Screen%20Shot%202021-01-13%20at%209.17.50%20PM.png'>
</p>

In the end, I chose this model and vectorizer combination because I felt it did the best job.
