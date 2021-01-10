import pandas as pd
import numpy as np

from ast import literal_eval

from urllib.parse import urlparse

import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer as PS

import pickle

# link cleaning, site and url creation
def get_site(link):
    path = urlparse(link).path
    site = path.split('/')[0]
    return site

def df_link_to_site(df, column_name, new_column_name):
    df[new_column_name] = df[column_name].apply(get_site)
    return df[new_column_name]

def convert_link_to_url(link):
    if link.startswith('http://') or link.startswith('https://'):
        return link
    else:
        url = 'http://' + link
        return url
    
def df_links_to_urls(links):
    urls = links.apply(convert_link_to_url)
    return urls


# string of list of strings columns cleaning
def literal_return(val):
    try:
        return literal_eval(val)
    except (ValueError, SyntaxError) as e:
        return val

def df_str_to_literal(df, column_name):
    df[column_name] = df[column_name].apply(literal_return)
    return df[column_name]

def clean_df(df, columns_list):
    for col in columns_list:
        df[col] = df_str_to_literal(df, col)
    return None


# bag of words
def make_bag_of_words(df, columns_list):
    df['bag_of_words'] = ''
    for col in columns_list:
        if col == 'title':
            df['bag_of_words'] += df[col] + ' '
        if col == 'ingredients':
            df['bag_of_words'] += df[col].apply(' '.join) + ' '
        if col == 'directions':
            df['bag_of_words'] += df[col].apply(' '.join)
    return df['bag_of_words']


# clean documents (bag of words)
def clean_document(document):
    '''
    Takes in a string.
    Returns cleaned string.
    '''
    # lowercase the strings
    doc_lower = document.lower() 

    #tokenize
    tokens = word_tokenize(doc_lower) 
    
    # remove punctuation
    punc = set(string.punctuation)
    tokens_no_punc = [word for word in tokens if word not in punc]
   
    # remove stopwords
    s_words = set(stopwords.words('english'))
    s_words_list = ['tablespoon', 'tbsp', 'teaspoon', 'tsp', 'cup', 'oz', 'lb', 'c.']
    for word in s_words_list:
        s_words.add(word)
    tokens_no_sw = [word for word in tokens_no_punc if word not in s_words]
    
    # stem the words to get rid of multiple forms of the same word
    porter = PS()
    tokens_stemmed = [porter.stem(word) for word in tokens_no_sw]
    
    # join all words into one string
    cleaned_doc = ' '.join(tokens_stemmed)
    
    return cleaned_doc


if __name__=="__main__":

# read in data, select only gathered source
data = pd.read_csv('../dataset/full_dataset.csv')
data = data.loc[data['source'] == 'Gathered']
data.drop(columns='source', inplace=True)

# convert link to site and urls columns, drop link column
data['site'] = df_link_to_site(data, 'link', 'site')
data['urls'] = df_links_to_urls(data['link'])
data.drop(columns='link', inplace=True)

# clean up columns that are strings of lists of strings
clean_df(data, ['ingredients', 'directions', 'NER'])

# make bag of words column, and drop ingredients and directions
columns_list = ['title', 'ingredients', 'directions']
data['bag_of_words'] = make_bag_of_words(data, columns_list)
data.drop(columns=['ingredients', 'directions'], inplace=True)

# clean bag of words
data['cleaned_bow'] = data['bag_of_words'].apply(clean_document)

# pickle dataframe
# data.to_pickle('./dataframes/full_data_clean_df_pickle4.pkl', protocol=4)
