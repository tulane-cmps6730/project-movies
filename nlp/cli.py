# -*- coding: utf-8 -*-

"""Demonstrating a very simple NLP project. Yours should be more exciting than this."""
import click
import glob
import pickle
import sys

import numpy as np
import pandas as pd
import re
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

from . import clf_path, config

@click.group()
def main(args=None):
    """Console script for nlp."""
    return 0

@main.command('web')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(port):
    """
    Launch the flask web app for the Movie Recommendation System.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)
    
@main.command('dl-data')
def dl_data():
    """
    Download training/testing data.
    """
    data_url = config.get('data', 'url')
    data_file = config.get('data', 'file')
    print('downloading from %s to %s' % (data_url, data_file))
    r = requests.get(data_url)
    with open(data_file, 'wt') as f:
        f.write(r.text)
    

def data2df():
    return pd.read_csv(config.get('data', 'file'))

@main.command('stats')
def stats():
    """
    Read the data files and print interesting statistics.
    """
    df = data2df()

    '''
    Editing to show the number of movies and average duration

    Step 1: Convert duration strings to total minutes
    Split the 'duration' string on space (' '), which gives us ['2h', '9m'] for '2h 9m'
    Then we convert hours to minutes and add up to the minutes part
    '''
    def duration_to_minutes(duration):
        parts = duration.split(' ')
        minutes = 0
        for part in parts:
            if 'h' in part:
                hours = int(part.replace('h', ''))
                minutes += hours * 60
            elif 'm' in part:
                minutes += int(part.replace('m', ''))
        return minutes

    df['time_in_min'] = df['duration'].apply(duration_to_minutes)

    '''
    Step 2: Calculate the average duration in minutes
    '''
    
    avg_duration = df['time_in_min'].mean()

    '''
    Step 3: Convert the average duration back to hours and minutes for display
    '''
    avg_hours = int(avg_duration // 60)
    avg_minutes = int(avg_duration % 60)

    '''
    Printing the number of movies and the average duration
    '''
    
    print(f'Unique Movies Count: {len(df.names.unique())}')
    print(f'Average Movie Duration: {avg_hours} hours and {avg_minutes} minutes')   

@main.command('train')
def train():
    """
    Train a classifier and save it.
    """
    # (1) Read the data...
    df = data2df()    
    # (2) Create classifier and vectorizer.
    clf = LogisticRegression(max_iter=1000, C=1, class_weight='balanced')         
    vec = CountVectorizer(min_df=5, ngram_range=(1,3), binary=True, stop_words='english')
    X = vec.fit_transform(df.names)
    y = df.mood.values
    # (3) do cross-validation and print out validation metrics
    # (classification_report)
    do_cross_validation(clf, X, y)
    # (4) Finally, train on ALL data one final time and
    # train. Save the classifier to disk.
    clf.fit(X, y)
    pickle.dump((clf, vec), open(clf_path, 'wb'))
    top_coef(clf, vec)

def do_cross_validation(clf, X, y):
    all_preds = np.zeros(len(y))
    for train, test in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X,y):
        clf.fit(X[train], y[train])
        all_preds[test] = clf.predict(X[test])
    print(classification_report(y, all_preds))    

def top_coef(clf, vec, labels=['liberal', 'conservative'], n=10):
    feats = np.array(vec.get_feature_names_out())
    print('top coef for %s' % labels[1])
    for i in np.argsort(clf.coef_[0])[::-1][:n]:
        print('%20s\t%.2f' % (feats[i], clf.coef_[0][i]))
    print('\n\ntop coef for %s' % labels[0])
    for i in np.argsort(clf.coef_[0])[:n]:
        print('%20s\t%.2f' % (feats[i], clf.coef_[0][i]))


if __name__ == "__main__":
    sys.exit(main())
