from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm


from .. import clf_path
import os
from flask import current_app
import os
from flask import current_app

import pickle
import sys

from nlp.cli import load_and_preprocess_documents, preprocess_text, answer_question, load_documents

from sklearn.feature_extraction.text import TfidfVectorizer
from flask import render_template, request, redirect, url_for

from sklearn.feature_extraction.text import TfidfVectorizer

import tiktoken



# Now construct the path to the RAG_DATA directory relative to the Flask app's root
# Get the absolute path of the current file (routes.py)
current_file_path = os.path.abspath(__file__)

# Get the directory in which the current file resides
current_directory = os.path.dirname(current_file_path)

# Construct the path to the RAG_DATA directory relative to the current file
data_dir = os.path.join(current_directory, 'RAG_DATA')


script_dir = os.path.dirname(os.path.abspath(__file__))

# Set the path for the pre_processed folder
pre_processed_folder = os.path.join(script_dir, 'pre_processed')

# Set the path for the processed documents file
processed_documents_path = os.path.join(pre_processed_folder, 'processed_documents.pkl')
processed_courses_path = os.path.join(pre_processed_folder, 'processed_courses.pkl')

# Assuming you have a function defined as load_documents to load your documents
programs_documents = load_documents(processed_documents_path)
courses_documents = load_documents(processed_courses_path)

vectorizer_programs = TfidfVectorizer(max_features=10000, min_df=2, stop_words="english")
tfidf_matrix_programs = vectorizer_programs.fit_transform(programs_documents)

vectorizer_courses = TfidfVectorizer(max_features=10000, min_df=2, stop_words="english")
tfidf_matrix_courses = vectorizer_courses.fit_transform(courses_documents)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    if form.validate_on_submit():
        input_query = form.input_field.data
        # Using the chat function to get response
        search_mode = 'courses' if 'search-mode' in request.form and request.form['search-mode'] == 'on' else 'majors/minors'
        
		
    

  

        if search_mode == 'courses':
            print("courses")
            vectorizer = vectorizer_courses
            tfidf_matrix = tfidf_matrix_courses
            documents = courses_documents
            #k_val = 20
        else:
            print("programs") 
            vectorizer = vectorizer_programs
            tfidf_matrix = tfidf_matrix_programs
            #k_val = 5
            documents = programs_documents
            
            
		
        response = answer_question(input_query, documents, vectorizer, tfidf_matrix, "gpt-3.5-turbo")
        # Directly pass the response to the template
        print(search_mode)
        return render_template('index.html', title='Chat with AI Advisor', form=form, response=response)
    # When there is no POST request or form submission, pass None for response
    return render_template('index.html', title='Chat with AI Advisor', form=form, response=None)




