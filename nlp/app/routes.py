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



'''
clf, vec = pickle.load(open(clf_path, 'rb'))
print('read clf %s' % str(clf))
print('read vec %s' % str(vec))
labels = ['liberal', 'conservative']

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	form = MyForm()
	result = None
	if form.validate_on_submit():
		input_field = form.input_field.data
		X = vec.transform([input_field])
		pred = clf.predict(X)[0]
		proba = clf.predict_proba(X)[0].max()
		# flash(input_field)
		return render_template('myform.html', title='', form=form, 
								prediction=labels[pred], confidence='%.2f' % proba)
		#return redirect('/index')
	return render_template('myform.html', title='', form=form, prediction=None, confidence=None)
'''


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

#Default data
#tfidf_matrix = tfidf_matrix_programs

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    if form.validate_on_submit():
        input_query = form.input_field.data
        # Using the chat function to get response
        search_mode = 'courses' if 'search-mode' in request.form and request.form['search-mode'] == 'on' else 'majors/minors'
        
        k_val = 4
        

		# switching between the two necessary parameter options
        if search_mode == 'courses':
            print("courses")
            vectorizer = vectorizer_courses
            tfidf_matrix = tfidf_matrix_courses
            documents = courses_documents
            k_val = 20
        else:
            print("programs") 
            vectorizer = vectorizer_programs
            tfidf_matrix = tfidf_matrix_programs
            k_val = 5
            documents = programs_documents
            
            
            

        response = answer_question(input_query, documents, vectorizer, tfidf_matrix, "gpt-3.5-turbo", top_k=k_val)
        # Directly pass the response to the template
        return render_template('index.html', title='Chat with AI Advisor', form=form, response=response)
    # When there is no POST request or form submission, pass None for response
    return render_template('index.html', title='Chat with AI Advisor', form=form, response=None, search_mode=search_mode)





# @app.route('/chat', methods=['GET', 'POST'])
# def chat():
#     form = MyForm()
#     if form.validate_on_submit():
#         message = form.input_field.data
#         # Assuming 'documents' and 'tfidf_matrix' are available here
#         # If not, you'll need to load and preprocess them first
#         answer = answer_question(message, documents, vectorizer, tfidf_matrix, "gpt-3.5-turbo")
#         # Pass the answer back to the template
#         return render_template('chat.html', form=form, answer=answer)
#     # Initial page load or not a POST request
#     return render_template('chat.html', form=form, answer=None)
