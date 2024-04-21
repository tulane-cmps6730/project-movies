# -*- coding: utf-8 -*-

import click
import glob
import pickle
import sys

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, classification_report
import click
import os
import sys

from dotenv import load_dotenv
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI



@click.group()
def main(args=None):
    """Console script for nlp."""
    return 0

@main.command('web')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(port):
    """
    Launch the flask web app.
    """
    from .app import app
    app.run(host='0.0.0.0', debug=True, port=port)

    



#setter up functions: ----------------------------------------------------------------------------------------------



# Initialize the OpenAI client with your API key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=)

def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    # Parse the document using spaCy
    doc = nlp(text.lower())  # Convert text to lower case

    # Remove punctuation and stop words, and apply lemmatization
    clean_tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]

    # Join the tokens back into a single string
    clean_text = ' '.join(clean_tokens)
    
    return clean_text

def load_and_preprocess_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                processed_text = preprocess_text(text)
                documents.append(processed_text)
    return documents

def retrieve(query, vectorizer, tfidf_matrix, data, top_k=3):
    # Validate inputs
    if not data or top_k <= 0:
        return []

    try:
        # Transform the query to the same vector space as the documents
        query_tf = vectorizer.transform([query])
        
        # Calculate cosine similarities between the query and all documents
        similarities = cosine_similarity(query_tf, tfidf_matrix).flatten()

        # Tokenize the query into keywords
        query_keywords = set(query.lower().split())

        # Prepare a list to store matches and their combined scores
        matches = []

        # Iterate over each document entry
        for i, document in enumerate(data):
            # Extract title from the document assuming it's the first sentence before the comma
            title = document.split(',')[0].lower()
            title_keywords = set(title.split())

            # Calculate the number of query keywords that appear in the title
            common_keywords = query_keywords.intersection(title_keywords)
            keyword_count = len(common_keywords)

            # Calculate a combined score
            # Here, you might want to balance the importance of cosine similarity and keyword count
            # For example, you could give a weight to keyword matches to adjust their influence
            combined_score = similarities[i] + (keyword_count * 0.5)  # Adjust the weight (0.1) as needed

            # Store the document along with its combined score
            matches.append((document, combined_score))

        # Sort by the combined scores in descending order
        matches.sort(key=lambda x: x[1], reverse=True)

        # Return the top_k most relevant documents based on the combined scores
        return matches[:top_k]

    except Exception as e:
        print(f"An error occurred: {e}")
        return []
    
# Function to answer questions using retrieved texts
def answer_question(question, documents, vectorizer, tfidf_matrix, model, top_k=5, max_tokens=200, stop_sequence=None):
    retrieved_texts = retrieve(question, vectorizer, tfidf_matrix, documents, top_k=top_k)
    context = " ".join([text for text, _ in retrieved_texts])
    

    if context:  # Check if there is any context retrieved
        try:
            # Create a chat completion using the question and context
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Answer the question based on the context below"},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                temperature=0,
                max_tokens=max_tokens,
                stop=stop_sequence,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return str(e)
    else:
        return "No relevant context found for the question."
    
#-------------------------------------------------------------------------------------------


@main.command('hello')
def hello():
    print("hello")

@main.command('chat')
def chat():
    """Interactive chat using the document retrieval system."""
    load_dotenv()

    print("Chat Active")
    
    script_dir = os.path.dirname(__file__)

    print(script_dir)

# Construct the path to the RAG_DATA directory
    data_dir = os.path.join(script_dir, '..', 'notebooks', 'RAG_DATA')

    print("Data Dir:", data_dir)
# Use the absolute path to load and preprocess documents
    documents = load_and_preprocess_documents(data_dir)

    print("Data Loaded")
    
    vectorizer = TfidfVectorizer(max_features=10000, min_df=2, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

    while True:
        try:
            message = input("Ask any questions about majors and minors at Tulane:\n")
            if message.lower() == 'exit':
                print("Exiting chat...")
                break
            answer = answer_question(message, documents, vectorizer, tfidf_matrix, "gpt-3.5-turbo")
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Sorry, I didn't understand that. Please try again.")



if __name__ == "__main__":
    sys.exit(main())






