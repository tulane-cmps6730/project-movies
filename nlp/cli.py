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
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter



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

#Important Variables setting up



encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
# Initialize the OpenAI client with your API key
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
# Initialize the OpenAI client with your API key
client = OpenAI(api_key=openai_api_key)

def preprocess_text(text):
    nlp = spacy.load("en_core_web_sm")
    # Parse the document using spaCy
    doc = nlp(text.lower())  # Convert text to lower case
    # Remove punctuation and stop words, and apply lemmatization
    clean_tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
    clean_text = ' '.join(clean_tokens)
    return clean_text



def chunk(chunk_size, documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=100, 
        length_function=len,
        is_separator_regex=False
    )
    all_docs = []
    for document in documents:
        chunks = text_splitter.create_documents([document])
        all_docs.extend(chunk.page_content for chunk in chunks)
    return all_docs



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

def load_documents(filepath):
    with open(filepath, 'rb') as file:
        return pickle.load(file)


def retrieve(query, vectorizer, tfidf_matrix, data, max_tokens=16000):
    if not data:
        return []

    try:
        query_tf = vectorizer.transform([query])
        similarities = cosine_similarity(query_tf, tfidf_matrix).flatten()
        query_keywords = set(query.lower().split())
        matches = []

        current_token_count = len(encoding.encode(query))
      
        
        for i, document in enumerate(data):
            title = document.split(',')[0].lower()
            title_keywords = set(title.split())
            common_keywords = query_keywords.intersection(title_keywords)
            keyword_count = len(common_keywords)
            combined_score = similarities[i] + (keyword_count * 0.5)  # Adjust the weight as needed

            # Tokenize the document to count token
            doc_token_count = len(encoding.encode(document))
            matches.append((document, combined_score, doc_token_count))
        matches.sort(key=lambda x: x[1], reverse=True)
        
        selected_documents = []
        current_token_count = 0


        iterator = 0
        for doc, combined_score, tokens in matches:
            if current_token_count + tokens > max_tokens:


                print("Tokens stopped at:", current_token_count)
                print(f'Relevant documents Found {iterator}')
                break  # Stop adding if the next document exceeds the token limit

            iterator += 1
            selected_documents.append((doc,combined_score))
            current_token_count += tokens
        return selected_documents 
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def answer_question(question, documents, vectorizer, tfidf_matrix, model, max_tokens=300, stop_sequence=None):
    retrieved_texts = retrieve(question, vectorizer, tfidf_matrix, documents)
    context = " ".join([text for text, _ in retrieved_texts])

    if context: 
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an academic advisor. Answer the question based on the context below"},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
                ],
                temperature=0,
                max_tokens=max_tokens,
                stop=stop_sequence,
            )
            # Get the response content
            response_content = response.choices[0].message.content.strip()
            html_response = '<p>' + '</p><p>'.join(response_content.split('\n')) + '</p>'
            
            return html_response
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
    

    print("Chat Active")
    
    script_dir = os.path.dirname(__file__)

# Construct the path to the "pre_processed" folder
    pre_processed_folder = os.path.join(script_dir, 'app', 'pre_processed')

# Set the path for the processed documents file
    processed_documents_path = os.path.join(pre_processed_folder, 'processed_documents.pkl')

# Load the processed documents
    documents = load_documents(processed_documents_path)
    

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
