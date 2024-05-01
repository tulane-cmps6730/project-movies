# CMPS 6730 Final Project: Tulane Academic Advisor with RAG

*This repository contains the code for the final project in CMPS 4730/6730: Natural Language Processing at Tulane University, focusing on creating a Retrieval Augmented Generation (RAG) system to answer Tulane-specific academic questions.*

## Goals, methods, conclusions: 

Our project aims to develop a robust academic advising system for Tulane University that uses RAG to provide quick and accurate answers to student inquiries about academic programs and courses. The system scrapes data from Tulane's website, processes it for NLP tasks, and utilizes a state-of-the-art language model to generate responses.


## Web Scraping
Our first step in the overall process was
to collect the relevant data. We used Beautiful
Soup to parse the Tulane courses page, where we
were able to extract relevant text data. After
obtaining a dictionary of each program name as
the keys, and the associated links as the values,
we parsed the two main pages for each
program, the requirements and home tabs
found under each program. From here, we
concatenated all the relevant text, and saved it
to a directory where each program was divided
into text files based on the topic. The result
was a directory of over 400 files, with a
hearty amount of text dedicated to each
program.
A similar procedure was done for the
Tulane courses page, only we had to
employ further text manipulation to accurately
extract each specific class offered. We used
regex matching to create a new entry every
time four capital letters were seen, indicative
of class descriptions like CMPS and ACCN.
We did further filtering out for courses that
had no descriptions, or special courses like
Independent Study which would offer no
relevance in our retrieval system. The output
of this were over a thousand very small class
files, each with the course code, and whatever
information about the class was given.
This web scraping process used no hard encoded
values besides the links to the websites, so the web
scraper can easily be run and obtain any new
courses, programs, or just general information
published on the Tulane website.
## Data Preparation
To turn our text file data into an easily retrievable
format, we first needed to create embeddings for
the documents. During data preprocessing, we
used the Spacy library to "clean" the text in each
document. This included normalizing the text to
all lowercase, removing punctuation and stop
words, and applying Lemmatization. In this
process, we also used Named Entity Recognition
to determine if each entity in a document was the
name of a program or class, in which case the
entire phrase/name would be added to the
processed text output, rather than a stripped
version.
This pre-processing is an essential step to
generate richer embeddings for the documents,
aiding in effective retrieval.
## Chunking
To further improve retrieval, we employed a
chunking technique to split up each document
into smaller parts. We found success with
relatively large chunks, around 5000 characters,
with 100 character overlaps. This improved the
number of documents retrieved from around 9-13
to 20-30 for the programs data. No chunking was
needed for the course data as documents are
already sufficiently small.
## Vectorization
We experimented with dense embeddings created
by models like BERT and the Openai model "textembeddings-
3-small", but concluded that they
provided no significant advantage over TF-ITF for our specific task.
## Retrieval
The actual retrieval method used was
a combination of cosine similarity and
keyword priority. the retrieval takes an input
query (a sentence) and vectorizes it into the
same space as the document embeddings using
TF-ITF again. From here we calculate a score for
each document based on the cosine similarity of
the vectors.
We add this to the keyword score, which
is calculated by taking the set of all word in
the query, and searching for those words in the
title of each document. This final score is then
added to a list called matches with the associated
document.
The next step in retrieval is to return the top_k to
form the context for the LLM. In our case,
the LLM we are using has a context window of
around 16,000 tokens, so in order to always
fill that context window, we employed a
dynamic top_k approach.
## Dynamic Top K
To optimize the amount of documents
retrieved with each query, we used tiktoken to
count the number of tokens appended to
the context window. This way, we can start at
the top of the matches list, getting the
documents with the highest combined scores,
and iteratively retrieve more documents until the
maximum token limit is reached. This step
proved essential, as it is effective on
databases of documents with greatly varying
lengths, and no top_k variable is needed to be
hardcoded.

## Conclusion:
Going forward, we plan on improving our
project in a variety of ways. Firstly, we would
like to gather much more data to draw from, as
we believe this is the primary factor that
improves the actual functionality of the project.
Furthermore, in terms of architecture, we could
replace some of our ground up functions with
more robust and concise methods, by employing
libraries like Langchain and Faiss. As our data
scales, it will be important to develop a more
robust data storage method, and we will
probably move from a simple directory of text
files to a Faise index. We collected informal user
feedback from around 15 people, and the main
queries had to do with finding easy classes.
Adding in some informal data from Reddit or
Rate my professor could provide this more
student-centered advice.
This project was instrumental in our
understanding of NLP, and engineering
methods as a whole. The biggest takeaway was
how useful vectors can be, and how obtaining
the semantic meaning of sentences in the
language of numbers can create very human-like
computer applications. Overall, it has made us
appreciative of the NLP researchers before us
that have worked so hard to create amazing
technologies that we can now leverage. None of
what we did would be possible without
breakthroughs like word2vec and "Attention is
all you need." We look forward to continuing as
contributers in the NLP community.

### Key Features:

- Data Collection: Utilizes Beautiful Soup for scraping data.
- Data Processing: Implements tokenization, lemmatization, and named entity recognition.
- Retrieval System: Uses cosine similarity and dynamic top-k retrieval to handle varied document lengths efficiently.
- Language Model Integration: Employs GPT-3.5-Turbo to enhance response quality.
- Web Interface: Developed using Flask, enabling easy interaction and mode switching between 'Program Expert' and 'Class Expert'.

### Structure:

- A simple web UI using Flask to support a demo of the project.
- A command-line interface to support running different stages of the project's pipeline.
- The ability to easily reproduce your work on another machine by using virtualenv and providing access to external data sources.

### Using this repository

- Clone the repository: git clone https://github.com/tulane-cmps6730/project-movies
- Install requirements: pip install -r requirements.txt
- Run the Flask app: 'nlp web'
- See [GettingStarted.md](GettingStarted.md) for instructions on using the starter code.


### Contents

- [docs](docs): template to create slides for project presentations.
- [nlp](nlp): Python project code for the RAG system.
- [notebooks](notebooks): Jupyter notebooks for project development and experimentation.
- [report](report): LaTeX report detailing the project's outcomes.
- [tests](tests): unit tests for project code.

### Background Resources

The following will give you some technical background on the technologies used here:

1. Refresh your Python by completing this online tutorial: <https://www.learnpython.org/> (3 hours)
2. Create a GitHub account at <https://github.com/>
3. Setup git by following <https://help.github.com/en/articles/set-up-git> (30 minutes)
4. Learn git by completing the [Introduction to GitHub](https://lab.github.com/githubtraining/introduction-to-github) tutorial, reading the [git handbook](https://guides.github.com/introduction/git-handbook/), then completing the [Managing merge conflicts](https://lab.github.com/githubtraining/managing-merge-conflicts) tutorial (1 hour).
5. Install the Python data science stack from <https://www.anaconda.com/distribution/> . **We will use Python 3** (30 minutes)
6. Complete the scikit-learn tutorial from <https://www.datacamp.com/community/tutorials/machine-learning-python> (2 hours)
7. Understand how python packages work by going through the [Python Packaging User Guide](https://packaging.python.org/tutorials/) (you can skip the "Creating Documentation" section). (1 hour)
8. Complete Part 1 of the [Flask tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world), which is the library we will use for making a web demo for your project.
