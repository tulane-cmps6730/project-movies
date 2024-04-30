# CMPS 6730 Final Project: Tulane Academic Advisor with RAG

*This repository contains the code for the final project in CMPS 4730/6730: Natural Language Processing at Tulane University, focusing on creating a Retrieval Augmented Generation (RAG) system to answer Tulane-specific academic questions.*

## Goals, methods, conclusions: 

Our project aims to develop a robust academic advising system for Tulane University that uses RAG to provide quick and accurate answers to student inquiries about academic programs and courses. The system scrapes data from Tulane's website, processes it for NLP tasks, and utilizes a state-of-the-art language model to generate responses.


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
