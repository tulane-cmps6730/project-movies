# CMPS 6730 Netflix Movie Recommendation

*This repository contains starter code for the final project in CMPS 4730/6730: Natural Language Processing at Tulane University.*
*This code has been copied to our project repository as a skeleton for us to use for our project.*

## Goals, methods, conclusions: 

For our Project, we are building a Movie recomendation system. The project is built on an ensemble of methods: Firstly, Collaborative-Filtering filtering is a way to reccomend movies to a user, based off the preferences of that user. In this case, this would be the movies this specific user has rated.
By aggregating the feature vector of each item (movie) a user has reviewed, we can build this user profile. From here we can use cosine similarity to compute the similarity between users. This already gives us a great start in predicting which movies a user may like, because odds are it is one that someone else
who shares their movie taste also likes.

While user - user similarity is important, we also want to calculate the similarity of movie - movie. To build profiles for these movies that we will later compare, we must perform a variety of NLP techniques in order to obtain the various features of each movie. This mainly
comes in the form of generating embeddings for the movie descriptions, as well as the mood, genre, and tag of the film. Capturing these features on a semantic level is important for comparison, and is relevant because films with genres like "thriller" and "horror" will now be considered similar. Simple categorical encoding fails to catch these dimensions.


The structure of the code supports the following:

- A simple web UI using Flask to support a demo of the project
- A command-line interface to support running different stages of the project's pipeline
- The ability to easily reproduce your work on another machine by using virtualenv and providing access to external data sources.

### Using this repository

- At the start of the course, students will be divided into project teams. Each team will receive a copy of this starter code in a new repository. E.g.:
https://github.com/tulane-cmps6730/project-alpha
- Each team member will then clone their team repository to their personal computer to work on their project. E.g.: `git clone https://github.com/tulane-cmps6730/project-alpha`
- See [GettingStarted.md](GettingStarted.md) for instructions on using the starter code.


### Contents

- [docs](docs): template to create slides for project presentations
- [nlp](nlp): Python project code
- [notebooks](notebooks): Jupyter notebooks for project development and experimentation
- [report](report): LaTeX report
- [tests](tests): unit tests for project code

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
