from flask import Flask
import os
import spacy
from .. import nlp_path
app = Flask(__name__)
app.config['SECRET_KEY'] = 'you-will-never-guess'  # for CSRF


try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

from . import routes