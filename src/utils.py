# adapted from 563_lab3 preprocess.py

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer # suffix stripping
import string

# initial download only
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def text_preprocessor(text):
    """
    Cleans and tokenizes text

    Parameters
    ----------
    text : str
        raw text

    Returns
    -------
    list of str
        list of clean tokens in text

    Examples
    --------
    >>> sample_text = "Primula Brew Buddy Portable Pour Over, Reusable Fine Mesh Filter, Dishwasher Safe, Single Cup of Coffee or Tea at Any Strength, Ideal for Travel or Camping, 404.88 milliliters, Red"
    >>> text_preprocessor(sample_text)
    ['primula', 'brew', 'buddi', 'portabl', 'pour', 'reusabl', 'fine', 'mesh', 'filter', 'dishwash', 'safe', 'singl', 'cup', 'coffe', 'tea', 'strength', 'ideal', 'travel', 'camp', '404.88', 'millilit', 'red']
    """
    text = text.lower()
    tokens = text.split()

    clean_tokens = []

    for token in tokens:
        token = token.strip(string.punctuation)

        if token and token not in stop_words:
            stem_token = stemmer.stem(token)
            clean_tokens.append(stem_token)

    return clean_tokens
