# adapted from 563_lab3 preprocess.py

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer # suffix stripping
import string

# download only if not present
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")

def text_preprocessor(text):
    """
    Cleans and tokenizes text.

    Parameters
    ----------
    text : str
        The raw text.

    Returns
    -------
    list of str
        List of clean tokens in text.

    Examples
    --------
    >>> sample_text = "Primula Brew Buddy Portable Pour Over, Reusable Filter"
    >>> text_preprocessor(sample_text)
    ['primula', 'brew', 'buddi', 'portabl', 'pour', 'reusabl', 'filter']
    
    """

    # catch blank/NaN entries
    if not isinstance(text, str):
        return []

    text = text.lower()
    tokens = text.split()

    clean_tokens = []

    for token in tokens:
        token = token.strip(string.punctuation)

        if token and token not in stop_words:
            stem_token = stemmer.stem(token)
            clean_tokens.append(stem_token)

    return clean_tokens
