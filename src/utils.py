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
    Cleans and tokenizes text. Can take in strings, lists of strings and missing values. It will lowercase, tokenize, remove punctuation and stopwords, and stem text for strings. For list of strings, it will apply the above process to each string in the list. For missing values, it will return an empty list.

    Parameters
    ----------
    text : str, list or None
        Raw input to be cleaned and tokenized.

    Returns
    -------
    list of str
        List of clean tokens in text.

    Examples
    --------
    >>> sample_text = "Primula Brew Buddy Portable Pour Over, Reusable Filter"
    >>> text_preprocessor(sample_text)
    ['primula', 'brew', 'buddi', 'portabl', 'pour', 'reusabl', 'filter']

    >>> sample_list = ["Primula Brew Buddy", "Portable Pour Over", "Reusable Filter"]
    >>> text_preprocessor(sample_list)
    ['primula', 'brew', 'buddi', 'portabl', 'pour', 'reusabl', 'filter']

    >>> mixed_list = [float('nan'), None, "Portable Pour Over", "Reusable Filter"]
    >>> text_preprocessor(mixed_list)
    ['portabl', 'pour', 'reusabl', 'filter']

    >>> text_preprocessor(None)
    []

    """

    # for text is type str
    if isinstance(text, str):
        tokens = text.lower().split()
    
    # for text is type list of str
    elif isinstance(text, list):
        tokens = [
            word
            for item in text if isinstance(item, str)
            for word in item.lower().split()
        ]
    
    # for text is type NaN or None
    else:
        return []

    clean_tokens = []

    for token in tokens:
        token = token.strip(string.punctuation)

        if token and token not in stop_words:
            stem_token = stemmer.stem(token)
            clean_tokens.append(stem_token)

    return clean_tokens
