# Title: Utility Helpers
# Purpose: Provide shared text cleaning, normalization, and
# document-building helpers used across retrieval scripts.
# Date: 2026-04-22
# NOTE: adapted from 563_lab3 preprocess.py

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer # suffix stripping
import string
import re
import html


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

def clean_html(text):
    """Remove HTML tags from text."""
    
    if not isinstance(text, str):
        return text
    
    # assisted by https://regex101.com to generate regex patterns 
    text = re.sub(r"</?br\s*/?>", "\n", text) # replace common break tags with line breaks
    text = re.sub(r"\[\[VIDEOID:[^\]]*\]\]", "", text) #remove video ID tags
    text = re.sub(r"<[^>]+>", " ", text) # remove any remaining simple HTML tags
    text = html.unescape(text) # decode HTML &#34 entities into chars, suggested by https://stackoverflow.com/questions/65833681/
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()

def normalize(col):
    """Normalize mixed column values into readable strings.

    String inputs are returned unchanged, lists are recursively joined
    into one string, and dictionaries are flattened into key-value
    text.

    Parameters
    ----------
    col : str, list, dict, or other
        Single dataframe cell value to normalize.

    Returns
    -------
    str
        Normalized string representation of the input value.
    """
    if isinstance(col, str):
        return clean_html(col)

    if isinstance(col, list):
        strings = [normalize(item) for item in col]
        return " ".join(string for string in strings if string).strip()

    if isinstance(col, dict):
        strings = [f"{k} {normalize(v)}" for k, v in col.items()]
        return " ".join(string for string in strings if string).strip()

    return ""

def build_documents(df, columns):
    """Build one searchable document string per dataframe row.

    The selected columns are normalized and concatenated in row order,
    preserving alignment with any row-based identifiers such as
    `parent_asin`.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing the source text columns.
    columns : str or list of str
        Column name or names to combine into each searchable document.

    Returns
    -------
    list of str
        Searchable document strings, one per dataframe row.
    """

    if isinstance(columns, str):
        columns = [columns]

    documents = []

    for _, row in df.iterrows():
        strings = [normalize(row[col]) for col in columns]
        doc = " ".join(string for string in strings if string).strip()
        documents.append(doc)

    return documents
