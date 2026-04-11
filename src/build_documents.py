from src.normalize import normalize

def build_documents(df, columns):
    """Buils one searchable document string per dataframe row by combining the normalized text from one or more selected column

    This is used before retrieval so that we have one source of combined text information

    Args:
        df (df): dataframe with normalized text columns
        columns (list or str): column name if one column, or list of columns 

    Returns:
        documents: list of documents where each item is a combination of normalized texts per observation
    """

    if isinstance(columns, str):
        columns = [columns]

    documents = []

    for _, row in df.iterrows():
        strings = [normalize(row[col]) for col in columns]
        doc = " ".join(string for string in strings if string).strip()
        documents.append(doc)

    return documents
