from src.normalize import normalize

def build_documents(df, columns):

    if isinstance(columns, str):
        columns = [columns]

    documents = []

    for _, row in df.iterrows():
        strings = [normalize(row[col]) for col in columns]
        doc = " ".join(string for string in strings if string).strip()
        documents.append(doc)

    return documents
