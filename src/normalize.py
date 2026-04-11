def normalize(col):
    if isinstance(col, str):
        return col

    if isinstance(col, list):
        strings = [normalize(item) for item in col]
        return " ".join(string for string in strings if string).strip()

    if isinstance(col, dict):
        strings = [f"{k} {normalize(v)}" for k, v in col.items()]
        return " ".join(string for string in strings if string).strip()

    return ""