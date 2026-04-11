def normalize(col):
    """Helper function for converting mixed dataframe
      column types into strings

    This module keeps string columns the same, and converts lists
    and dictionaries into readable text.
    Output is still in original row and column index.

    Args:
        col (str, list, dict): name of input column

    Returns:
        _type_: col of strings
    """
    if isinstance(col, str):
        return col

    if isinstance(col, list):
        strings = [normalize(item) for item in col]
        return " ".join(string for string in strings if string).strip()

    if isinstance(col, dict):
        strings = [f"{k} {normalize(v)}" for k, v in col.items()]
        return " ".join(string for string in strings if string).strip()

    return ""