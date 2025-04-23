def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary.
    Example:
        {"a": {"b": 1, "c": 2}} -> {"a_b": 1, "a_c": 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)