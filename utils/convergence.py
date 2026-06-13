def time_to_threshold(acc_list, threshold=0.70):
    """
    Returns epoch where accuracy first crosses threshold
    """
    for i, acc in enumerate(acc_list):
        if acc >= threshold:
            return i
    return None