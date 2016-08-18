import numpy as np

def multi_log_loss(actual, predicted):
    """
    Columns of predicted are assumed to be in alphabetical order of the class
    :param actual: List of observed classes
    :param predicted: Matrix with predicted probabilities
    :return: Score value (numeric)
    """
    if not isinstance(actual, list):
        raise TypeError('actual must be a list!')
    if not isinstance(predicted, np.ndarray):
        raise TypeError('predicted must be a numpy array!')

    # Normalize predicted
    row_sums = predicted.sum(axis=1)
    predicted = predicted / row_sums[:, np.newaxis]

    unique_values = list(set(actual))
    unique_values.sort()
    if len(unique_values) != predicted.shape[1]:
        raise TypeError('Length of unique values of actual must match number of columns of predicted!')
    column = [i for row in actual for i, val in enumerate(unique_values) if row == val]
    predicted_log_probs = [np.log(x[i]) for x, i in zip(predicted, column)]
    return -sum(predicted_log_probs)