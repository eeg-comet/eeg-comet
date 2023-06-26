

def mean_durations(segment, sampling_rate):
    """
    Computes the mean duration of each element in the input segment in milliseconds.

    Args:
    segment (numpy array): input sequence of symbols.
    sampling_rate (int): sampling frequency of the sequence.

    Returns:
    dict: dictionary with keys as symbols and values as their mean duration in seconds.
    """
    durations = {}
    current_symbol = None
    current_duration = 0

    for symbol in segment:
        if symbol != current_symbol:
            if current_symbol is not None:
                # If the symbol has changed, add the duration to the list for the previous symbol
                if current_symbol not in durations:
                    durations[current_symbol] = []
                durations[current_symbol].append(current_duration / sampling_rate)

            current_duration = 1
            current_symbol = symbol
        else:
            current_duration += 1

    # Add the final duration to the list for the last symbol
    if current_symbol is not None:
        if current_symbol not in durations:
            durations[current_symbol] = []
        durations[current_symbol].append(current_duration / sampling_rate)

    # Compute the mean duration for each symbol
    mean_dur = {symbol: 1000 * sum(durations[symbol]) / len(durations[symbol]) for symbol in durations}

    return mean_dur
