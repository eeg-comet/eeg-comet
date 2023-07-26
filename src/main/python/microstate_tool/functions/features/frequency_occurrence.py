from collections import Counter
import numpy as np


def frequency_occurrence(segment, sampling_rate, duration_of_window=1):
    """
    Computes the frequency of occurrence of each element in the input segment every one second of data.

    Args:
    segment (numpy array): input sequence of symbols.
    sampling_rate (int, optional): sampling frequency of the sequence.

    Returns:
    dict: dictionary with keys as symbols and values as their frequency of occurrence per second.
    """

    sampling_rate = int(sampling_rate / duration_of_window)
    # Compute the duration of the segment in seconds
    duration = len(segment) / sampling_rate

    # Compute the number of one-second intervals in the segment
    num_intervals = int(np.ceil(duration))

    # Divide the segment into one-second intervals and count the occurrence of each symbol in each interval
    symbols_per_interval = [[] for _ in range(num_intervals)]
    for i, s in enumerate(segment):
        interval_index = int(i / sampling_rate)
        if not symbols_per_interval[interval_index] or s != symbols_per_interval[interval_index][-1]:
            symbols_per_interval[interval_index].append(s)

    # Compute the occurrence per second of each symbol in the entire segment
    symbol_counts = [Counter(interval) for interval in symbols_per_interval]
    total_counts = sum(symbol_counts, Counter())
    freq_per_second = {symbol: count / duration for symbol, count in total_counts.items()}

    return freq_per_second
