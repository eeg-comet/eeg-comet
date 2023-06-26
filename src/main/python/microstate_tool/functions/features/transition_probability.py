
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def transition_probability(segment):
    # Convert the segment to a list if it is a numpy array
    if isinstance(segment, np.ndarray):
        segment = segment.tolist()

    # Create a dictionary to store the frequency of transitions
    transitions = {}

    # Iterate over the segment and count the number of transitions
    for i in range(len(segment)-1):
        current = segment[i]
        next_ = segment[i+1]
        if current in transitions:
            if next_ in transitions[current]:
                transitions[current][next_] += 1
            else:
                transitions[current][next_] = 1
        else:
            transitions[current] = {next_: 1}

    # Convert the transition counts to probabilities
    for current in transitions:
        total = sum(transitions[current].values())
        for next_ in transitions[current]:
            transitions[current][next_] /= total

    # Convert the dictionary to a numpy array
    symbols = sorted(set(segment))
    n = len(symbols)
    matrix = np.zeros((n, n))
    for i, current in enumerate(symbols):
        if current in transitions:
            for j, next_ in enumerate(symbols):
                if next_ in transitions[current]:
                    matrix[i,j] = transitions[current][next_]

    # Return the transition probability matrix and the symbol order
    return matrix, symbols


def transition_matrix(segment, visualize=False, colormap='Blues'):
    #list_unique_segment_each = remove_consecutive_duplicates(segment)
    #list_unique_segment_each = [(list_unique_segment_each[i:i + 1]) for i in range(0, len(list_unique_segment_each), 1)]
    #segment = np.asarray(list_unique_segment_each)
    #df = pd.DataFrame(segment)
    #df['shift'] = df[0].shift(-1)
    #df['count'] = 1
    #trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)
    #labels = list(trans_mat.columns.levels[1])
    #trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values
    #np.fill_diagonal(trans_mat, 0)

    trans_mat, labels = transition_probability(segment)
    if visualize:
        plt.matshow(trans_mat, cmap=colormap)
        x_pos = np.arange(len(labels))
        plt.xticks(x_pos, labels)
        y_pos = np.arange(len(labels))
        plt.yticks(y_pos, labels)
        plt.colorbar()
        plt.show()
    return trans_mat
