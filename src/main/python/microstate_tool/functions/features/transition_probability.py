
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions.utils.remove_consecutive_duplicates import remove_consecutive_duplicates

def transition_matrix(segmentation, visualize=False, colormap='Blues'):
    list_unique_segment_each = remove_consecutive_duplicates(segmentation)
    list_unique_segment_each = [(list_unique_segment_each[i:i + 1]) for i in range(0, len(list_unique_segment_each), 1)]
    segmentation = np.asarray(list_unique_segment_each)
    df = pd.DataFrame(segmentation)
    df['shift'] = df[0].shift(-1)
    df['count'] = 1
    trans_mat = df.groupby([0, 'shift']).count().unstack().fillna(0)
    labels = list(trans_mat.columns.levels[1])
    trans_mat = trans_mat.div(trans_mat.sum(axis=1), axis=0).values
    np.fill_diagonal(trans_mat, 0)
    if visualize == True:
        plt.matshow(trans_mat, cmap=colormap)
        x_pos = np.arange(len(labels))
        plt.xticks(x_pos, labels)
        y_pos = np.arange(len(labels))
        plt.yticks(y_pos, labels)
        plt.colorbar()
        plt.show()
    return trans_mat