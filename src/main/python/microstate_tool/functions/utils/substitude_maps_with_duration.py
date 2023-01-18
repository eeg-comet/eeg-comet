
import numpy as np
from itertools import groupby


def substitude_maps_with_duration(segmentation, fs, remove_segments_less_than, option):
    segmentation = np.asarray(segmentation)
    segmentation = segmentation + 1
    count_dups = [sum(1 for _ in group) for _, group in groupby(segmentation)]
    #if remove_segments_less_than:
    remove_segments_less_than = int(int(remove_segments_less_than) / (1000 / fs))
    for C in range(len(count_dups)):
        if count_dups[C] <= remove_segments_less_than:
            start = int(np.sum(count_dups[0:C]))
            stop = int(start + count_dups[C])
            if option == 'replace':
                if C == 0:
                    segmentation[start:stop] = segmentation[stop + 1]
                else:
                    segmentation[start:stop] = segmentation[start - 1]
            elif option == 'remove':
                segmentation[start:stop] = 0
    return segmentation
