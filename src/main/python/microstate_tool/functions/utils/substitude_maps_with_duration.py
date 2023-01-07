
import numpy as np
from itertools import groupby

def substitude_maps_with_duration(segmentation, min_duration):
    segmentation = np.array(segmentation)
    count_dups = [sum(1 for _ in group) for _, group in groupby(segmentation)]

    if min_duration:
        for C in range(len(count_dups)):
            print(100*C/len(count_dups))
            if count_dups[C] <= min_duration:
                start = int(np.sum(count_dups[0:C]))
                stop = int(start + count_dups[C])
                if C == 0:
                    segmentation[start:stop] = segmentation[stop + 1]
                else:
                    segmentation[start:stop] = segmentation[start - 1]
    return segmentation