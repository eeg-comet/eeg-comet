
#
'''
def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele
        # return string
    return str1
'''


# Remove consecutive duplicates from string
def remove_consecutive_duplicates(segmentation):
    segmentation = segmentation.tolist()
    # str_seg = ""
    # traverse in the string
    # for ele in segmentation:
    #     str_seg += ele
    # edited version using join
    str_seg = "".join(segmentation)

    # didn't understand .
    new_seg = ""
    prev = ""
    for c in str_seg:
        if len(new_seg) == 0:
            new_seg += c
            prev = c
        if c == prev:
            continue
        else:
            new_seg += c
            prev = c
    return new_seg

