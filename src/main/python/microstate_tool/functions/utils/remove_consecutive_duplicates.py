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
"""
Last Modified: April 18th, 2023
Description: This file defines a function for removing consecutive duplicates from a string.

Authors:
    Amin Kabir
    Raaj Chatterjee
    Faranak Farzan

Organization: SFU eBrain Lab, www.ebrainlab.ca
"""

def remove_consecutive_duplicates(segmentation):
    """
    Removes consecutive duplicate characters from a string.

    Inputs:
        segmentation (numpy array or list of strings): The string or list of strings to remove duplicates from.

    Outputs:
        new_seg (string): The new string with consecutive duplicates removed.
    """
    
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
            prev = c  # Add each character to the new string only if it is different from the previous character

    return new_seg  # Return the new string with consecutive duplicates removed

