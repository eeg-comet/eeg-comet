

def lempel_ziv_complexity(segment):
    """
    Calculate Lempel-Ziv complexity using the LZ76 algorithm and a sliding window implementation.

    Reference:
    F. Kaspar, H. G. Schuster, "Easily-calculable measure for the complexity of spatiotemporal patterns",
    Physical Review A, Volume 36, Number 2 (1987).

    Dolan D. et al (2018). The Improvisational State of Mind: A Multidisciplinary
    Study of an Improvisatory Approach to Clasegmentical Music Repertoire Performance.
    Front. Psychol. 9:1341. doi: 10.3389/fpsyg.2018.01341
    Pedro Mediano and Fernando Rosas, 2019

    Args:
        segment (numpy array): input sequence of symbols.

    Returns:
        float: Lempel-Ziv complexity of the input sequence.
    """
    segment = segment.flatten().tolist()
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    n = len(segment)
    while True:
        if segment[i + k - 1] == segment[l + k - 1]:
            k = k + 1
            if l + k > n:
                c = c + 1
                break
        else:
            if k > k_max:
                k_max = k
            i = i + 1
            if i == l:
                c = c + 1
                l = l + k_max
                if l + 1 > n:
                    break
                else:
                    i = 0
                    k = 1
                    k_max = 1
            else:
                k = 1
    return c / n

