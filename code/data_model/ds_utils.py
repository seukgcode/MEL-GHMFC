"""
    -----------------------------------
    dataset utils
"""


# Longest common substring
def lcss(s1, s2):
    """
        time complexity:
        ------------------------------------------
        Args:
        Returns:
    """
    m, n = len(s1), len(s2)
    if m > n:
        return lcss(s2, s1)
    if m < 1:
        return "", 0

    max_len = 0
    max_i = 0

    row = 0
    col = len(s2) - 1
    while row < len(s1):
        len_now = 0
        i, j = row, col
        while i < len(s1) and j < len(s2):
            if s1[i] == s2[j]:
                len_now += 1
                if len_now > max_len:
                    max_len = len_now
                    max_i = i
            else:
                len_now = 0

            i += 1
            j += 1

        if col > 0:
            col -= 1
        else:
            row += 1

    return s1[max_i + 1 - max_len: max_i + 1], max_len

