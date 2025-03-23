from cpython cimport array
import numpy as np

def testType(object o):
    print type(o)

def dist(str1, str2):
    '''
    :param str1, str2: numpy arrays of integers (words)
    :return: levenshtein distance between str1 and str2
    '''
    cdef int N = len(str1)
    cdef int M = len(str2)
    cdef int i
    cdef int j
    cdef int dl
    cdef int ins
    cdef int subst
    #cdef array.array a1 = array.array('I', str1)
    #cdef array.array a2 = array.array('I', str2)
    cdef int[:] a1 = str1
    cdef int[:] a2 = str2
    #cdef int dist[N][M]
    distnp = np.empty((N+1, M+1), dtype=np.int32)
    cdef int [:, :] dist = distnp
    for i in range(N+1): dist[i][0] = i
    for i in range(M+1): dist[0][i] = i
    # calc levenshtein
    for i in range(1, N+1):
        for j in range(1, M+1):
            if a1[i-1] == a2[j-1]: dist[i][j] = dist[i-1][j-1]
            else:
                dl = dist[i-1][j]+1
                min = dl
                ins = dist[i][j-1]+1
                if ins < min: min = ins
                subst = dist[i-1][j-1]+1
                if subst < min: min = subst
                dist[i][j] = min
    return dist[N][M]

def distStop(str1, str2, max):
    '''
    :param str1, str2: numpy arrays of integers (words)
    :param max: if min. possible distance exceeds max, return max+1
    :return: levenshtein distance between str1 and str2, or max+1
    '''
    # make s1 (row axis) larger of the two string, and s2 (column axis) smaller
    if len(str1) < len(str2): s2 = str1; s1 = str2
    else: s1 = str1; s2 = str2
    # init lengths, and string memviews
    cdef int N = len(s1)
    cdef int M = len(s2)
    cdef int[:] a1 = s1
    cdef int[:] a2 = s2
    # init helper and indexes
    cdef int mx = max
    cdef int mn
    cdef int i
    cdef int j
    cdef int dl
    cdef int ins
    cdef int subst
    #cdef array.array a1 = array.array('I', str1)
    #cdef array.array a2 = array.array('I', str2)
    #cdef int dist[N][M]
    distnp = np.empty((N+1, M+1), dtype=np.int32)
    cdef int [:, :] dist = distnp
    for i in range(N+1): dist[i][0] = i
    for i in range(M+1): dist[0][i] = i
    # calc levenshtein
    for i in range(1, N+1):
        mn = 10000000
        for j in range(1, M+1):
            if a1[i-1] == a2[j-1]: dist[i][j] = dist[i-1][j-1]
            else:
                dl = dist[i-1][j]+1
                min = dl
                ins = dist[i][j-1]+1
                if ins < min: min = ins
                subst = dist[i-1][j-1]+1
                if subst < min: min = subst
                dist[i][j] = min
            if dist[i][j] < mn: mn = dist[i][j]
        if mn > mx: return mx+1
    return dist[N][M]

