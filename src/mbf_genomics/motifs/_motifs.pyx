import cython
from libc.string cimport strchr
cimport numpy as np

     
def differential_mismatch_motif_scan(iupac_consensus, sequence, max_score, minimum_threshold, penalties):
    """Scan a sequence for an iupac consensus which may be weighted by penalties.
    If penalties is a constant value, degenerato to a mismatch motif scan
    """
    if minimum_threshold is None:
        minimum_threshold = len(iupac_consensus) / 2
    cdef int max_number_of_mismatches = len(iupac_consensus) - minimum_threshold
    starts = []
    stops = []
    scores = []
    cdef char* seq = sequence
    cdef Py_ssize_t iupac_length = len(iupac_consensus)
    cdef char* iupac_code = iupac_consensus
    cdef Py_ssize_t ii
    cdef Py_ssize_t tt
    cdef float mismatch_count

    for ii in range(0, len(sequence) - max_score + 1):
        mismatch_count = 0
        for tt in range(0, iupac_length):
            if (
                (iupac_code[tt] == seq[ii + tt]) or 
                (iupac_code[tt] == 'N') or 
                #(seq[ii + tt] == 'N') or an n is an automatic mismatch
                (iupac_code[tt] == 'R' and (strchr("AGSWKM", seq[ii + tt]) != NULL)) or
                (iupac_code[tt] == 'S' and strchr("CGRKMY", seq[ii + tt]) != NULL) or 
                (iupac_code[tt] == 'W' and strchr("TARYKM", seq[ii + tt]) != NULL) or 
                (iupac_code[tt] == 'K' and strchr("TGYRWS", seq[ii + tt]) != NULL) or 
                (iupac_code[tt] == 'M' and strchr("CAYRSW", seq[ii + tt]) != NULL) or 
                (iupac_code[tt] == 'Y' and strchr("CTSWKM", seq[ii + tt]) != NULL) or 
                (iupac_code[tt] == 'B' and strchr("CTGRYMKWS", seq[ii + tt]) != NULL) or  #these accept any of the three letter codes!
                (iupac_code[tt] == 'D' and strchr("ATGRYMKWS", seq[ii + tt]) != NULL) or  #these accept any of the three letter codes!
                (iupac_code[tt] == 'H' and strchr("ATCRYMKWS", seq[ii + tt]) != NULL) or  #these accept any of the three letter codes!
                (iupac_code[tt] == 'V' and strchr("ACGRYMKWS", seq[ii + tt]) != NULL)  #these accept any of the three letter codes!
                
                ):
                continue
            else:
                mismatch_count += penalties[tt];
                if (mismatch_count > max_number_of_mismatches):
                    break

        if (tt == iupac_length - 1 and mismatch_count <= max_number_of_mismatches):
                scores.append(max_score - mismatch_count)
                starts.append(ii)
                stops.append(ii + iupac_length)
    return starts, stops, scores

    
def scan_pwm(sequence, matrix_forward, matrix_reverse, threshold, min_needed_matrix, min_rev_needed_matrix, keep_list):
    cdef Py_ssize_t ii
    cdef double cum_score = 0
    cdef double  max_score = 0
    cdef char* seq = sequence
    cdef double score_forward
    cdef double score_reverse
    cdef int c
    cdef Py_ssize_t sequence_length = len(sequence)
    cdef Py_ssize_t Nmatrix0 = matrix_forward.shape[0]
    cdef long sp
    cdef long ep
    cdef long temp
    cdef double score_used
    cdef float _threshold = threshold
    cdef int _keep_list = bool(keep_list)
    starts = []
    stops = []
    scores = []
    cdef np.ndarray[float, ndim=2] _matrix_forward = matrix_forward
    cdef np.ndarray[float, ndim=2] _matrix_reverse = matrix_reverse
    cdef np.ndarray[float, ndim=1] _min_needed_matrix = min_needed_matrix
    cdef np.ndarray[float, ndim=1] _min_rev_needed_matrix = min_rev_needed_matrix
     
    for ii in range(sequence_length):
        score_forward = 0;
        score_reverse = 0;
        for c in range(Nmatrix0):
            if (ii + c == sequence_length):
                score_forward = 0
                score_reverse = 0
                break
            score_forward += _matrix_forward[c, (seq[ii + c])]
            score_reverse += _matrix_reverse[c, (seq[ii + c])]
            if ( (score_forward + 0.0001< _min_needed_matrix[c]) and (score_reverse + 0.0001< _min_rev_needed_matrix[c])):
                break

        if ((score_forward + 0.0001>= _threshold) or (score_reverse+ 0.0001 >= _threshold)):
            sp = ii
            ep = sp + Nmatrix0
            score_used = 0
            if (score_reverse + 0.0001> _threshold):
                temp = sp
                sp = ep
                ep = temp
                score_used = score_reverse
            else:
                score_used = score_forward
            if (score_used > max_score):
                max_score = score_used
            cum_score += score_used
            if (keep_list):
                starts.append(sp)
                stops.append(ep)
                scores.append(score_used)
    return cum_score, max_score, starts, stops, scores
           
