from scipy import linalg as lin
import numpy as np

M = np.array([[1, 2], [2, 1], [3, 4], [4, 3]])
U, S, V = lin.svd(M, full_matrices=False)
print('U: ', U)
print('S: ', S)
print('V: ', V)

MM = np.dot(np.transpose(M), M)
evals, evecs = lin.eigh(MM)
print('evals: ', evals)
print('evecs: ', evecs)

evals_sort = sorted(evals, reverse=True)
print('evals_sort: ', evals_sort)

