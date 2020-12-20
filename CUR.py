import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

class CUR():
    def __init__(self, k, eps, it=None, truncated=False):
        self.k = k
        self.eps = eps
        self.trunc = truncated 
        self.c = k * np.log(k) / eps**2 #expectation number of sampled columns
        self.C, self.U, self.R = None, None, None #matrices of decomposition
        self.pi_col, self.pi_row = None, None #leverage scores of corresponding columns/rows
        self.col_indices = None
        self.row_indices = None
    
    def column_select(self, A):
        n = A.shape[1]
        A = np.array(A.copy())
        if self.trunc:
            _, _, v_k = randomized_svd(A, self.k) #for very big matrices
        else:
            _, _, vh = np.linalg.svd(A, full_matrices=False)
            v_k = vh[0:self.k, :]
        
        pi = 1 / self.k * np.sum(v_k**2, axis=0)
        c_index = [np.random.choice(2, 
                        p=[1 - min(1, self.c * pi[i]), min(1, self.c * pi[i])]) for i in range(n)
                  ]
        c_index = np.nonzero(c_index)[0]
        
        C = A[:, c_index]
        return C, c_index, pi

    def run_CUR(self, A):
        A = np.array(A.copy())
        self.C, self.col_indices, self.pi_col = self.column_select(A)
        self.R, self.row_indices, self.pi_row = self.column_select(A.T)
        self.U = np.linalg.pinv(self.C) @ A @ np.linalg.pinv(self.R.T)
        return self.C, self.U, self.R.T
