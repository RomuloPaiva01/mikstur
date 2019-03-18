"""
Quality metrics for assessing mixture decomposition (blind source separation) performance
"""

# Author:  Sergey Astakhov (astakhov@gmail.com)
# License: BSD 3 clause

import numpy as np

def amari_index(A, B):
       
    P = np.abs(np.dot(A, np.linalg.pinv(B)))
    n = P.shape[0]
    
    Pmax0 = np.array(P.max(0))
    Pmax1 = np.array(P.max(1))
            
    D  = P / np.tile(Pmax0, (n, 1))
    D += P / np.tile(Pmax1, (n, 1)).T 
    a_index = (D.sum()/(2*n) - 1)/(n-1) 
    
    return a_index
