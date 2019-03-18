"""
Mutual Information Least-Dependent Component Analysis (MILCA)
Reference:  Stogbauer, H.; Kraskov, A.; Astakhov, S. A.; Grassberger, P. 
Phys. Rev. E, 2004, 70, 066123.
https://doi.org/10.1103/PhysRevE.70.066123
https://arxiv.org/abs/physics/0405044
"""

# Author:  Sergey Astakhov (astakhov@gmail.com)
# License: BSD 3 clause


import numpy as np
import scipy as sp
from sklearn.feature_selection import mutual_info_ as mi
from joblib import Parallel, delayed


class MILCA:
    
    def __init__(self, n_components=None, n_neighbours=None, n_angles = None, smoothing_band = None, n_jobs=None):
        self.n_components = n_components
        self.n_jobs = n_jobs
        self.n_neighbours = n_neighbours
        self.n_angles = n_angles
        self.smoothing_band = smoothing_band
        self.parallelizer = Parallel(n_jobs = self.n_jobs)
        
    def fit(self, X, y=None):
    
        if self.n_components is None:
            self.n_components = X.shape[1]
                
        if self.n_neighbours is None:
            self.n_neighbours = 10
        
        if self.n_angles is None:
            self.n_angles = 128
            
        if self.smoothing_band is None:
            self.smoothing_band = int(self.n_angles/4)
        
        d, E = np.linalg.eigh(np.cov(X.T))
        indx = np.argsort(d)[::-1][:self.n_components]
        d, E = d[indx], E[:, indx]
        D = np.diag(d)
        K = np.dot(np.linalg.inv(sp.linalg.sqrtm(D)), E.T)
        X1 = np.dot(X, K.T)
                      
        R = self.minimize_mi(X1)
                       
        self.components_ = np.dot(K.T, R.T).T
                      
        return self
        
    def transform(self, X, y=None): 
        #TODO assert fit
        Xt = np.dot(X,self.components_.T)
        return Xt
        
    def get_unmixing(self):
        #TODO assert fit
        return self.components_
        
    def minimize_mi(self, X):
                                
        R = np.eye(self.n_components)
        Y = X 
                
        for i in range(0, self.n_components):
          for j in [jj for jj in range(0, self.n_components) if jj != i]:
            R2 = np.eye(self.n_components) 
            R2[np.ix_([i,j], [i,j])] = self.minimize_mi_2D(Y[:,[i,j]])
            R = np.dot(R2, R)
            Y = np.dot(Y, R2.T)
            #print([i,j])
        
        return R
                                  
    def minimize_mi_2D(self, X):
     
        angles = np.linspace(0, np.pi/2, self.n_angles)
        MI = self.parallelizer(delayed(estimate_rotated_2D)(X, alpha, self.n_neighbours) for alpha in angles)
        MI_smooth = self.low_pass_filter(np.array(MI))
        R = rotation_matrix_2D(angles[np.argmin(MI_smooth)])
        return R
        
    def low_pass_filter (self, x):
    
        s = np.fft.fft(x)
        window = np.zeros(max(x.shape))
        window[:self.smoothing_band+1] = 1
        window[-self.smoothing_band:] = 1     
        return np.fft.ifft(s*window)

def estimate_rotated_2D(X, angle, n_neighbours):
    
    R = rotation_matrix_2D(angle)
    Y = np.dot(X, R.T)
    mutual_information = mi._compute_mi_cc(Y[:,0], Y[:,1], n_neighbours)
    return mutual_information

def rotation_matrix_2D(angle):
    
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return R    
    
