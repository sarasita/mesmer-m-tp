import numpy as np
from tqdm   import tqdm 

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def transform_residuals(residuals, n_components = None):
    standard_scaler          = StandardScaler()
    pca_trafo                = PCA(n_components) 
    estimators               = [("standard_scaler", standard_scaler), ("pca_transform", pca_trafo)]
    StdPca_pipeline          = Pipeline(estimators).fit(residuals)
    return(StdPca_pipeline)

def month_specific_kde(residuals):
    kdes             = []
    for m in tqdm(range(12), total = 12):         
        kde_m          = KernelDensity(metric = 'chebyshev', bandwidth = 0.05).fit(residuals[m::12, :])
        kdes.append(kde_m)
    return(kdes)