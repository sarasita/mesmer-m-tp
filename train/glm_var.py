import numpy as np
from tqdm   import tqdm 

from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

def transform_residuals(residuals, n_components = None):
    '''
    Given an array of residuals (n_years, n_locations), 
    this function fits & applies a StandardScaler to the residuals 
    to get them all on the same scale and then applies a PC-decomposition
    If n_components is not given, it keeps as many components as there are
    sampels in the original dataset

    Parameters
    ----------
    residuals: ndarray of shape (n_years, n_locations)
        Array of precipitationr residuals
    n_componets: int or fraction in (0, 1)  
        If int, defines the number of components to keep during PCA,
        if fraction the number of components is chosesn such that those
        components keep the given fraction of variance in the signal
    
    Returns
    -------
    StdPca_pipeline: fitted skleanr.pipeline.Pipeline obejct
        Pipeline object that combines results for StandardScaler and PCA
    '''
       
    standard_scaler          = StandardScaler()
    pca_trafo                = PCA(n_components) 
    estimators               = [("standard_scaler", standard_scaler), ("pca_transform", pca_trafo)]
    StdPca_pipeline          = Pipeline(estimators).fit(residuals)
    return(StdPca_pipeline)

def month_specific_kde(residuals):
    '''
    Fits a KDE to the reisduals

    Parameters
    ----------
    residuals: ndarray of shape (n_years, n_components)
        Array of PCA-transformed precipitation residuals
    
    Returns
    -------
    kdes: list of fitted sklearn.neighbors.KernelDensity objects
        - 
    '''
    kdes             = []
    for m in tqdm(range(12), total = 12):         
        kde_m          = KernelDensity(metric = 'chebyshev', bandwidth = 0.05).fit(residuals[m::12, :])
        kdes.append(kde_m)
    return(kdes)