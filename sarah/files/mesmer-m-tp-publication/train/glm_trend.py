#################################
#       Import Statements       #
#################################
# computation
import xarray as xr
import numpy as np
import joblib
from joblib              import Parallel, delayed

# utils
from tqdm   import tqdm 
from pathlib import Path
import warnings

# plotting
import matplotlib.pyplot as plt
import cartopy.crs       as ccrs
import matplotlib as mpl
import seaborn as sns

# setting paths such that execution works from console & interactive shell 
import config.settings as cset
import config.constants as ccon

from sklearn.metrics.pairwise  import haversine_distances
from sklearn.decomposition     import PCA
from sklearn.preprocessing     import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

import statsmodels.api as sm

# response
# - locations used as predictors 
def get_closest_locations(coordinate_path = '/home/ubuntu/sarah/files/mesmer-m-tp-publication/data/new_coords.npy', 
                          n_closest = ccon.n_closest):
    '''
    Given an array of (lat, lon) coordinates, this function computes the pairwise distance 
    between all coordinate locations in the array and then returns an array 
    of indices that gives the indices of the n_closest locations 

    Parameters
    ----------
    coordinate_path: str or Path 
        String or pahtlib.Path object pointing to an .npy file that contains an array of shape (n_locations, 2)
        tghat contains (lat, lon) coordinates for all n_locations 
    n_closest: int 
        number of closest locations to compute
    
    Returns
    -------
    selected_loc_: list of ndarrays
        Contains the indices of the n_closest coordinates for each location. That is, for the location
        with index i, i.e. coords[i], the coordinates of the n_closest locations are 
        given by coords[selected_loc_[i]]
       
    '''
    coords           = np.load(coordinate_path)
    dist             = haversine_distances(np.pi/180*coords, np.pi/180*coords) * 6371 # *6371000/1000  to conver to km
    selected_loc_    = [np.argsort(dist[j,:])[:n_closest] for j in range(ccon.n_sindex)]
    # selected_loc      = [np.argwhere(dist[j,:] <= 3500).flatten() for j in range(n_index)] # for computing based on a radius; produces uneven results bc of missing values over ocean
    return(selected_loc_)

# - PCA transforming predictors 
def get_gridpoint_month_pca(X_tas):
    '''
    Performs a PCA on X_tas keeping only the first 8 principal components 

    Parameters
    ----------
    X_tas: ndarray of shape (year, n_closest_locations)
        X_tas is usually a collection of temperatures of a given month over multiple years at multiple locations; 
        The locations are consistent with that set in ccon.n_closest
    
    Returns
    -------
    PCA: fitted sklearn.PCA object
        Fitted PCA object
    '''
    return(PCA(n_components = 8).fit(X_tas))

# - fitting GLM 
# - fitting GLM 
def gridpoint_month_glm(X_tas, y, pca):
    '''
    Given an array of temperature predictor variables (X_tas), 
    a PCA object to map X_tas onto its principal components 
    and the target precipitation data (y), this function fits a 
    Generalized Linear Model to construct as much of y using X_tas 
    as predcitors. 
    Short Explanation: The first principal component of pca(X_tas) contains 
    a strong inter-annual trend and is therefore devided into a trend and a variability 
    component using lowess smoothing. Interactions between the trend component and the 
    other variability components are also allowed as predictor variables. To avoid 
    overfitting, we rely on an alpha parameter that is empirically estimated and tried to 
    keep as low as possible. For more detailed references refer to the MESMER-M-TP paper 

    Parameters
    ----------
    X_tas: ndarray of shape (n_years, n_closest_locations)
        X_tas is usually a collection of temperatures of a given month over multiple years at multiple locations; 
        The locations are consistent with that set in ccon.n_closest
        Used to construct predictor matrix for GLM
    y: ndarray of shape (n_years)
        Precipitation data. Target variable of the GLM. Usually describes precipitation for a given month 
        at a given location sampled over differnt years
    
    Returns
    -------
    std: fitted sklearn.preprocessing.StandardScaler object
        Scales the predictor matrix X_pred
    glm: statsmodels.genmod.generalized_linear_model.GLMResults
        GLM object that contains parameters and result metrics for the fitted GLM 
       
    '''
    n_years = np.shape(X_tas)[0]
    n_ssps = int((n_years - ccon.n_hist_years)/ccon.n_ssp_years)
    
    def generate_predictor_matrix(X_trafo_):
        # filter first component & devide in two 
        n_ts             = ccon.n_hist_years + ccon.n_ssp_years
        frac_lowess      = 50 / n_ts
        for i_ssp in range(n_ssps): 
            t_lowess_tmp    = lowess(np.append(X_trafo_[:ccon.n_hist_years, 0], X_trafo_[ccon.n_hist_years+i_ssp*ccon.n_ssp_years:ccon.n_hist_years+(i_ssp+1)*ccon.n_ssp_years, 0]),
                                            np.arange(n_ts), 
                                            return_sorted=False, 
                                            frac=frac_lowess, 
                                            it=0)
            if i_ssp == 0:
                t_lowess = t_lowess_tmp
            else: 
                t_lowess = np.append(t_lowess, t_lowess_tmp[ccon.n_hist_years:])
        
        X_pred_ = np.c_[t_lowess,
                        t_lowess**2,
                        X_trafo_[:, 0]-t_lowess, 
                        X_trafo_[:, 1:8],
                        t_lowess*(X_trafo_[:, 0]-t_lowess),
                        t_lowess*X_trafo_[:, 1],
                        t_lowess*X_trafo_[:, 2],
                        t_lowess*X_trafo_[:, 3],
                        t_lowess*X_trafo_[:, 4],
                        t_lowess*X_trafo_[:, 5],
                        t_lowess*X_trafo_[:, 6],
                        t_lowess*X_trafo_[:, 7]
                        ]
        return(X_pred_)

    X_trafo = pca.transform(X_tas)[:, :15]
        
    X_pred      = generate_predictor_matrix(X_trafo)
    std         = StandardScaler()
    X_pred_std  = np.c_[np.ones(n_years),  std.fit_transform(X_pred)]

    warnings.filterwarnings('ignore')
    
    L1_wt = 0.001
    
    trend = np.ones_like(y)*1000
    
    all_alphas       = [4*[0] + 15*[0], 
                        4*[0] + 7*[0] + 8*[1],
                        2*[0] + 1*[1] + 1*[0] +  7*[0] + 8*[1],
                        4*[0] + 15*[1], 
                        4*[0] + 7*[1] + 8*[10],
                        4*[0] + 15*[10],
                        4*[0] + 7*[10] + 8*[100],
                        4*[0] + 15*[100], 
                        1*[0] + 18*[100],
                        1*[0] + 18*[1000],
                        1*[0] + 18*[100000]]
    
    n_fitting_rounds = len(all_alphas)
    
    i_fitting_round = 0 
    
    while (trend.max()>1.1*y.max()) and (i_fitting_round < n_fitting_rounds):
        alphas = all_alphas[i_fitting_round]
        i_fitting_round += 1
        try: 
            glm   = sm.GLM(endog = y, 
                        exog = X_pred_std, 
                        family = sm.families.Gamma(link = sm.families.links.Log())).fit_regularized(alpha = alphas, 
                                                                                                    L1_wt = L1_wt, 
                                                                                                    refit = False)
            trend = glm.fittedvalues
        except: 
            continue 
        
    if (i_fitting_round == n_fitting_rounds) and (trend.max()>1.1*y.max()): 
        print('fitting failed')
        return(0)
    else:
        return(std, glm)