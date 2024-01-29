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
def get_closest_locations(n_closest = ccon.n_closest):
    coords           = np.load('/home/ubuntu/sarah/files/mesmerdev/data/new_coords.npy')
    dist             = haversine_distances(np.pi/180*coords, np.pi/180*coords) * 6371 # *6371000/1000  to conver to km
    selected_loc_    = [np.argsort(dist[j,:])[:n_closest] for j in range(ccon.n_sindex)]
    # selected_loc      = [np.argwhere(dist[j,:] <= 3500).flatten() for j in range(n_index)] # for computing based on a radius; produces uneven results bc of missing values over ocean
    return(selected_loc_)

# - PCA transforming predictors 
def get_gridpoint_month_pca(X_tas):
    return(PCA(n_components = 15).fit(X_tas))

# - fitting GLM 
def gridpoint_month_glm(X_tas, y, pca, alphas):
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
    try: 
        glm   = sm.GLM(endog = y, 
                    exog = X_pred_std, 
                    family = sm.families.Gamma(link = sm.families.links.Log())).fit_regularized(alpha = alphas, 
                                                                                                L1_wt = L1_wt, 
                                                                                                refit = False)
        trend = glm.fittedvalues
        # to prevent overfitting, we mark fits that generate
        # unrealistically high trend estimates as invalid 
        if trend.max() > 1.1*y.max():
            return(std, 0)
        else:
            return(std, glm)
    except: 
        return(std, 0)