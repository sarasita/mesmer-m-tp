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

# general
def converting_precipitation(pr_in_kgms, n_digits = 10):
    '''
    precipitation data from ESMs does usually not contain a cut-off value 
    therefore, the models usually do not output zero-percipitation, but rather
    very small residual values and sometimes also very small negative values.
    This function converts precipitation from kg/m^2/s to mm/day andapplies
    a cut-off to set ver small precipitaiton values to zero.

    Parameters
    ----------
    pr_in_kgms: ndarray of arbitrary shape
        Precipitation data in kg/m^2/s
    n_digits: int 
        number of digits behind the . to keep, precipitaiton is rounded and small
        values are rounded to 0
    
    Returns
    -------
    pr_in_mm : ndarray with same shape as pr_in_kgms
        Precipitation data in mm/day rounded s.t. only n digits after . are kept
    '''
    
    pr_in_mm                          = pr_in_kgms*86400
    # size                                    = len(pr_in_mm[pr_in_mm < 1*10**(-10)])
    pr_in_mm[pr_in_mm <  1*10**(-10)] = 1*10**(-n_digits)
    pr_in_mm                          = np.round(pr_in_mm, n_digits)
    return(pr_in_mm)