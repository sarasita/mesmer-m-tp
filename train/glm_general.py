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
    pr_in_mm                          = pr_in_kgms*86400
    # size                                    = len(pr_in_mm[pr_in_mm < 1*10**(-10)])
    pr_in_mm[pr_in_mm <  1*10**(-10)] = 1*10**(-n_digits)
    pr_in_mm                          = np.round(pr_in_mm, n_digits)
    return(pr_in_mm)