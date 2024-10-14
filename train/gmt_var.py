

import xarray as xr 
import numpy as np
import pandas as pd

from statsmodels.tsa.ar_model import AutoReg, ar_select_order


def Tglob_AR_parameters(dataset):
    """"
    At the moment, this function can only be used with continous data. 
    ToDo: Exchange usage of AR model with SARIMAX (https://www.statsmodels.org/devel/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
    Then, encode non-continous values by adding a nan estimate between any two values that do not belong together
    and follow https://www.statsmodels.org/devel/examples/notebooks/generated/statespace_sarimax_internet.html 
    """
    ar_lags     = ar_select_order(dataset['gmt_var'].values, maxlag = 12, ic = 'bic', old_names = False).ar_lags
    AR_model    = AutoReg(dataset['gmt_var'].values, lags = ar_lags, old_names = False)
    AR_results  = AR_model.fit()

    # time coeffs
    ar_params   = AR_results.params
    # noise term 
    ar_sigma    = np.sqrt(AR_results.sigma2)
    return(ar_lags, ar_params, ar_sigma)

def Tglob_generate_var(ar_lags, ar_params, ar_sigma, shape):
    """"
    Generates realisations for a specified Auto-Regressive (AR) process. 

    Args:
        ar_lags (np.array): Lags that should be included in AR. E.g. [1,4,7] for including lag-1,4 and 7
        ar_params (np.array): Parameter belonging to each of the lags 
        ar_sigma (np.float): Standard Deviation of innovation terms
        shape (tuple): Shape of the number of AR samples drawn 
    """
    
    buffer      = 50
    gmt_innovs  = np.random.normal(loc=0, scale=ar_sigma, size=shape+buffer)

    gmt_glob_var     = np.zeros(shape+buffer, dtype = float)
    gmt_glob_var[:2]  = gmt_innovs[:2]

    for t in range(np.max(ar_lags) + 1, shape + buffer):
        gmt_glob_var[t] = ar_params[0] + np.sum([gmt_glob_var[t - ar_lags[i]]*ar_params[1+i] for i in range(np.max(ar_lags))]) + gmt_innovs[t]

    return(gmt_glob_var[buffer:])