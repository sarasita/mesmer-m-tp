import xarray as xr 
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.linear_model import LinearRegression

def Tglob_lowess_smoothed(dataarray):
    # 50 years 
    # weights decaying with increasing distance according to tricube weight function 
    # number of timesteps
    n_ts                = len(dataarray.year)
    # add routine in case more than one variable is present
    frac_lowess         = 50 / n_ts
    # frac_lowess_name    = "50/n_ts"
    gt_lowess           = lowess(dataarray.values, np.arange(n_ts), return_sorted=False, frac=frac_lowess, it=0)
    dataarray_lowess    = xr.DataArray(data  = gt_lowess, dims = ['year'], coords = dict(year = dataarray.year.values)) 
    return(dataarray_lowess)
    # return(dataarray_lowess.expand_dims(dim = dict(run_id = [run_id])))

def Tglob_volcanic(dataset, da_aod = None):
    if da_aod is None: 
        da_aod         = historic_aod()
    reg            = LinearRegression().fit(da_aod.values.reshape(-1, 1), (dataset.values).reshape(-1, 1))
    Tglob_volcanic = reg.predict(da_aod.values.reshape(-1, 1)).flatten()
    return(Tglob_volcanic, np.array([reg.coef_.flatten(), reg.intercept_.flatten()]).flatten())

def historic_aod(last_year = 2100, aod_path = '/home/ubuntu/sarah/files/mesmerdev/data/isaod_gl.txt'):
    aod_og_df           = pd.read_csv(aod_path, sep = "  | ", header  = None, skiprows = 11, engine = 'python')
    aod_og_df.columns   = ['year', 'month', 'aod']
    aod_fill_df         = pd.DataFrame(data = np.array([np.repeat(np.arange(2020,last_year + 1),12), 
                                                        np.tile(np.arange(1, 13), last_year-2019), 
                                                        np.zeros(12*(last_year-2019)) ]).T, columns = ['year', 'month', 'aod'])
    aod_df              = pd.concat([aod_og_df, aod_fill_df])
    aod_df['date']      = pd.to_datetime(aod_df[['year', 'month']].assign(DAY=1))
    da_aod              = xr.DataArray(data = aod_df.aod, dims = ['time'], coords = dict(time = aod_df['date']))
    month_length        = da_aod.time.dt.days_in_month
    weights             = month_length.groupby("time.year") / month_length.groupby("time.year").sum()
    da_aod_weighted     = (da_aod * weights).groupby("time.year").sum(dim="time")
    return(da_aod_weighted)









