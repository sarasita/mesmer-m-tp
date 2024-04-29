#%%
'''
Script opens the gmt output from calibrate_gmt_volcanic and approximates the global mean variability
of an autoregressive process where the number of lags is chosen using the AIC. The AR parameters are 
stored. 
'''

# setting paths correctly 
import os
import sys
os.chdir('/home/ubuntu/sarah/files/mesmer-m-tp-dev/')
sys.path.append('/home/ubuntu/sarah/files/mesmer-m-tp-dev/')

# computation
import xarray as xr
import numpy as np
import joblib
from joblib              import Parallel, delayed

# utils
from tqdm   import tqdm 
from pathlib import Path

# plotting
import matplotlib.pyplot as plt
import cartopy.crs       as ccrs
import matplotlib as mpl
import seaborn as sns

# setting paths such that execution works from console & interactive shell 
import config.settings as cset
import config.constants as ccon

from utils.load_data import load_and_prepare_land_dataset, load_land_gmt_dataset
from train.gmt_var   import Tglob_generate_var

#%%

if __name__ == '__main__':
    for model_id in tqdm(cset.model_ids[:], total = cset.n_models):
            run_id_training = cset.model_training_mapping[model_id]
            GMT_ds          = xr.open_dataset(cset.output_path.joinpath(f'processed_CMIP6/{model_id}/{model_id}_GMT-LO_processed.nc'))

            c_ssp_ids       = GMT_ds['rel_gmt'].sel(run_id = run_id_training).dropna(dim = 'year', how = 'all').ssp_id.values
            n_c_ssp_ids     = len(c_ssp_ids)-1 
            n_years         = 165 + n_c_ssp_ids*86
            
            # generate variability residuals for specific scenarios 
            for ssp_id in c_ssp_ids[1:]:                 
                GMT_emus                 =  np.zeros((100, ccon.n_hist_years + ccon.n_ssp_years))
                GMT_var                  =  np.zeros((100, ccon.n_hist_years + ccon.n_ssp_years))
                            
                AR_calib                 = joblib.load(cset.output_path.joinpath(f'calib/GMT-LO/GMT_var/{model_id}/AR-calib_{model_id}_{ssp_id}_{run_id_training}.pkl'))
                
                run_ids_all              = GMT_ds['rel_gmt_trend'].sel(ssp_id =  ssp_id).dropna(dim = 'year', how = 'all').dropna(dim = 'run_id').run_id.values
                for i in range(100):
                    # randomly select a GMT_Trend
                    GMT_trend            = np.zeros((ccon.n_hist_years + ccon.n_ssp_years))
                    run_id_rand          = run_ids_all[np.random.randint(0, len(run_ids_all))] 
                    GMT_trend[ccon.n_hist_years:] = GMT_ds['rel_gmt_trend'].sel(ssp_id =  ssp_id, run_id = run_id_rand).dropna(dim = 'year').isel().values
                    GMT_trend[:ccon.n_hist_years] = GMT_ds['rel_gmt_trend'].sel(ssp_id = 'historical', run_id = run_id_rand).dropna(dim = 'year').isel().values
                    
                    
                    GMT_var[i, :]        = Tglob_generate_var(AR_calib['AR_lags'], AR_calib['AR_params'], AR_calib['AR_sigma'], shape = ccon.n_hist_years + ccon.n_ssp_years)
                    GMT_emus[i, :]       = GMT_trend + GMT_var[i, :]
                
                # storing full emulations 
                storage_path = cset.output_path.joinpath(f'gmt_emus_LO/full/{model_id}/')
                Path(storage_path).mkdir(parents = True, exist_ok = True)
                joblib.dump(GMT_emus, storage_path.joinpath(f'GMT-emus_{model_id}_{ssp_id}_{run_id_training}.pkl'))
                
                # storing variability
                storage_path = cset.output_path.joinpath(f'gmt_emus_LO/var/{model_id}')
                Path(storage_path).mkdir(parents = True, exist_ok = True)
                joblib.dump(GMT_var, storage_path.joinpath(f'GMT-var_{model_id}_{ssp_id}_{run_id_training}.pkl'))

            

#%%


            