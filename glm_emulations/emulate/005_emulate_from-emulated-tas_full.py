#################################
#       Import Statements       #
#################################
# setting paths correctly 
import os
import sys
os.chdir('/home/ubuntu/sarah/files/mesmer-m-tp-publication/')
sys.path.append('/home/ubuntu/sarah/files/mesmer-m-tp-publication/')


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

# from utils.load_data import load_and_prepare_land_dataset, load_land_gmt_dataset
from utils.prepare_input import get_input_data

# scipr specific 
# training process
from sklearn.preprocessing   import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from statsmodels.api import GLM 

# global trend  
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import PolynomialFeatures
from sklearn.linear_model    import LinearRegression
from sklearn.model_selection import cross_val_score

# local trend 
from sklearn.metrics.pairwise   import haversine_distances

from statsmodels.api import GLM 
import statsmodels.api as sm

#%%

ssp_id      = 'ssp585'
n_emu_years = 251

if __name__ == '__main__':
    for model_id in cset.model_ids:
        print(model_id)
        
        run_id_training   = cset.model_training_mapping[model_id]
        n_emus            = 100
        n_emu_years       = 251
        
        # load emulations and glue variability to it 
        precip_emuvar_path  = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/loc_var/{model_id}/')
        new_samples         = joblib.load(precip_emuvar_path.joinpath(f'pr_var_{model_id}_{run_id_training}_v0.pkl'))[:100, :, :]
        
        precip_emutrend_path   = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/loc_trend/{model_id}/full/')
        precip_emu_path        = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/full/{model_id}/full/')
        Path(precip_emu_path).mkdir(parents = True, exist_ok = True)
        for i_run in range(n_emus): 
            pr_trend_emu = joblib.load(precip_emutrend_path.joinpath(f'pr_full-trend-emu_{model_id}_{ssp_id}_{i_run}.pkl'))
            pr_full      = pr_trend_emu*np.exp(new_samples[i_run, :, :])
            
            joblib.dump(pr_emus_v02, precip_emu_path.joinpath(f'pr_full-emus_{model_id}_{ssp_id}_{i_run}_v0.pkl'))
            
            # handling unrealistically high values, i.e. introducing a cut-pff
            pr_max   = pr_full.reshape(-1, 12, ccon.n_sindex).max(axis = 0)
            pr_q99   = np.quantile(pr_full.reshape(-1, 12, ccon.n_sindex), q = .99, axis = 0)
            idx_fail = np.argwhere(pr_max > 5*pr_q99)
            
            pr_emus_v02 = np.copy(pr_full)
            
            for m_, i_ in idx_fail: 
                pr_tmp                              = pr_emus_v02[m_::12, i_]
                pr_tmp[pr_tmp > pr_q99[m_, i_]*1.1] = pr_q99[m_, i_]*(1.1+np.random.uniform(-1, 1)/100)
                pr_emus_v02[m_::12, i_]             = pr_tmp

            joblib.dump(pr_emus_v02, precip_emu_path.joinpath(f'pr_full-emus_{model_id}_{ssp_id}_{i_run}_v1.pkl'))

