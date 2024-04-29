#%%
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

from train.glm_general import converting_precipitation

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
from statsmodels.nonparametric.smoothers_lowess import lowess
# local trend 
from sklearn.metrics.pairwise   import haversine_distances

from statsmodels.api import GLM 
import statsmodels.api as sm

#%%

if __name__ == '__main__':
    for model_id in cset.model_ids:
        run_id_training   = cset.model_training_mapping[model_id]
        precip_emutrend_path   = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/loc_trend/{model_id}/')
        
        tmp_files         = [f for f in os.listdir(precip_emutrend_path) if not 'full' in f]
        
        precip_emuvar_path  = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/loc_var/{model_id}/')
        new_samples         = joblib.load(precip_emuvar_path.joinpath(f'pr_var_{model_id}_{run_id_training}_v0.pkl'))[:100, :, :]
        
        for f in tqdm(tmp_files, total = len(tmp_files)): 
            run_id = f.split('_')[4]
            i_run  = int(run_id.split('i')[0][1:])
            ssp_id = f.split('_')[3]
            
            pr_trend_emu = joblib.load(precip_emutrend_path.joinpath(f'{f}'))
            
            if ssp_id == 'historical':
                pr_full = pr_trend_emu*np.exp(new_samples[i_run, :165*12, :])
            else: 
                pr_full = pr_trend_emu*np.exp(new_samples[i_run, 165*12:, :])
    
            # store full predictions 
            precip_emu_path   = cset.output_path.joinpath(f'glm_emus_v02/emus/pr/full/{model_id}/')
            Path(precip_emu_path).mkdir(parents = True, exist_ok = True)
            joblib.dump(pr_full, precip_emu_path.joinpath(f'pr_esm-emus_{model_id}_{ssp_id}_{run_id}_v0.pkl'))
