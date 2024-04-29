# mesmer-m-tp

# code for calibrating mesmer-m-tp and using the calibrated parameters to generate emulations 

File structure: 
./config 
    contains settings and parameters used to generate emulations 
    path variables need to be set correctly in order for the scripts 
    to work properly 
./data 
    contains data necessary to understand the scripts & results
./glm_emulations
    contains all the code that is directly related to mesmer-m-tp
    calibrate: 
        code required to calibrate the deterministic fraction of the precipitation signal
        and to calibrate the parameters required for sampling variability
     emulate: 
        scripts for using the calibrated parameters to generate additional precipitaiton
        realisations
        the emulations can be based off actual esm data (_from_esm_) or can be based off
        emulated temperature data (from-emulated-tas) 
./gmt & ./simple_tas_emulations
    contains code that is only relevant in order to emulate gmt & temperature data;
    these scripts were only used to assess the full uncertainty/error in the 
    coupled emulation chain 
./train & ./utils
    helpfer scripts for ./glm_emulations
    
    
