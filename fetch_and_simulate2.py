import os
import re
import glob
import copy
import uuid
import json
import urllib
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pyf3d import TimeUTC, Grid, MeteoData, Source, Fall3DInputFile, CARRASource, Fall3DBatch, YesNo
import cartopy.crs as ccrs
import nest_asyncio
# hack to get Stan to work
nest_asyncio.apply()
#import cmdstanpy
import arviz as az
import dask.array as da
# seed fpor test 5
seed = 39864
rng = np.random.default_rng(seed)



class Emulator2:

    def __init__(self, name, file, num_runs, height_low, height_high, flux_low, flux_high, start_date, end_date, duration):

        self.name = name

        self.file = file
        
        self.height_low = height_low
        
        self.height_high = height_high

        self.flux_low = flux_low

        self.flux_high = flux_high

        self.num_runs = num_runs

        self.start_date = start_date

        self.end_date = end_date

        self.interval_in_hours = (self.end_date - self.start_date).days * 24

        self.duration = duration


        
    def get_runs_dataframe(self):


        runs  = [self.get_random_run() for n in range(self.num_runs)]
                
        df_runs = pd.DataFrame(runs)
        
        #df_runs['meteo_data.meteo_data_file'] = "mnt/runs/test_shared_meteo_data/joint_meteo_test2.nc"
        df_runs['meteo_data.meteo_data_file'] = [ 
            os.path.join("mnt/runs/", self.name ,str(i), str(i)+"_meteo.nc") for i in range(len(df_runs))
            
            ]

        # make sure everything we don't need is off
        df_runs['model_output.output_3d_concentration'] = YesNo('no')
        df_runs['model_output.output_3d_concentration_bins'] = YesNo('no')
        df_runs['model_output.output_surface_concentration'] = YesNo('yes')
        df_runs['model_output.output_column_load'] = YesNo('no')
        df_runs['model_output.output_cloud_top'] = YesNo('no')
        df_runs['model_output.output_ground_load'] = YesNo('no')
        df_runs['model_output.output_ground_load_bins'] = YesNo('no')
        df_runs['model_output.output_wet_deposition'] = YesNo('no')
        df_runs['model_output.output_concentrations_at_fl'] = YesNo('no')
        
        df_runs['ensemble.perturbate_column_height'] = 'NO'
        df_runs['ensemble.perturbate_suzuki_a'] = 'NO'
        df_runs['ensemble.perturbate_suzuki_l'] = 'NO'
        df_runs['ensemble.perturbate_fi_mean'] = 'NO'
        df_runs['ensemble.random_numbers_from_file'] = YesNo('NO')

        df_runs['ensemble_postprocess.postprocess_median']=YesNo('no')

        return(df_runs)

    def get_random_run(self):

      
        random_hour = rng.integers( low=0, high=self.interval_in_hours)

        random_date  = self.start_date + datetime.timedelta(hours=int(random_hour))

        year = random_date.year

        month = random_date.month
        
        day = random_date.day
        
        run_start= random_date.hour 

        run_end = random_date.hour  + self.duration
            
        source_start = np.arange(
                            run_start,
                            run_end-1
                        )
        
        num_source_start = len(source_start)
        
        source_end = source_start + 1
        
        height_above_vent = np.random.uniform(
                                low=self.height_low,
                                high=self.height_high,
                                size=num_source_start
                            )
        
        flux = np.random.uniform(
                                low=self.flux_low,
                                high=self.flux_high,
                                size=num_source_start
                            )
        
        source_start_as_string = " ".join(source_start.astype(str))
        
        source_end_as_string = " ".join(source_end.astype(str))
        
        fluxes_as_string = " ".join(flux.astype(str))
        
        height_above_vent_as_string = " ".join(height_above_vent.astype(str))
        
        run_dict = {
                #'run':run,
                'source.source_start': source_start_as_string,
                'source.source_end':source_end_as_string,
                'source.mass_flow_rate': fluxes_as_string,
                'source.height_above_vent':height_above_vent_as_string,
                'time_utc.year':year,
                'time_utc.month':month,
                'time_utc.day':day,
                'time_utc.run_start':run_start,
                'time_utc.run_end':run_end,
                'meteo_data.dbs_begin_meteo_data':run_start,
                'meteo_data.dbs_end_meteo_data':run_end
            }
    
        return(run_dict)

    

# Alternatively, we can load an exisiting file, inspect it, and modify it
# to suit our purposes.
# we load the example  Fall3D file into memory as an object
file = Fall3DInputFile.from_file("mnt/examples/default_so2_reykjanes2.inp")

file.time_utc.update({
    'year':2021,
    'month':4,
    'day':1,
    'run_start':0,
    'run_end':24,
})

file.meteo_data.update({
    'dbs_end_meteo_data':24,
    'meteo_coupling_interval':3*60,
    'meteo_data_file':'mnt/runs/tests3/test3.nc',
    'meteo_data_dictionary_file':'/home/talfan/Software/Fall3D/RUNS/CARRA.tbl',
    'meteo_levels_file':"/home/talfan/Software/Fall3D/RUNS/L137_ECMWF.levels"
    
})
file.model_output.update({
    'output_track_points_file':"/home/talfan/Documents/projects/dt-geo/SS5401/SS5401_simple_bayes/SS5401/stations2.pts",#'mnt/runs/tests/stations.pts'
})

#file.model_physics.update({
    #'cfl_safety_factor':0.1,
    #'cfl_criterion':'ONE_DIMENSIONAL'
#    'limiter':'MINMOD'
#})

file.emsemble_postprocess.update({
    'postprocess_median':YesNo('no')
})

file.source.update({
    'mass_flow_rate':"1.0"
})


# Get random starting dates


start_date = datetime.datetime(year=2022,month=1,day=1)

end_date = datetime.datetime(year=2022,month=12,day=31)

name = "20210401"

em2 = Emulator2(
                name = name,
                num_runs = 2,
                file = file,
                height_low = 125.0,
                height_high = 2000.0,
                flux_low = 45.0,  
                flux_high = 200.0,
                start_date=start_date,
                end_date=end_date,
                duration=24
        )

df_runs = em2.get_runs_dataframe()

#df_runs.to_csv("test.csv")
print(df_runs)
batch = Fall3DBatch(
    name=name, 
    basefile=em2.file, 
    df=df_runs, 
    basedir="mnt/runs", 
    path_fall3d="/home/talfan/Software/Fall3D_local/fall3d/bin/Fall3d.r8.x"
)

batch.initialise()

batch.get_meteo_data()

batch.run()
