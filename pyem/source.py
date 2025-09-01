import re
import glob
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pyf3d import YesNo, Fall3DBatch, Fall3DInputFile
from scipy.stats import binned_statistic

class Emulator:

    def __init__(self, 
                 basefile, 
                 start:datetime.datetime, 
                 hours:int, 
                 heights:np.array,
                 #stations:xr.Dataset,
                 base_dir:str = 'mnt/runs',
                 name = "emulator_test",
                 path_fall3d = "/home/talfan/Software/Fall3D_local/fall3d/bin/Fall3d.r8.x",
                ):

        self.basefile = basefile

        self.start = start

        self.hours = hours

        self.heights = heights

        self.base_dir = base_dir
        
        self.name = name

        self.path_fall3d = path_fall3d
        
        source_starts = range(hours-1)

        # outer product
        source_starts_grid, heights_grid = np.meshgrid(source_starts,heights)

        

        df = pd.DataFrame({
                        'source.source_start' : source_starts_grid.flatten(),
                        'source.height_above_vent' : heights_grid.flatten()
                    })
        df['source.source_end'] = df['source.source_start']+1
        #df['source.source_type'] = 'POINT'

        #meteo_file_names  =  [self.base_dir + "/" + name + "/" + str(i) + "/" + str(i) + "_meteo.nc" for i in range(len(df))]

        #df['meteo_data.meteo_data_file'] = meteo_file_names
        #df['meteo_data.meteo_data_file'] = "mnt/runs/test_shared_meteo_data/joint_meteo_test2.nc"

        #df['meteo_data.meteo_data_file'] = self.base_dir + "/" + self.name + "/shared_meteo_data.nc" 

        # make sure everything we don't need is off
        df['model_output.output_3d_concentration'] = YesNo('no')
        df['model_output.output_3d_concentration_bins'] = YesNo('no')
        df['model_output.output_surface_concentration'] = YesNo('yes')
        df['model_output.output_column_load'] = YesNo('yes')
        df['model_output.output_cloud_top'] = YesNo('no')
        df['model_output.output_ground_load'] = YesNo('no')
        df['model_output.output_ground_load_bins'] = YesNo('no')
        df['model_output.output_wet_deposition'] = YesNo('no')
        df['model_output.output_concentrations_at_fl'] = YesNo('no')


        df['ensemble.perturbate_column_height'] = 'NO'
        df['ensemble.perturbate_suzuki_a'] = 'NO'
        df['ensemble.perturbate_suzuki_l'] = 'NO'
        df['ensemble.perturbate_fi_mean'] = 'NO'
        df['ensemble.random_numbers_from_file'] = YesNo('NO')

        
        df['source.source_start'] = df['source.source_start'].astype(str)
        df['source.source_end'] = df['source.source_end'].astype(str)
        df['source.height_above_vent'] = df['source.height_above_vent'].astype(str)

        
        self.df = df

        # ... we then initialise the object ...
        self.batch = Fall3DBatch(name=name, basefile=basefile, df=df, basedir="mnt/runs", path_fall3d=path_fall3d)

        

    def initialise(self):
        # ... initialise the batch - creates the diurectories and input files ...
        print("INITIALISING!!!!!!")
        self.batch.initialise()

    def get_meteo_data(self):
        # ... iterates over the input files and gets the appropriate met data ....
        self.batch.get_meteo_data()

    def run(self):
        # ... iterates over every input file and runs Fall3D
        self.batch.run()

    
    def build_surface_emulator(self):
        
    
        
        # we construct the emulator datarray in blocks of source_start
        # and concatenate them in the last step - this is the list to 
        # hold them as we build them
        dss = []
        
        
        # we build the emulator dataarray
        for source_start, gp in self.df.groupby('source.source_start'):
        
            das = []
        
            for i, r in gp.iterrows():
            
                file ="/".join(["mnt/runs/",self.name, str(i), str(i) + ".res.nc"])
                
                ds = xr.open_dataset(file)
                
                da = ds['SO2_con_surface']
            
                da = da.expand_dims(
                    dim={
                        'height_above_vent':np.array([r['source.height_above_vent']]).astype(float)
                    })
            
                das.append(da)
        
            ds = xr.concat(das,dim='height_above_vent')
        
            #source_start_date = pd.to_datetime(ds_puff["date"].values[0]) + datetime.timedelta(hours = int(source_start))
            source_start_date = self.start + datetime.timedelta(hours = int(source_start))

        
            ds = ds.expand_dims(
                    dim={
                        'source_start':[source_start_date]
                        })
        
            dss.append(ds)
    
        ds = xr.concat(dss,dim='source_start')

        ds = ds.sortby('source_start')
       
        self.da_emulator = ds



    def build_station_emulator(self):

        # first we need a search string to get a list of all of output station files for each run ...
        search_string = "/".join([self.base_dir, self.name, "*", "*.SO2.res"])

        # ... search using that string to get the list of files ...
        files = glob.glob(search_string)

        # ... we regex each file to get an array of the whole path, the index number, and the station id,
        # then put them all into a dataframe ...
        df_results = pd.DataFrame([    
            re.findall(r"(.*/(\d+)\.(.*)\.SO2\.res)",file)[0] for file in files
            ],columns=['file','index','local_id'])

        # ... convert the index to an int so we can join with it later ...
        df_results['index'] = df_results['index'].astype(int)

        # ... and we sort  by the index as the files from glob are in a rabndom order.
        df_results = df_results.sort_values("index")

        # Now, we join the list of file names with the dataframe of run information, specifically
        # the source start and the height above vent, so we can relate each file of concentrations
        # at a given station with the ESPs used to produce it
        df_all = self.df[['source.height_above_vent','source.source_start'	]].join(df_results.set_index('index'))

        # next we need to load the SO2 concentration time series associated with each file.
        # that's quite a complicated step, so we define a dedicated function for it ...
        def get_file(file):
            
            # ... which reads the csv ...
            df = pd.read_csv(
                file,
                delimiter="   ",
                skiprows=7,
                names=['date','load ground','conc. ground','conc pm5 ground','conc pm10 ground','conc pm20 ground'],
                engine='python'
            
            )
            
            # ... formats the time appropriately ...
            df['date'] = df['date'].apply(lambda r: datetime.datetime.strptime(r,"%d%b%Y_%H:%M"))
            
            # ... and sets the appropriate data type ...
            for name in ['load ground','conc. ground','conc pm5 ground','conc pm10 ground','conc pm20 ground']:
            
                df[name] = df[name].astype(float)
            
            # ... before returning the data within that file as a dataframe:
            return(df)

        # .... we then store all the data in a list ...
        dfs = []

        # ... by iterating over every row in the joined dataframe containing the ESPs and the filenames ...
        for i, r in df_all.iterrows():

            # ... loading the data in the file using the function we just defined ...
            df_file  = get_file(r['file'])

            # ... and adding the ESPs and the station id as a separate column ....
            df_file['source.height_above_vent'] = r['source.height_above_vent']
            df_file['source.source_start'] = r['source.source_start']
            df_file['local_id'] = r['local_id']
            dfs.append(df_file)

        # ... so that when we concat all the station concentration data into one giant dataframe.
        dfs = pd.concat(dfs).reset_index(drop=True)

        # We fix the ESP data types ...
        dfs['source.source_start'] = dfs['source.source_start'].astype(int)

        dfs['source.height_above_vent'] = dfs['source.height_above_vent'].astype(float)

        # ... and convert the dataframe to a dataset, setting the ESPs and the station id to be the index ...
        da_puff = dfs.set_index(['source.source_start','local_id','date','source.height_above_vent'])['conc. ground'].to_xarray()

        # ... we need to remember to sort by the ESPs so that when we export the array to Stan it is ordered as we expect it ... 
        da_puff = da_puff.sortby('source.source_start').sortby('source.height_above_vent')

        da_puff = da_puff.sortby("date")

        self.da_puff = da_puff


    
    def build_col_mass_emulator(self):
        
    
        
        # we construct the emulator datarray in blocks of source_start
        # and concatenate them in the last step - this is the list to 
        # hold them as we build them
        dss = []
        
        
        # we build the emulator dataarray
        for source_start, gp in self.df.groupby('source.source_start'):
        
            das = []
        
            for i, r in gp.iterrows():
            
                file ="/".join(["mnt/runs/",self.name, str(i), str(i) + ".res.nc"])
                
                ds = xr.open_dataset(file)
                
                da = ds['SO2_col_mass']
            
                da = da.expand_dims(
                    dim={
                        'height_above_vent':np.array([r['source.height_above_vent']]).astype(float)
                    })
            
                das.append(da)
        
            ds = xr.concat(das,dim='height_above_vent')
        
            #source_start_date = pd.to_datetime(ds_puff["date"].values[0]) + datetime.timedelta(hours = int(source_start))
            source_start_date = self.start + datetime.timedelta(hours = int(source_start))
    
        
            ds = ds.expand_dims(
                    dim={
                        'source_start':[source_start_date]
                        })
        
            dss.append(ds)
    
        ds = xr.concat(dss,dim='source_start')
    
        ds = ds.sortby('source_start')
    
        self.da_col_mass_emulator = ds
                
        
        
    def estimate(self, esps:pd.DataFrame):
    
        # get zero datarray with the correct dims and coords
        total = self.da_emulator.isel(source_start=0, height_above_vent=0).copy()*0.0
    
        for i, r in esps.iterrows():
    
            s = r['source_start']
    
            h = r['height_above_vent']
    
            f = r['flux']
    
            increment = (
                                self.da_emulator
                                .sel({'source_start':s})
                                .interp(
                                    {'height_above_vent':h},
                                    method='linear'
                                    #method='cubic'
                                ) * f 
                        )
    
            total = total + increment
    
        return total

    def estimate_stations(self, esps:pd.DataFrame):
    
        # get zero datarray with the correct dims and coords
        total = self.da_puff.isel({"source.source_start":0, "source.height_above_vent":0}).copy()*0.0
    
        for i, r in esps.iterrows():
    
            s = r['source_start']
    
            h = r['height_above_vent']
    
            f = r['flux']
    
            increment = (
                            self
                                .da_puff
                                .sel({'source.source_start':s})
                                .interp(
                                        {'source.height_above_vent':h},
                                        method='linear'
                                        #method='cubic'
                                        ) * f 
                        )
    
            total = total + increment
    
        return total
        

    def get_random_test_esp(self, height_low, height_high, flux_low, flux_high):
        # ... and for each test we will create a random time series of
        # plume heights and fluxes. We start by gettingt the source
        # starting times covered by the emulator as a dataframe ...
        df_esp = self.da_emulator.source_start.to_dataframe().copy()

        # ... and we create a time series of random plume heights
        # and fluxes, one for each value of source_start ...
        num_source_start = len(df_esp)

        df_esp['height_above_vent'] = np.random.uniform(
                    low=height_low,
                    high=height_high,
                    size=num_source_start
                )

        df_esp['flux'] = np.random.uniform(
                    low=flux_low,
                    high=flux_high,
                    size=num_source_start
                )
        
        df_esp = df_esp.reset_index(drop=True)


        # ... and the source start and stop as integers, in addition
        # to datetimes ...
        df_esp['source_start_hour'] = range(num_source_start)

        df_esp['source_end_hour'] = df_esp['source_start_hour'] +1

        return(df_esp)
    
    def get_random_test_esps(self, num_tests, height_low, height_high, flux_low, flux_high):
         # create an empty list to hold the dataframes
        # we will create (one for each run)
        all_esps = []

        # now we iterate over each test ...
        for n in range(num_tests):

            df_esp = self.get_random_test_esp(height_low, height_high, flux_low, flux_high)

            # ... we also need to remember the run number ...
            df_esp['run'] = n

            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # Fall3D doesn't like more than ~ 24 separate
            # source terms in the .inp file, so we drop the rest here
            # LOOK INTO IF THIS CAN BE FIXED
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #df_esp = df_esp[df_esp['source_start_hour']<24]

            # ... lastly, we append the dataframe for this test run
            # to the list for all runs
            all_esps.append(df_esp)

        # concatenate all test runs into one dataframe
        df_all_esps = pd.concat(all_esps).reset_index(drop=True)

        return(df_all_esps)
    
    def convert_esps_to_runs(self,df_all_esps):

        # now we have the test runs in a single dataframe , we need
        # to transform it into the appropriate format for use by the
        # Fall3DBatch object ..
        runs = []

        # ... we will wind up with a new dataframe with one entry per test run...
        for run, gp in df_all_esps.groupby('run'):

            # ... for each test run we need the list of source.start_hour as a string ...
            source_start_hour_as_string = " ".join(gp['source_start_hour'].astype(str))

            # ... the list of source end hour as a string ...
            source_end_hour_as_string = " ".join(gp['source_end_hour'].astype(str))

            # ... the list of SO2 fluxes as a string ....
            fluxes_as_string = " ".join(gp['flux'].astype(str))

            # ... and the list of plume heigyhts as a string ...
            height_above_vent_as_string = " ".join(gp['height_above_vent'].astype(str))

            # ... which we put into a dict ...
            run_dict = {
                #'run':run,
                'source.source_start': source_start_hour_as_string,
                'source.source_end':source_end_hour_as_string,
                'source.mass_flow_rate': fluxes_as_string,
                'source.height_above_vent':height_above_vent_as_string
            }

            # ... and append to our list of runs ...
            runs.append(run_dict)

        # ... which we concatenate to get a list of runs approipriate for 
        # # Fall3DBatch 
        df_runs = pd.DataFrame(runs)

        #df_runs['meteo_data.meteo_data_file'] = "mnt/runs/test_shared_meteo_data/joint_meteo_test2.nc"
        #df_runs['meteo_data.meteo_data_file'] = self.base_dir + "/" + self.name + "/shared_meteo_data.nc" 

        # and finally, we need to  make sure everything we don't need is off
        df_runs['model_output.output_3d_concentration'] = YesNo('no')
        df_runs['model_output.output_3d_concentration_bins'] = YesNo('no')
        df_runs['model_output.output_surface_concentration'] = YesNo('yes')
        df_runs['model_output.output_column_load'] = YesNo('yes')
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

    def run_tests(self, df_runs, num_tests):
        # ... then we initialise the 
        self.batch = Fall3DBatch(
            name=self.name + "_diagnostics", 
            basefile=self.basefile, 
            df=df_runs, 
            basedir="mnt/runs", 
            path_fall3d=self.path_fall3d
            )
        
        self.batch.initialise()

        self.batch.get_meteo_data()

        self.batch.run() 

        das = []

        # load the test data
        for i in range(num_tests):
            
            path = "/".join([
                "mnt",
                "runs",
                self.name + "_diagnostics", 
                str(i),
                str(i)+".res.nc"
                ])

            da = xr.open_dataset(path)['SO2_con_surface']

            da = da.expand_dims(dim={'num':[i]})

            das.append(da)
        
        da = xr.concat(das, dim='num')

        da.name = 'fall3d surface SO2'


        return da   




    def emulate_tests(self, df_esps, num_tests):

        das = []

        for i in range(num_tests):

            df = df_esps[df_esps['run']==i]
            
            da = self.estimate(df)

            da = da.expand_dims(dim={'num':[i]})

            das.append(da)

        da = xr.concat(das, dim='num')

        da.name = 'emulated surface SO2'

        return da   

                        
                    

    def get_emulator_diagnostics(self, 
                    num_tests=1, 
                    height_low=None, 
                    height_high=None,
                    flux_low=45.0, 
                    flux_high=85.0,

                    
                    ):
        """Calculates diagnostics for the emulator
        """
        
        # if height_low or height_high aren't  specified
        # by the user we default to the min and max values
        # used to build the emulator
        if height_low is None:
            height_low = min(self.heights)

        if height_high is None:
            height_high = max(self.heights)

        # get random ESPs ...
        self.df_esps = self.get_random_test_esps(num_tests, height_low, height_high, flux_low, flux_high)

        # ... convert those random ESPs to a dataframe fo runs for Fall3D Batch ...
        self.df_runs = self.convert_esps_to_runs(self.df_esps)

        # ... run the tests and load the results ...
        ds_tests_fall3d = self.run_tests(self.df_runs, num_tests)

        # ... get the emulated output for the random ESPs ...
        ds_tests_emulated = self.emulate_tests(self.df_esps, num_tests)

        self.ds_tests = xr.merge([
                                    ds_tests_fall3d,
                                    ds_tests_emulated
                                ])

    def plot_emulator_diagnostics(self):
                
        xx = self.ds_tests['fall3d surface SO2'].isel(time=slice(0,24)).values.flatten()
        yy = self.ds_tests['emulated surface SO2'].isel(time=slice(0,24)).values.flatten()

        logbins = np.exp(
                    np.linspace(
                    start=np.log(1e-6),
                    stop = np.log(xx.max()),
                    num = 20
                    )
                )
        logbin_centers = (logbins[:-1] + logbins[1:])/2


        residuals = (yy -xx)**2


        mean_stat = binned_statistic(xx, residuals, 
                                    statistic='mean', 
                                    bins=logbins, 
                                    )

        percentages  = 100*(mean_stat.statistic**.5)/(logbin_centers)


        fig, axs  = plt.subplots(3,1,figsize=(5,10),sharex=True)


        axs[0].scatter(
            xx,
            yy,
            marker='.'
        )
        #plt.plot([0,max(yy)],[0,max(yy)])
        axs[0].set_ylabel("Emulator\n$SO_2$ $\mu g m^{-3}$ ")
        #axs[0].set_xscale("log")
        axs[0].set_yscale("log")

        axs[1].scatter(
                logbin_centers,
                mean_stat.statistic**.5
            )
        axs[1].set_xscale("log")
        axs[1].set_ylabel("Emulator RMS error\n$SO_2$ $\mu g m^{-3}$")



        axs[2].scatter(
            logbins[:-1],
            percentages
        )
        axs[2].set_xscale("log")
        axs[2].set_xlabel("$SO_2$  $\mu g m^{-3}$")
        axs[2].set_ylabel("Emulator RMS error\n% of total")

        axs[0].set_xlim([1e-7,1e0])
        axs[0].set_ylim([1e-7,1e0])
        axs[2].set_xlabel("Fall3D\n$SO_2$  $\mu g m^{-3}$")



        axs[0].plot(
            [min(xx),max(xx)],
            [min(yy), max(yy)],
            color='r'
        )
    
    
    def to_netcdf(self, filename):
        """Saves emulator as netcdf file
        """
        
    #    ds_batch = self.df.to_xarray()
    #    das = []
    #    for name in ds_batch:
    #        da = ds_batch[name]
    #        da.name = name.replace(".","___")
    #        das.append(da.astype(str))
        
    #    ds_batch = xr.merge(das)
        
        emulator_data = xr.merge([
            self.da_emulator,
            self.da_col_mass_emulator,
            self.da_puff,
            self.ds_tests         
        ])
        
        emulator_data.attrs = {
            "base_file":self.basefile.to_string(),
            'start':str(self.start),
            'hours':self.hours,
            'heights' :self.heights,
            'name' :self.name,
            'path_fall3d':self.path_fall3d
        
        }
        
        emulator_data.to_netcdf(filename)
    
        return emulator_data
    
    
    @classmethod
    def from_netcdf(cls, filename):
        """Loads emulator from netcdf file
        """
    
        ds = xr.open_dataset(filename)
    
        basefile_as_string = ds.attrs['base_file']
        
        basefile = Fall3DInputFile.from_string(basefile_as_string)
    
        start_as_string = ds.attrs['start']
        
        start = datetime.datetime.fromisoformat(start_as_string)
    
        heights = ds.attrs['heights']
    
        hours = ds.attrs['hours']
    
        name = ds.attrs['name']
    
        path_fall3d = ds.attrs['path_fall3d']
    
        emulator = cls(
            basefile = basefile,
            start = start,
            hours = hours,
            heights = heights,
            name = name,
            path_fall3d = path_fall3d
        )

        emulator.ds_tests = ds[['fall3d surface SO2', 'emulated surface SO2']]

        emulator.da_puff = ds['conc. ground']

        emulator.da_emulator = ds['SO2_con_surface']

        emulator.da_col_mass_emulator = ds['SO2_col_mass']


    
        return(emulator)
