import os
import re
import json
import copy
import glob
import urllib
import cdsapi
import cfgrib
import datetime
import subprocess
import numpy as np
import arviz as az
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from tqdm import tqdm
from .fstrings import *
from .regex_strings import *
from multiprocessing.pool import ThreadPool
from cartopy.feature import NaturalEarthFeature
from scipy.stats import binned_statistic
# hack to get Stan to work
import nest_asyncio
nest_asyncio.apply()
import stan





def is_valid_date(year:int, month:int, day:int)->bool:
    """Function by 'Anon' that checks a date to see if it exists
    https://stackoverflow.com/a/51981596 
    """    
    day_count_for_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    if year%4==0 and (year%100 != 0 or year%400==0):
        day_count_for_month[2] = 29
    
    return (1 <= month <= 12 and 1 <= day <= day_count_for_month[month])


class YesNo:
    """Class to associate various spellings of "yes" and "no" with True and False
    """
    
    def __init__(self,val:str|float):
        
        if val in ['YES','Yes','yes']:
            self.bool = True
            self.val = val
            
        elif val in ['NO','No','no']:
            self.bool = False
            self.val = val
            
        elif val == True:
            self.bool = True
            self.val = 'Yes'
            
        elif val == False:
            self.bool = False
            self.val = 'No'
            
        # the class needs to be able to be initialised from another
        # instance for easy program flow
        elif type(val) == type(self):
            self.bool = val.bool
            self.val = val.val
            
        else:
            raise TypeError(val," not one of Yes, No, True, False")
    
    def __repr__(self):
        # the value of the YesNo object that is inserted into the fstring
        return(self.val)
    
    def __str__(self):
        # the value of the YesNo object that is inserted into the fstring
        return(self.val)
    
    def __bool__(self):
        # the value of the YesNo object that is used for logical operations
        return(self.bool)
    


class OnOff:
    """Class to associate various spellings of "On" and "Off" with True and False
    """
    
    def __init__(self,val:str|float):
        """
        """
        
        if val in ['ON','On','on']:
            self.bool = True
            self.val = val
            
        elif val in ['OFF','Off','off']:
            self.bool = False
            self.val = val
            
        elif val == True:
            self.bool = True
            self.val = 'On'
            
        elif val == False:
            self.bool = False
            self.val = 'Off'
            
        # the class needs to be able to be initialised from another
        # instance for easy program flow
        elif type(val) == type(self):
            self.bool = val.bool
            self.val = val.val
            
        else:
            raise TypeError(val,"not one of On, Off, True, False")
    
    def __repr__(self):
        """
        """
        # the value of the OnOff object that is inserted into the fstring
        return(self.val)
    
    def __str__(self):
        """
        """
        # the value of the YesNo object that is inserted into the fstring
        return(self.val)
    
    def __bool__(self):
        """
        """
        # the value of the OnOff object that is used for logical operations
        return(self.bool)

class Longitude:
    """Class to remove annoying, hard to debug longitude definition errors
    """

    def __init__(self, value:float, type:int):

        
        if type == 360:
            if value<0:
                raise TypeError("Longitude cannot be negative when defined 0-360")
            else:
                self._360 = value
                self._180 = (value + 180) % 360 - 180

        if type == 180:
            if value>180:
                raise TypeError("Longitude cannot be > 180 when defined -180 to 180")
            else:
                self._360 = value %360
                self._180 = value


def Lon360(value:float)->Longitude:
    """Quicker to type than Longitude(value,360)
    """
    return(Longitude(value, 360))
    

def Lon180(value:float)->Longitude:
    """Quicker to type than Longitude(value, 180)
    """
    return(Longitude(value, 180))





class Section:
    """Generic class for a Fall3D input file section. 
    All sections need to be able to do 1 of 3 things:
    
    (1) create new from string from valid Fall3D input file
    (2) write to a string that is a valid part of a Fall3D input file
    (3) update values while performing all initialization type and value checks all over again
    
    plus domain specific visualisations, etc.., which are implemented in the child classes
    
    """
    @classmethod    
    def from_string(cls,string:str):
        """
        """
        
        # get variables from string
        variables = re.search(cls.regex, string,re.DOTALL).groupdict()
        
        # initialise a dict to hold the variables after we turn them into
        # the appropriate types
        variables_typed = {}
        
        # 
        for key, item in variables.items():
        
            variables_typed[key] = cls.types[key](item)
        
        cls = cls(**variables_typed)
        
        return(cls)
    
    def to_string(self):
        """
        """
        
        variables = {}
        
        for key in self.types.keys():
            
            variables[key] = getattr(self,key)
        
        string = self.fstring.format(**variables)
        
        return(string)


    def __repr__(self):
        """
        """
        return(self.to_string())

    def __str__(self):
        """
        """
        return(self.to_string())


    def update(self,new_values:dict):
        """Update witgh values in dict
        """
        # createb a dict that wil be used to reinitialise theb object
        params = {}

        for key, item in self.types.items():
            
            params[key] = getattr(self,key)

        for key, item in new_values.items():

            params[key] = item

        self.__init__(**params)


        




class TimeUTC(Section):
    
    fstring = fstring_time_utc
    
    regex = regex_time_utc
    
    types = {
        'year':int,
        'month':int,
        'day':int,
        'run_start':int,
        'run_end':int,
        'initial_condition':str,
        'restart_file':str,
        'restart_ensemble_basepath':str
    }
    
    
    
    def __init__(self, 
                year:int, 
                month:int, 
                day:int, 
                run_start:int, 
                run_end:int, 
                initial_condition:str, 
                restart_file:str, 
                restart_ensemble_basepath:str
                ):
        
        #---------------------------------------------------------------------------------
        # TEST INPUTS
        #---------------------------------------------------------------------------------
        

        # tests for year
        if (year<1970) or (year>2024):
            raise ValueError('Year must be >= 1970 and <= 2024')

        if type(year) != int:
            raise TypeError("Year must be type int")


        # tests for month
        if (type(month) !=int):
            raise TypeError("Month must be type int")

        if (month<1) or (month>12):
            raise TypeError("Month must be >= 1 and <= 12")
        
        # tests for day
        if (type(day) !=int):
            raise TypeError("Day must be type int")

        if (day<1) or (day>31):
            raise TypeError('Day must be between 1 and 31')

        # test for whole date
        assert (is_valid_date(year,month,day)), ["/"].join(str(year),str(month),str(day))+" is not a valid date."
        
        # tests for initial_condition
        assert (type(initial_condition)==str), "initial_conditions must be type str"
        assert (initial_condition in ['NONE', 'RESTART', 'INSERTION']), "initial_condition must be one of NONE, RESTART, INSERTION"
        
        
        # tests for restart_file
        assert (type(restart_file)==str), "restart_file must be type string"
        
        # tests for restart_ensemble_basepath
        assert (type(restart_ensemble_basepath)==str), "restart_ensemble_basepath must be type string"
        
        #---------------------------------------------------------------------------------       
        # OPTIONS THAT HAVE BECOME ACTIVATED
        #---------------------------------------------------------------------------------

        if initial_condition in ['RESTART']:
            print("RESTART_FILE in use as INITIAL_CONDITION = RESTART  )")
            
        self.year = year
        self.month = month
        self.day = day
        self.run_start = run_start
        self.run_end = run_end
        self.initial_condition = initial_condition
        self.restart_file = restart_file
        self.restart_ensemble_basepath = restart_ensemble_basepath  
        




class InsertionData(Section):
    
    fstring = fstring_insertion_data
    
    regex = regex_insertion_data
    
    types = {
        'insertion_file':str,
        'insertion_dictionary_file':str,
        'insertion_time_slab':int,
        'diameter_cut_off':int
    }
    
    def __init__(self,
                insertion_file:str, 
                insertion_dictionary_file:str, 
                insertion_time_slab:int,
                diameter_cut_off:int
                ):
        
        # TEST INPUTS
        
        # tests for insertion_file
        assert (type(insertion_file)==str), "insertion_file must be string"
        
        # tests for insertion_dictionary_file
        assert (type(insertion_dictionary_file)==str), "insertion_dictionary_file must be string"
        
        
        # tests for insertion_time_slab
        assert (type(insertion_time_slab)==int), "insertion_time_slab must be int"
        
        # tests for diameter_cut_off_mic
        assert (type(diameter_cut_off)==int)
        
        self.insertion_file = insertion_file
        self.insertion_dictionary_file = insertion_dictionary_file
        self.insertion_time_slab = insertion_time_slab
        self.diameter_cut_off = diameter_cut_off
        


class MeteoData(Section):
    
    
    fstring = fstring_meteo
    
    regex = regex_meteo
    
    types={
        
        'meteo_data_format':str,
        'meteo_data_dictionary_file':str,
        'meteo_data_file':str,
        'meteo_ensemble_basepath':str,
        'meteo_levels_file':str,
        'dbs_begin_meteo_data':int,
        'dbs_end_meteo_data':int,
        'meteo_coupling_interval':int,
        'memory_chunk_size':int
    }
    
    def __init__(self, 
                meteo_data_format:str, 
                meteo_data_dictionary_file:str,
                meteo_data_file:str,
                meteo_ensemble_basepath:str,
                meteo_levels_file:str,
                dbs_begin_meteo_data:int,
                dbs_end_meteo_data:int,
                meteo_coupling_interval:int,
                memory_chunk_size:int
                ):

        # TEST INPUTS
        
        # test meteo_data_format
        assert(type(meteo_data_format)==str)
        assert( meteo_data_format in ['WRF', 'GFS' , 'ERA5' , 'ERA5ML' , 'IFS' , 'CARRA'])
        
        # test meteo_data_dictionary
        assert(type(meteo_data_dictionary_file)==str)
        
        # test meteo_data_file
        assert(type(meteo_data_file)==str)
        
        # test meteo_ensemble_basepath
        assert(type(meteo_ensemble_basepath)==str)
        
        # test meteo_levels_file
        assert(type(meteo_levels_file)==str)
        
        # test dbs_begin_meteo_data
        assert(type(dbs_begin_meteo_data)==int)
        assert(dbs_begin_meteo_data>=0)
        
        # test dbs_end_meteo_data
        assert(type(dbs_end_meteo_data)==int)
        assert(dbs_end_meteo_data>=0)
        assert(dbs_end_meteo_data>dbs_begin_meteo_data)
        
        # test meteo_coupling_interval
        assert(type(meteo_coupling_interval)==int)
        assert(meteo_coupling_interval>0)
        
        # test memory_chunk_size
        assert(type(memory_chunk_size)==int)
        assert(memory_chunk_size>0)
        
        # print options that have become activated
        if meteo_data_format in ['ERA5ML', 'IFS']:
            print("METEO_DATA_FORMAT has been set to ERA5ML or IFS so METEO_LEVELS_FILE is in use.")
        
        self.meteo_data_format = meteo_data_format 
        self.meteo_data_dictionary_file = meteo_data_dictionary_file  
        self.meteo_data_file = meteo_data_file 
        self.meteo_ensemble_basepath = meteo_ensemble_basepath 
        self.meteo_levels_file = meteo_levels_file
        self.dbs_begin_meteo_data = dbs_begin_meteo_data
        self.dbs_end_meteo_data = dbs_end_meteo_data
        self.meteo_coupling_interval = meteo_coupling_interval
        self.memory_chunk_size = memory_chunk_size

    def plot_on_map(self,ax=None):

        if not os.path.exists(self.meteo_data_file):
            raise OSError("Meteo data file does not exist.")

        ds = xr.open_dataset(self.meteo_data_file)

        # get outline of meteo data

        west_lat = ds.latitude.sel(x=0).values
        west_lon = ds.longitude.sel(x=0).values

        east_lat = ds.latitude.sel(x=-1).values
        east_lon = ds.longitude.sel(x=-1).values

        south_lat = ds.latitude.sel(y=0).values
        south_lon = ds.longitude.sel(y=0).values

        north_lat = ds.latitude.sel(y=-1).values
        north_lon = ds.longitude.sel(y=-1).values
        
        extent = [
                    ds.longitude.min(), 
                    ds.longitude.max(),
                    ds.latitude.min(), 
                    ds.latitude.max()
                ]

        

        if ax is None:
            plt.figure("Test Map")
            crs = ccrs.PlateCarree()
            ax = plt.subplot(111, projection=crs)
            
            ax.set_extent(extent, crs=crs)
            
            #ax.add_feature(NaturalEarthFeature('physical', 'ocean', '50m'))
            ax.coastlines(resolution='10m',color='blue')
    
            ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)

        ax.plot(west_lon-360, west_lat, color='blue')
        ax.plot(east_lon-360, east_lat, color='blue')
        ax.plot(north_lon-360, north_lat, color='blue')
        ax.plot(south_lon-360, south_lat, color='blue')

        return(extent)
        
        #plt.show()
        





class Grid(Section):
    
    
    fstring = fstring_grid

    
    regex = regex_grid
    
    types={
        'horizontal_mapping':str,
        'vertical_mapping':str,
        'lonmin':float,
        'lonmax':float,
        'latmin':float,
        'latmax':float,
        'nx':int,
        'ny':int,
        'nz':int,
        'zmax':int,
        #'sigma_values':str,
    }
    
    def __init__(self,
                horizontal_mapping:str,
                vertical_mapping:str, 
                lonmin:float, 
                lonmax:float, 
                latmin:float,
                latmax:float, 
                nx:int, 
                ny:int, 
                nz:int, 
                zmax:int#, 
                #sigma_values:str
                ):

        print("WARNING sigma_values_currently_disabled")
        # TEST INPUTS
        
        # test horizontal_mapping
        assert(type(horizontal_mapping)==str)
        assert( horizontal_mapping in ['CARTESIAN', 'SPHERICAL' ])
        
        # test vertical_mapping
        assert(type(vertical_mapping)==str)
        assert( vertical_mapping in ['SIGMA_NO_DECAY', 'SIGMA_LINEAR_DECAY', 'SIGMA_EXPONENTIAL_DECAY'])
        
        # test lonmin
        assert(type(lonmin)==float)
        assert(lonmin>-180)
        assert(lonmin<180)
        
        # test lonmax
        assert(type(lonmax)==float)
        assert(lonmin>-180)
        assert(lonmin<180)       
        
        # check lonmax > lonmin
        assert(lonmax>lonmin)
        
        # test latmin
        assert(type(latmin)==float)
        assert(latmin>-90)
        assert(latmin<90)
                
        
        # test latmax
        assert(type(latmax)==float)
        assert(latmax>-90)
        assert(latmax<90)
        
        # check latmin < latmax
        assert(latmin < latmax)
        
        # test nx
        assert(type(nx)==int)
        assert(nx>1)
        
        #test ny
        assert(type(ny)==int)
        assert(ny>1)
        
        #test nz
        assert(type(nz)==int)
        assert(nz>1)
        
        #test zmax
        assert(type(zmax)==int)
        assert(zmax>0)
        
        #test sigma_values
        #assert(type(sigma_values)==str)
        
        
        
        
        
        
        self.horizontal_mapping = horizontal_mapping
        self.vertical_mapping = vertical_mapping 
        self.lonmin = lonmin 
        self.lonmax = lonmax 
        self.latmin = latmin 
        self.latmax = latmax 
        self.nx = nx 
        self.ny = ny 
        self.nz = nz 
        self.zmax = zmax
        #self.sigma_values =sigma_values

    def plot_on_map(self, ax=None):

        extent = [self.lonmin-1, self.lonmax+1,self.latmin-1, self.latmax+1]

        bbox = np.array([
                    [self.lonmin, self.latmin], # lower left
                    [self.lonmin, self.latmax], # upper left
                    [self.lonmax, self.latmax], # upper right
                    [self.lonmax, self.latmin], # lower right
                    [self.lonmin, self.latmin] # back to lower left again
        ])


        if ax == None:
            plt.figure("Test Map")
            ax = plt.subplot(111, projection=ccrs.PlateCarree())
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            #ax.add_feature(NaturalEarthFeature('physical', 'ocean', '50m'))
            ax.coastlines(resolution='10m',color='blue')
    
            ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)

        ax.plot(*bbox.T, color='red')
        
        #plt.show()





class Species(Section):
    
    fstring = fstring_species
    
    regex = regex_species
    
    types={
        
        'tephra':OnOff,
        'dust':OnOff,
        'h2o':OnOff,
        'so2':OnOff,
        'cs134':OnOff,
        'cs137':OnOff,
        'i131':OnOff,
        'sr90':OnOff,
        'y90':OnOff,
        'h2o_mass_fraction':float,
        'so2_mass_fraction':float,
        'cs134_mass_fraction':float,
        'cs137_mass_fraction':float,
        'i131_mass_fraction':float,
        'sr90_mass_fraction':float,
        'y90_mass_fraction':float
    }
    
    def __init__(self, 
                tephra:OnOff, 
                dust:OnOff, 
                h2o:OnOff, 
                so2:OnOff, 
                cs134:OnOff, 
                cs137:OnOff, 
                i131:OnOff, 
                sr90:OnOff, 
                y90:OnOff, 
                h2o_mass_fraction:float,
                so2_mass_fraction:float, 
                cs134_mass_fraction:float, 
                cs137_mass_fraction:float,
                i131_mass_fraction:float, 
                sr90_mass_fraction:float, 
                y90_mass_fraction:float
                ):
        
        

        assert((type(tephra)==bool)or(type(tephra)==OnOff))
        assert((type(dust)==bool)or(type(dust)==OnOff))
        assert((type(h2o)==bool)or(type(h2o)==OnOff))
        assert(type(h2o_mass_fraction)==float)
        assert((type(so2)==bool)or(type(so2)==OnOff))
        assert(type(so2_mass_fraction)==float)
        assert((type(cs137)==bool)or(type(cs137)==OnOff))
        assert(type(cs137_mass_fraction)==float)
        assert((type(sr90)==bool)or(type(sr90)==OnOff))
        assert(type(sr90_mass_fraction)==float)
        assert((type(y90)==bool)or(type(y90)==OnOff))
        assert(type(y90_mass_fraction)==float)
        
        # check constraints
        # (1) only one of tephra or dust is allowed
        assert(not (tephra and dust))
        
        # (2) if tephra = Off, then mass fraction of aeraosols must sum to 1 (100% apparently)
        if not tephra:
            assert(h2o_mass_fraction+so2_mass_fraction==100)
        
        # (3) cannot run PARTICLES (tephra, dust) and RADIONUCLEIDES at the same time
        assert(
            not (
                    any([tephra, dust]) and any([cs134, cs137, i131, sr90,  y90])
            )
        )
        
        # (4) mass fraction of radionucleides must sum to 1 (100%, apparently)
        if (cs134 or cs137 or i131 or sr90 or y90):
            assert(
                (cs134_mass_fraction + cs137_mass_fraction + 
                 i131_mass_fraction + sr90_mass_fraction + 
                 y90_mass_fraction) == 100
            )
        
        self.tephra = OnOff(tephra)
        self.dust = OnOff(dust) 
        self.h2o = OnOff(h2o)
        self.h2o_mass_fraction = h2o_mass_fraction
        self.so2 = OnOff(so2)
        self.so2_mass_fraction = so2_mass_fraction
        self.cs134 = OnOff(cs134)
        self.cs134_mass_fraction = cs134_mass_fraction
        self.cs137 = OnOff(cs137)
        self.cs137_mass_fraction = cs137_mass_fraction
        self.i131 = OnOff(i131)
        self.i131_mass_fraction = i131_mass_fraction
        self.sr90 = OnOff(sr90)
        self.sr90_mass_fraction = sr90_mass_fraction
        self.y90 = OnOff(y90)
        self.y90_mass_fraction = y90_mass_fraction
               


class TephraTgsd(Section):
    
    fstring = fstring_tephra_tgsd
    
    regex = regex_tephra_tgsd
    
    types = {
        'number_of_bins':int,
        'fi_range1':int, 
        'fi_range2':int,
        'density_range1':int, 
        'density_range2':int,
        'sphericity_range1':float, 
        'sphericity_range2':float,
        'distribution':str,
        'gaussian_fi_mean':float, 
        'gaussian_fi_disp':float,
        'bigaussian_fi_mean1':float, 
        'bigaussian_fi_mean2':float,
        'bigaussian_fi_disp1':float, 
        'bigaussian_fi_disp2':float,
        'bigaussian_mixing_factor':float,
        'weibull_fi_scale':float, 
        'weibull_w_shape':float,
        'biweibull_fi_scale1':float, 
        'biweibull_fi_scale2':float,
        'biweibull_w_shape1':float, 
        'biweibull_w_shape2':float,
        'biweibull_mixing_factor':float,
        'custom_file':str,
        'estimate_viscosity':str,
        'estimate_height_above_vent':str
    }
    
    def __init__(self,
                number_of_bins:int,
                fi_range1:int, 
                fi_range2:int,
                density_range1:int, 
                density_range2:int,
                sphericity_range1:float, 
                sphericity_range2:float,
                distribution:str,
                gaussian_fi_mean:float, 
                gaussian_fi_disp:float,
                bigaussian_fi_mean1:float, 
                bigaussian_fi_mean2:float,
                bigaussian_fi_disp1:float, 
                bigaussian_fi_disp2:float,
                bigaussian_mixing_factor:float,
                weibull_fi_scale:float, 
                weibull_w_shape:float,
                biweibull_fi_scale1:float, 
                biweibull_fi_scale2:float,
                biweibull_w_shape1:float, 
                biweibull_w_shape2:float,
                biweibull_mixing_factor:float,
                custom_file:str,
                estimate_viscosity:str,
                estimate_height_above_vent:str
                ):
        
        assert(type(number_of_bins)==int)
        assert(type(fi_range1)==int)
        assert(type(fi_range2)==int)
        assert(type(density_range1)==int) 
        assert(type(density_range2)==int)
        assert(type(sphericity_range1)==float) 
        assert(type(sphericity_range2)==float)
        assert(type(distribution)==str)
        assert(type(gaussian_fi_mean)==float) 
        assert(type(gaussian_fi_disp)==float)
        assert(type(bigaussian_fi_mean1)==float) 
        assert(type(bigaussian_fi_mean2)==float)
        assert(type(bigaussian_fi_disp1)==float) 
        assert(type(bigaussian_fi_disp2)==float)
        assert(type(bigaussian_mixing_factor)==float)
        assert(type(weibull_fi_scale)==float)
        assert(type(weibull_w_shape)==float)
        assert(type(biweibull_fi_scale1)==float) 
        assert(type(biweibull_fi_scale2)==float)
        assert(type(biweibull_w_shape1)==float)
        assert(type(biweibull_w_shape2)==float)
        assert(type(biweibull_mixing_factor)==float)
        assert(type(custom_file)==str)
        assert(type(estimate_viscosity)==str)
        assert(type(estimate_height_above_vent)==str)
        
        self.number_of_bins = number_of_bins
        self.fi_range1 = fi_range1 
        self.fi_range2 = fi_range2
        self.density_range1 = density_range1
        self.density_range2 = density_range2
        self.sphericity_range1 = sphericity_range1
        self.sphericity_range2 = sphericity_range2
        self.distribution = distribution
        self.gaussian_fi_mean = gaussian_fi_mean  
        self.gaussian_fi_disp = gaussian_fi_disp
        self.bigaussian_fi_mean1 = bigaussian_fi_mean1 
        self.bigaussian_fi_mean2 = bigaussian_fi_mean2
        self.bigaussian_fi_disp1 = bigaussian_fi_disp1 
        self.bigaussian_fi_disp2 = bigaussian_fi_disp2
        self.bigaussian_mixing_factor = bigaussian_mixing_factor
        self.weibull_fi_scale = weibull_fi_scale 
        self.weibull_w_shape = weibull_w_shape
        self.biweibull_fi_scale1 = biweibull_fi_scale1 
        self.biweibull_fi_scale2 = biweibull_fi_scale2
        self.biweibull_w_shape1 = biweibull_w_shape1 
        self.biweibull_w_shape2 = biweibull_w_shape2 
        self.biweibull_mixing_factor = biweibull_mixing_factor
        self.custom_file = custom_file
        self.estimate_viscosity = estimate_viscosity 
        self.estimate_height_above_vent = estimate_height_above_vent
        
        
        
        
        
        
class RadionucleidesTgsd(Section):
    
    fstring = fstring_radionucleides_tgsd

    regex = regex_radionucleides_tgsd
        
    types = {'number_of_bins':int,
            'fi_range1':int,
            'fi_range2':int,
            'density_range1':int,
            'density_range2':int,
            'sphericity_range1':float,
            'sphericity_range2':float,
            'distribution':str,
            'gaussian_fi_mean':float,
            'gaussian_fi_disp':float,
            'bigaussian_fi_mean1':float,
            'bigaussian_fi_mean2':float,
            'bigaussian_fi_disp1':float,
            'bigaussian_fi_disp2':float,
            'bigaussian_mixing_factor':float,
            'weibull_fi_scale':float,
            'weibull_w_shape':float,
            'biweibull_fi_scale1':float,
            'biweibull_fi_scale2':float,
            'biweibull_w_shape1':float,
            'biweibull_w_shape2':float,
            'biweibull_mixing_factor':float,
            'custom_file':str,
            'estimate_viscosity':str,
            'estimate_height_above_vent':str}
    
    def __init__(self,
                number_of_bins:int,
                fi_range1:int,
                fi_range2:int,
                density_range1:int,
                density_range2:int,
                sphericity_range1:float,
                sphericity_range2:float,
                distribution:str,
                gaussian_fi_mean:float,
                gaussian_fi_disp:float,
                bigaussian_fi_mean1:float,
                bigaussian_fi_mean2:float,
                bigaussian_fi_disp1:float,
                bigaussian_fi_disp2:float,
                bigaussian_mixing_factor:float,
                weibull_fi_scale:float,
                weibull_w_shape:float,
                biweibull_fi_scale1:float,
                biweibull_fi_scale2:float,
                biweibull_w_shape1:float,
                biweibull_w_shape2:float,
                biweibull_mixing_factor:float,
                custom_file:str,
                estimate_viscosity:str,
                estimate_height_above_vent:str,
                ):
    
        assert(type(number_of_bins)==int)
        assert(type(fi_range1)==int)
        assert(type(fi_range2)==int)
        assert(type(density_range1)==int)
        assert(type(density_range2)==int)
        assert(type(sphericity_range1)==float)
        assert(type(sphericity_range2)==float)
        assert(type(distribution)==str)
        assert(type(gaussian_fi_mean)==float)
        assert(type(gaussian_fi_disp)==float)
        assert(type(bigaussian_fi_mean1)==float)
        assert(type(bigaussian_fi_mean2)==float)
        assert(type(bigaussian_fi_disp1)==float)
        assert(type(bigaussian_fi_disp2)==float)
        assert(type(bigaussian_mixing_factor)==float)
        assert(type(weibull_fi_scale)==float)
        assert(type(weibull_w_shape)==float)
        assert(type(biweibull_fi_scale1)==float)
        assert(type(biweibull_fi_scale2)==float)
        assert(type(biweibull_w_shape1)==float)
        assert(type(biweibull_w_shape2)==float)
        assert(type(biweibull_mixing_factor)==float)
        assert(type(custom_file)==str)
        assert(type(estimate_viscosity)==str)
        assert(type(estimate_height_above_vent)==str)
    
        self.number_of_bins = number_of_bins
        self.fi_range1 = fi_range1
        self.fi_range2 = fi_range2 
        self.density_range1 = density_range1
        self.density_range2 = density_range2 
        self.sphericity_range1 = sphericity_range1 
        self.sphericity_range2 = sphericity_range2
        self.distribution = distribution 
        self.gaussian_fi_mean = gaussian_fi_mean
        self.gaussian_fi_disp = gaussian_fi_disp
        self.bigaussian_fi_mean1 = bigaussian_fi_mean1 
        self.bigaussian_fi_mean2 = bigaussian_fi_mean2 
        self.bigaussian_fi_disp1 = bigaussian_fi_disp1
        self.bigaussian_fi_disp2 = bigaussian_fi_disp2 
        self.bigaussian_mixing_factor = bigaussian_mixing_factor 
        self.weibull_fi_scale =  weibull_fi_scale
        self.weibull_w_shape = weibull_w_shape 
        self.biweibull_fi_scale1 = biweibull_fi_scale1 
        self.biweibull_fi_scale2 = biweibull_fi_scale2 
        self.biweibull_w_shape1 = biweibull_w_shape1 
        self.biweibull_w_shape2 = biweibull_w_shape2
        self.biweibull_mixing_factor = biweibull_mixing_factor 
        self.custom_file = custom_file 
        self.estimate_viscosity = estimate_viscosity 
        self.estimate_height_above_vent = estimate_height_above_vent
    
    
    
    
class ParticleAggregation(Section):
    
    fstring = fstring_particle_aggregation
    
    regex = regex_particle_aggregation
    
    types = {
        'particle_cut_off':str,
        'aggregation_model':str,
        'number_of_aggregate_bins':int,
        'diameter_aggregates1':float,
        'diameter_aggregates2':float,
        'density_aggregates1':float,
        'density_aggregates2':float,
        'percentage1':float,
        'percentage2':float,
        'vset_factor':float,
        'fractal_exponent':float
    }
    
    def __init__(self,
                particle_cut_off:str,
                aggregation_model:str,
                number_of_aggregate_bins:int,
                diameter_aggregates1:float,
                diameter_aggregates2:float,
                density_aggregates1:float,
                density_aggregates2:float,
                percentage1:float,
                percentage2:float,
                vset_factor:float,
                fractal_exponent:float
                ):
        
        
        assert(type(particle_cut_off)==str)
        assert(type(aggregation_model)==str)
        assert(type(number_of_aggregate_bins)==int)
        assert(type(diameter_aggregates1)==float)
        assert(type(diameter_aggregates2)==float)
        assert(type(density_aggregates1)==float)
        assert(type(density_aggregates2)==float)
        assert(type(percentage1)==float)
        assert(type(percentage2)==float)
        assert(type(vset_factor)==float)
        assert(type(fractal_exponent)==float)
        


        self.particle_cut_off = particle_cut_off
        self.aggregation_model = aggregation_model
        self.number_of_aggregate_bins = number_of_aggregate_bins
        self.diameter_aggregates1 = diameter_aggregates1
        self.diameter_aggregates2 = diameter_aggregates2
        self.density_aggregates1 = density_aggregates1
        self.density_aggregates2 = density_aggregates2 
        self.percentage1 = percentage1 
        self.percentage2 = percentage2 
        self.vset_factor = vset_factor 
        self.fractal_exponent = fractal_exponent
        
        
        
        
        
        
        
     
    
class Source(Section):
    
    fstring = fstring_source
    
    regex = regex_source
    
    types = {
        'source_type':str,
        'source_start':str,
        'source_end':str,
        'lon_vent':float,
        'lat_vent':float,
        'vent_height':float,
        'height_above_vent':str,
        'mass_flow_rate':str,
        'alfa_plume':float,
        'beta_plume':float,
        'exit_temperature':float,
        'exit_water_fraction':float,
        'a1':float,
        'a2':float,
        'l':float,
        'thickness':float,
        'solve_plume_for':str,
        'mfr_search_range1':int,
        'mfr_search_range2':int,
        'exit_velocity':float,
        'exit_gas_water_temperature':float,
        'exit_liquid_water_temperature':float,
        'exit_solid_water_temperature':float,
        'exit_gas_water_fraction':float,
        'exit_liquid_water_fraction':float,
        'exit_solid_water_fraction':float,
        'wind_coupling':YesNo,
        'air_moisture':YesNo,
        'latent_heat':YesNo,
        'reentrainment':YesNo,
        'bursik_factor':float,
        'z_min_wind':float,
        'c_umbrella':float,
        'a_s':str,
        'a_v':str      
    }
    
    def __init__(self,
                source_type:str,
                source_start:str,
                source_end:str,
                lon_vent:float,
                lat_vent:float,
                vent_height:float,
                height_above_vent:str,
                mass_flow_rate:float,
                alfa_plume:float,
                beta_plume:float,
                exit_temperature:float,
                exit_water_fraction:float,
                a1:float,
                a2:float,
                l:float,
                thickness:float,
                solve_plume_for:str,
                mfr_search_range1:int,
                mfr_search_range2:int,
                exit_velocity:float,
                exit_gas_water_temperature:float,
                exit_liquid_water_temperature:float,
                exit_solid_water_temperature:float,
                exit_gas_water_fraction:float,
                exit_liquid_water_fraction:float,
                exit_solid_water_fraction:float,
                wind_coupling:YesNo,
                air_moisture:YesNo,
                latent_heat:YesNo,
                reentrainment:YesNo,
                bursik_factor:float,
                z_min_wind:float,
                c_umbrella:float,
                a_s:str,
                a_v:str      
                ):
    
        assert(type(source_type)==str)
        assert(type(source_start)==str)
        assert(type(source_end)==str)
        assert(type(lon_vent)==float)
        assert(type(lat_vent)==float)
        assert(type(vent_height)==float)
        assert(type(height_above_vent)==str)
        assert(type(mass_flow_rate)==str)
        assert(type(alfa_plume)==float)
        assert(type(beta_plume)==float)
        assert(type(exit_temperature)==float)
        assert(type(exit_water_fraction)==float)
        assert(type(a1)==float)
        assert(type(a2)==float)
        assert(type(l)==float)
        assert(type(thickness)==float)
        assert(type(solve_plume_for)==str)
        assert(type(mfr_search_range1)==int)
        assert(type(mfr_search_range2)==int)
        assert(type(exit_velocity)==float)
        assert(type(exit_gas_water_temperature)==float)
        assert(type(exit_liquid_water_temperature)==float)
        assert(type(exit_solid_water_temperature)==float)
        assert(type(exit_gas_water_fraction)==float)
        assert(type(exit_liquid_water_fraction)==float)
        assert(type(exit_solid_water_fraction)==float)
        assert(type(wind_coupling)==YesNo)
        assert(type(air_moisture)==YesNo)
        assert(type(latent_heat)==YesNo)
        assert(type(reentrainment)==YesNo)
        assert(type(bursik_factor)==float)
        assert(type(z_min_wind)==float)
        assert(type(c_umbrella)==float)
        assert(type(a_s)==str)
        assert(type(a_v)==str)     
        
  
        self.source_type = source_type
        self.source_start = source_start
        self.source_end = source_end
        self.lon_vent = lon_vent
        self.lat_vent = lat_vent
        self.vent_height = vent_height
        self.height_above_vent = height_above_vent
        self.mass_flow_rate = mass_flow_rate
        self.alfa_plume = alfa_plume
        self.beta_plume = beta_plume
        self.exit_temperature = exit_temperature
        self.exit_water_fraction = exit_water_fraction
        self.a1 = a1
        self.a2 = a2
        self.l = l
        self.thickness = thickness
        self.solve_plume_for = solve_plume_for
        self.mfr_search_range1 = mfr_search_range1
        self.mfr_search_range2 = mfr_search_range2
        self.exit_velocity = exit_velocity
        self.exit_gas_water_temperature = exit_gas_water_temperature
        self.exit_liquid_water_temperature = exit_liquid_water_temperature
        self.exit_solid_water_temperature = exit_solid_water_temperature
        self.exit_gas_water_fraction = exit_gas_water_fraction
        self.exit_liquid_water_fraction = exit_liquid_water_fraction
        self.exit_solid_water_fraction =  exit_solid_water_fraction
        self.wind_coupling = YesNo(wind_coupling)
        self.air_moisture = YesNo(air_moisture)
        self.latent_heat = YesNo(latent_heat) 
        self.reentrainment = YesNo(reentrainment) 
        self.bursik_factor = bursik_factor 
        self.z_min_wind = z_min_wind
        self.c_umbrella = c_umbrella
        self.a_s = a_s
        self.a_v = a_v  

    def plot_on_map(self,ax=None):


        if  ax==None:
            plt.figure("Test Map")
            ccrs.PlateCarree()
            crs = ccrs.PlateCarree()
            extent = [self.lon_vent-1, self.lon_vent+1,self.lat_vent-1, self.lat_vent+1]

            ax = plt.subplot(111, projection=crs)
            ax.set_extent(extent, crs=crs)
            
            #ax.add_feature(NaturalEarthFeature('physical', 'ocean', '50m'))
            ax.coastlines(resolution='10m',color='blue')
    
            ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)

        ax.plot([self.lon_vent], [self.lat_vent], marker='o', color='red')
            
        
        
        
        
        
class Ensemble(Section):
    
    fstring = fstring_ensemble
    
    regex = regex_ensemble
    
    types = {
        'random_numbers_from_file':YesNo,
        'perturbate_column_height':str,
        'column_height_perturbation_range':int,
        'column_height_pdf':str,
        'perturbate_mass_flow_rate':str,
        'mass_flow_rate_perturbation_range':int,
        'mass_flow_rate_pdf':str,
        'perturbate_source_start':str,
        'perturbate_source_start_range':int,
        'perturbate_source_start_pdf':str,
        'perturbate_source_duration':str,
        'perturbate_source_duration_range':int,
        'perturbate_source_duration_pdf':str,
        'perturbate_top_hat_thickness':str,
        'perturbate_top_hat_thickness_range':int,
        'perturbate_top_hat_pdf':str,
        'perturbate_suzuki_a':str,
        'perturbate_suzuki_a_range':int,
        'perturbate_suzuki_a_pdf':str,
        'perturbate_suzuki_l':str,
        'perturbate_suzuki_l_range':int,
        'perturbate_suzuki_l_pdf':str,
        'perturbate_wind':str,
        'perturbate_wind_range':int,
        'perturbate_wind_pdf':str,
        'perturbate_data_insertion_cloud_height':str,
        'perturbate_data_insertion_cloud_height_range':int,
        'perturbate_data_insertion_cloud_height_pdf':str,
        'perturbate_data_insertion_cloud_thickness':str,
        'perturbate_data_insertion_cloud_thickness_range':int,
        'perturbate_data_insertion_cloud_thickness_pdf':str,
        'perturbate_fi_mean':str,
        'perturbate_fi_range':int,
        'perturbate_fi_pdf':str,
        'perturbate_diamater_aggregates':YesNo,
        'perturbate_diamater_aggregates_range':int,
        'perturbate_diamater_aggregates_pdf':str,
        'perturbate_density_aggregates':str,
        'perturbate_density_aggregates_range':int,
        'perturbate_density_aggregates_pdf':str,
    }
    
    def __init__(self,
                random_numbers_from_file:YesNo,
                perturbate_column_height:str,
                column_height_perturbation_range:int,
                column_height_pdf:str,
                perturbate_mass_flow_rate:str,
                mass_flow_rate_perturbation_range:int,
                mass_flow_rate_pdf:str,
                perturbate_source_start:str,
                perturbate_source_start_range:int,
                perturbate_source_start_pdf:str,
                perturbate_source_duration:str,
                perturbate_source_duration_range:int,
                perturbate_source_duration_pdf:str,
                perturbate_top_hat_thickness:str,
                perturbate_top_hat_thickness_range:int,
                perturbate_top_hat_pdf:str,
                perturbate_suzuki_a:str,
                perturbate_suzuki_a_range:int,
                perturbate_suzuki_a_pdf:str,
                perturbate_suzuki_l:str,
                perturbate_suzuki_l_range:int,
                perturbate_suzuki_l_pdf:str,
                perturbate_wind:str,
                perturbate_wind_range:int,
                perturbate_wind_pdf:str,
                perturbate_data_insertion_cloud_height:str,
                perturbate_data_insertion_cloud_height_range:int,
                perturbate_data_insertion_cloud_height_pdf:str,
                perturbate_data_insertion_cloud_thickness:str,
                perturbate_data_insertion_cloud_thickness_range:int,
                perturbate_data_insertion_cloud_thickness_pdf:str,
                perturbate_fi_mean:str,
                perturbate_fi_range:int,
                perturbate_fi_pdf:str,
                perturbate_diamater_aggregates:YesNo,
                perturbate_diamater_aggregates_range:int,
                perturbate_diamater_aggregates_pdf:str,
                perturbate_density_aggregates:str,
                perturbate_density_aggregates_range:int,
                perturbate_density_aggregates_pdf:str,
                ):
        
        
        
        assert(type(random_numbers_from_file)==YesNo)
        assert(type(perturbate_column_height)==str)
        assert(type(column_height_perturbation_range)==int)
        assert(type(column_height_pdf)==str)
        assert(type(perturbate_mass_flow_rate)==str)
        assert(type(mass_flow_rate_perturbation_range)==int)
        assert(type(mass_flow_rate_pdf)==str)
        assert(type(perturbate_source_start)==str)
        assert(type(perturbate_source_start_range)==int)
        assert(type(perturbate_source_start_pdf)==str)
        assert(type(perturbate_source_duration)==str)
        assert(type(perturbate_source_duration_range)==int)
        assert(type(perturbate_source_duration_pdf)==str)
        assert(type(perturbate_top_hat_thickness)==str)
        assert(type(perturbate_top_hat_thickness_range)==int)
        assert(type(perturbate_top_hat_pdf)==str)
        assert(type(perturbate_suzuki_a)==str)
        assert(type(perturbate_suzuki_a_range)==int)
        assert(type(perturbate_suzuki_a_pdf)==str)
        assert(type(perturbate_suzuki_l)==str)
        assert(type(perturbate_suzuki_l_range)==int)
        assert(type(perturbate_suzuki_l_pdf)==str)
        assert(type(perturbate_wind)==str)
        assert(type(perturbate_wind_range)==int)
        assert(type(perturbate_wind_pdf)==str)
        assert(type(perturbate_data_insertion_cloud_height)==str)
        assert(type(perturbate_data_insertion_cloud_height_range)==int)
        assert(type(perturbate_data_insertion_cloud_height_pdf)==str)
        assert(type(perturbate_data_insertion_cloud_thickness)==str)
        assert(type(perturbate_data_insertion_cloud_thickness_range)==int)
        assert(type(perturbate_data_insertion_cloud_thickness_pdf)==str)
        assert(type(perturbate_fi_mean)==str)
        assert(type(perturbate_fi_range)==int)
        assert(type(perturbate_fi_pdf)==str)
        assert(type(perturbate_diamater_aggregates)==YesNo)
        assert(type(perturbate_diamater_aggregates_range)==int)
        assert(type(perturbate_diamater_aggregates_pdf)==str)
        assert(type(perturbate_density_aggregates)==str)
        assert(type(perturbate_density_aggregates_range)==int)
        assert(type(perturbate_density_aggregates_pdf)==str)
        
        self.random_numbers_from_file = YesNo(random_numbers_from_file)
        self.perturbate_column_height = perturbate_column_height
        self.column_height_perturbation_range  = column_height_perturbation_range
        self.column_height_pdf  = column_height_pdf
        self.perturbate_mass_flow_rate  = perturbate_mass_flow_rate
        self.mass_flow_rate_perturbation_range  = mass_flow_rate_perturbation_range
        self.mass_flow_rate_pdf  = mass_flow_rate_pdf
        self.perturbate_source_start  = perturbate_source_start
        self.perturbate_source_start_range  = perturbate_source_start_range
        self.perturbate_source_start_pdf  = perturbate_source_start_pdf
        self.perturbate_source_duration  = perturbate_source_duration
        self.perturbate_source_duration_range  = perturbate_source_duration_range
        self.perturbate_source_duration_pdf  = perturbate_source_duration_pdf
        self.perturbate_top_hat_thickness  = perturbate_top_hat_thickness
        self.perturbate_top_hat_thickness_range  = perturbate_top_hat_thickness_range
        self.perturbate_top_hat_pdf  = perturbate_top_hat_pdf
        self.perturbate_suzuki_a  = perturbate_suzuki_a
        self.perturbate_suzuki_a_range  = perturbate_suzuki_a_range
        self.perturbate_suzuki_a_pdf = perturbate_suzuki_a_pdf
        self.perturbate_suzuki_l = perturbate_suzuki_l
        self.perturbate_suzuki_l_range  = perturbate_suzuki_l_range
        self.perturbate_suzuki_l_pdf  = perturbate_suzuki_l_pdf
        self.perturbate_wind  = perturbate_wind
        self.perturbate_wind_range  = perturbate_wind_range
        self.perturbate_wind_pdf  = perturbate_wind_pdf
        self.perturbate_data_insertion_cloud_height  = perturbate_data_insertion_cloud_height
        self.perturbate_data_insertion_cloud_height_range  = perturbate_data_insertion_cloud_height_range
        self.perturbate_data_insertion_cloud_height_pdf  = perturbate_data_insertion_cloud_height_pdf
        self.perturbate_data_insertion_cloud_thickness =  perturbate_data_insertion_cloud_thickness
        self.perturbate_data_insertion_cloud_thickness_range  = perturbate_data_insertion_cloud_thickness_range
        self.perturbate_data_insertion_cloud_thickness_pdf  = perturbate_data_insertion_cloud_thickness_pdf
        self.perturbate_fi_mean  = perturbate_fi_mean
        self.perturbate_fi_range  = perturbate_fi_range
        self.perturbate_fi_pdf  = perturbate_fi_pdf
        self.perturbate_diamater_aggregates  = YesNo(perturbate_diamater_aggregates)
        self.perturbate_diamater_aggregates_range  = perturbate_diamater_aggregates_range
        self.perturbate_diamater_aggregates_pdf  = perturbate_diamater_aggregates_pdf
        self.perturbate_density_aggregates  = perturbate_density_aggregates
        self.perturbate_density_aggregates_range  = perturbate_density_aggregates_range
        self.perturbate_density_aggregates_pdf  = perturbate_density_aggregates_pdf

class TimeUTC(Section):
    
    fstring = fstring_time_utc

    regex = regex_time_utc
    
    types = {
        'year':int,
        'month':int,
        'day':int,
        'run_start':int,
        'run_end':int,
        'initial_condition':str,
        'restart_file':str,
        'restart_ensemble_basepath':str
    }
    
    
    
    def __init__(self, 
                year:int, 
                month:int, 
                day:int, 
                run_start:int, 
                run_end:int, 
                initial_condition:str, 
                restart_file:str, 
                restart_ensemble_basepath:str
                ):
        
        #---------------------------------------------------------------------------------
        # TEST INPUTS
        #---------------------------------------------------------------------------------
        
        # tests for year
        assert(type(year)==int)
        assert(year>=1970)
        assert(year<=2024)
        
        # tests for month
        assert(type(month)==int)
        assert(month>=1)
        assert(month<=12)
        
        # tests for day
        assert(type(day)==int)
        assert(day>=1)
        assert(day<=31)
        
        # tests for initial_condition
        assert(type(initial_condition)==str)
        assert(initial_condition in ['NONE', 'RESTART', 'INSERTION'])
        
        
        # tests for restart_file
        assert(type(restart_file)==str)
        
        # tests for restart_ensemble_basepath
        assert(type(restart_ensemble_basepath)==str)
        
        #---------------------------------------------------------------------------------       
        # OPTIONS THAT HAVE BECOME ACTIVATED
        #---------------------------------------------------------------------------------

        if initial_condition in ['RESTART']:
            print("RESTART_FILE in use as INITIAL_CONDITION = RESTART  )")
            
        self.year = year
        self.month = month
        self.day = day
        self.run_start = run_start
        self.run_end = run_end
        self.initial_condition = initial_condition
        self.restart_file = restart_file
        self.restart_ensemble_basepath = restart_ensemble_basepath  
        


class EnsemblePostprocess(Section):

    fstring = fstring_ensemble_postprocess
    
    regex = regex_ensemble_postprocess

    types = {
        'postprocess_members':YesNo,
        'postprocess_mean':YesNo,
        'postprocess_logmean':YesNo,
        'postprocess_median':YesNo,
        'postprocess_standard_dev':YesNo,
        'postprocess_probability':YesNo,
        'postprocess_percentiles':YesNo,
        'postprocess_probability_concentration_thresholds':int,
        'postprocess_probability_column_mass_thresholds_gm2':int,
        'postprocess_probability_column_mass_thresholds_du':int,
        'postprocess_probability_ground_load_thresholds':int ,
        'postprocess_percentiles_percentile_values':int  
    }


    def __init__(self,
                postprocess_members:YesNo,
                postprocess_mean:YesNo,
                postprocess_logmean:YesNo,
                postprocess_median:YesNo,
                postprocess_standard_dev:YesNo,
                postprocess_probability:YesNo,
                postprocess_percentiles:YesNo,
                postprocess_probability_concentration_thresholds:int,
                postprocess_probability_column_mass_thresholds_gm2:int,
                postprocess_probability_column_mass_thresholds_du:int,
                postprocess_probability_ground_load_thresholds:int ,
                postprocess_percentiles_percentile_values:int  
                ):

        assert(type(postprocess_members)==YesNo)
        assert(type(postprocess_mean)==YesNo)
        assert(type(postprocess_logmean)==YesNo)
        assert(type(postprocess_median)==YesNo)
        assert(type(postprocess_standard_dev)==YesNo)
        assert(type(postprocess_probability)==YesNo)
        assert(type(postprocess_percentiles)==YesNo)
        assert(type(postprocess_probability_concentration_thresholds)==int)
        assert(type(postprocess_probability_column_mass_thresholds_gm2)==int)
        assert(type(postprocess_probability_column_mass_thresholds_du)==int)
        assert(type(postprocess_probability_ground_load_thresholds)==int )
        assert(type(postprocess_percentiles_percentile_values)==int)  

        self.postprocess_members = YesNo(postprocess_members)
        self.postprocess_mean = YesNo(postprocess_mean)
        self.postprocess_logmean = YesNo(postprocess_logmean)
        self.postprocess_median = YesNo(postprocess_median)
        self.postprocess_standard_dev = YesNo(postprocess_standard_dev)
        self.postprocess_probability = YesNo(postprocess_probability)
        self.postprocess_percentiles = YesNo(postprocess_percentiles)
        self.postprocess_probability_concentration_thresholds = postprocess_probability_concentration_thresholds
        self.postprocess_probability_column_mass_thresholds_gm2 = postprocess_probability_column_mass_thresholds_gm2 
        self.postprocess_probability_column_mass_thresholds_du = postprocess_probability_column_mass_thresholds_du
        self.postprocess_probability_ground_load_thresholds = postprocess_probability_ground_load_thresholds 
        self.postprocess_percentiles_percentile_values = postprocess_percentiles_percentile_values

class ModelPhysics(Section):

    fstring = fstring_model_physics
    
    regex = regex_model_physics

    types = {
        'limiter':str,
        'time_marching':str,
        'cfl_criterion':str,
        'cfl_safety_factor':float,
        'terminal_velocity_model':str,
        'horizontal_turbulence_model':str,
        'vertical_turbulence_model':float,
        'rams_cs':float,
        'wet_deposition':YesNo,
        'dry_deposition':YesNo,
        'gravity_current':YesNo,
        'c_flow_rate':str,
        'lambda_grav':float,
        'k_entrain':float,
        'brunt_vaisala':float,
        'gc_start':int,
        'gc_end':int
    }

    def __init__(self,
                limiter:str,
                time_marching:str,
                cfl_criterion:str,
                cfl_safety_factor:float,
                terminal_velocity_model:str,
                horizontal_turbulence_model:str,
                vertical_turbulence_model:float,
                rams_cs:float,
                wet_deposition:YesNo,
                dry_deposition:YesNo,
                gravity_current:YesNo,
                c_flow_rate:str,
                lambda_grav:float,
                k_entrain:float,
                brunt_vaisala:float,
                gc_start:int,
                gc_end:int
                ):
        
        assert(type(limiter)==str)
        assert(type(time_marching)==str)
        assert(type(cfl_criterion)==str)
        assert(type(cfl_safety_factor)==float)
        assert(type(terminal_velocity_model)==str)
        assert(type(horizontal_turbulence_model)==str)
        assert(type(vertical_turbulence_model)==float)
        assert(type(rams_cs)==float)
        assert(type(wet_deposition)==YesNo)
        assert(type(dry_deposition)==YesNo)
        assert(type(gravity_current)==YesNo)
        assert(type(c_flow_rate)==str)
        assert(type(lambda_grav)==float)
        assert(type(k_entrain)==float)
        assert(type(brunt_vaisala)==float)
        assert(type(gc_start)==int)
        assert(type(gc_end)==int)


        self.limiter = limiter
        self.time_marching = time_marching
        self.cfl_criterion = cfl_criterion
        self.cfl_safety_factor = cfl_safety_factor
        self.terminal_velocity_model = terminal_velocity_model
        self.horizontal_turbulence_model = horizontal_turbulence_model
        self.vertical_turbulence_model = vertical_turbulence_model
        self.rams_cs = rams_cs
        self.wet_deposition = YesNo(wet_deposition)
        self.dry_deposition = YesNo(dry_deposition)
        self.gravity_current = YesNo(gravity_current)
        self.c_flow_rate = c_flow_rate
        self.lambda_grav = lambda_grav
        self.k_entrain = k_entrain
        self.brunt_vaisala = brunt_vaisala
        self.gc_start = gc_start 
        self.gc_end = gc_end




class ModelOutput(Section):

    fstring = fstring_model_output
    
    regex = regex_model_output

    types = {
        'parallel_io':YesNo,
        'log_file_level':str,
        'restart_time_interval':str,
        'output_intermediate_files':YesNo,
        'output_time_start':str,
        'output_time_interval':float,
        'output_3d_concentration':YesNo,
        'output_3d_concentration_bins':YesNo,
        'output_surface_concentration':YesNo,
        'output_column_load':YesNo,
        'output_cloud_top':YesNo,
        'output_ground_load':YesNo,
        'output_ground_load_bins':YesNo,
        'output_wet_deposition':YesNo,
        'output_track_points':YesNo,
        'output_track_points_file':str,
        'output_concentrations_at_xcuts':YesNo,
        'x_values':str,
        'output_concentrations_at_ycuts':YesNo,
        'y_values':str,
        'output_concentrations_at_zcuts':YesNo,
        'z_values':str,
        'output_concentrations_at_fl':YesNo,
        'fl_values':str,
    }
        
    def __init__(self,
            parallel_io:YesNo,
            log_file_level:str,
            restart_time_interval:str,
            output_intermediate_files:YesNo,
            output_time_start:str,
            output_time_interval:float,
            output_3d_concentration:YesNo,
            output_3d_concentration_bins:YesNo,
            output_surface_concentration:YesNo,
            output_column_load:YesNo,
            output_cloud_top:YesNo,
            output_ground_load:YesNo,
            output_ground_load_bins:YesNo,
            output_wet_deposition:YesNo,
            output_track_points:YesNo,
            output_track_points_file:str,
            output_concentrations_at_xcuts:YesNo,
            x_values:str,
            output_concentrations_at_ycuts:YesNo,
            y_values:str,
            output_concentrations_at_zcuts:YesNo,
            z_values:str,
            output_concentrations_at_fl:YesNo,
            fl_values:str,
            ):


        assert(type(parallel_io)==YesNo)
        assert(type(log_file_level)==str)
        assert(type(restart_time_interval)==str)
        assert(type(output_intermediate_files)==YesNo)
        assert(type(output_time_start)==str)
        assert(type(output_time_interval)==float)
        assert(type(output_3d_concentration)==YesNo)
        assert(type(output_3d_concentration_bins)==YesNo)
        assert(type(output_surface_concentration)==YesNo)
        assert(type(output_column_load)==YesNo)
        assert(type(output_cloud_top)==YesNo)
        assert(type(output_ground_load)==YesNo)
        assert(type(output_ground_load_bins)==YesNo)
        assert(type(output_wet_deposition)==YesNo)
        assert(type(output_track_points)==YesNo)
        assert(type(output_track_points_file)==str)
        assert(type(output_concentrations_at_xcuts)==YesNo)
        assert(type(x_values)==str)
        assert(type(output_concentrations_at_ycuts)==YesNo)
        assert(type(y_values)==str)
        assert(type(output_concentrations_at_zcuts)==YesNo)
        assert(type(z_values)==str)
        assert(type(output_concentrations_at_fl)==YesNo)
        assert(type(fl_values)==str)


        self.parallel_io = parallel_io
        self.log_file_level = log_file_level
        self.restart_time_interval = restart_time_interval
        self.output_intermediate_files = output_intermediate_files
        self.output_time_start = output_time_start
        self.output_time_interval = output_time_interval
        self.output_3d_concentration = YesNo(output_3d_concentration)
        self.output_3d_concentration_bins = YesNo(output_3d_concentration_bins)
        self.output_surface_concentration = YesNo(output_surface_concentration)
        self.output_column_load = YesNo(output_column_load)
        self.output_cloud_top = YesNo(output_cloud_top)
        self.output_ground_load = YesNo(output_ground_load)
        self.output_ground_load_bins = YesNo(output_ground_load_bins)
        self.output_wet_deposition = YesNo(output_wet_deposition)
        self.output_track_points = YesNo(output_track_points)
        self.output_track_points_file = output_track_points_file
        self.output_concentrations_at_xcuts = YesNo(output_concentrations_at_xcuts)
        self.x_values = x_values
        self.output_concentrations_at_ycuts = YesNo(output_concentrations_at_ycuts)
        self.y_values = y_values
        self.output_concentrations_at_zcuts = YesNo(output_concentrations_at_zcuts)
        self.z_values = z_values
        self.output_concentrations_at_fl = YesNo(output_concentrations_at_fl)
        self.fl_values = fl_values




class ModelValidation(Section):

    fstring = fstring_model_validation
    
    regex = regex_model_validation

    types = {
        'observations_type':str,
        'observations_file':str,
        'observations_dictionary_file':str,
        'results_file':str,
        'column_mass_observations_threshold_gm2':float,
        'column_mass_observations_threshold_du':int,
        'ground_load_observation_threshold_kgm2':float 
    }

    def __init__(self,
                observations_type:str,
                observations_file:str,
                observations_dictionary_file:str,
                results_file:str,
                column_mass_observations_threshold_gm2:float,
                column_mass_observations_threshold_du:int,
                ground_load_observation_threshold_kgm2:float, 
                ):
        
        assert(type(observations_type)==str)
        assert(type(observations_file)==str)
        assert(type(observations_dictionary_file)==str)
        assert(type(results_file)==str)
        assert(type(column_mass_observations_threshold_gm2)==float)
        assert(type(column_mass_observations_threshold_du)==int)
        assert(type(ground_load_observation_threshold_kgm2)==float)

        self.observations_type = observations_type
        self.observations_file = observations_file
        self.observations_dictionary_file = observations_dictionary_file
        self.results_file = results_file
        self.column_mass_observations_threshold_gm2 = column_mass_observations_threshold_gm2
        self.column_mass_observations_threshold_du = column_mass_observations_threshold_du
        self.ground_load_observation_threshold_kgm2 = ground_load_observation_threshold_kgm2 

class Fall3DInputFile:

    types = {
                'time_utc': TimeUTC,
                'insertion_data': InsertionData,
                'meteo_data': MeteoData,
                'grid':Grid,
                'species':Species,
                'tephra_tgsd':TephraTgsd,
                'radionucleides_tgsd':RadionucleidesTgsd,
                'particle_aggregation':ParticleAggregation,
                'source':Source,
                'ensemble':Ensemble,
                'emsemble_postprocess': EnsemblePostprocess,
                'model_physics':ModelPhysics,
                'model_output':ModelOutput,
                'model_validation':ModelValidation
        
    }

    def __init__(self,
                time_utc: TimeUTC,
                insertion_data: InsertionData,
                meteo_data: MeteoData,
                grid:Grid,
                species:Species,
                tephra_tgsd:TephraTgsd,
                radionucleides_tgsd:RadionucleidesTgsd,
                particle_aggregation:ParticleAggregation,
                source:Source,
                ensemble:Ensemble,
                emsemble_postprocess: EnsemblePostprocess,
                model_physics:ModelPhysics,
                model_output:ModelOutput,
                model_validation:ModelValidation,
                file=None
                ):
        
        assert(type(time_utc)==TimeUTC)
        assert(type(insertion_data)==InsertionData)
        assert(type(meteo_data)==MeteoData)
        assert(type(grid)==Grid)
        assert(type(species)==Species)
        assert(type(tephra_tgsd)==TephraTgsd)
        assert(type(radionucleides_tgsd)==RadionucleidesTgsd)
        assert(type(particle_aggregation)==ParticleAggregation)
        assert(type(source)==Source)
        assert(type(ensemble)==Ensemble)
        assert(type(emsemble_postprocess)== EnsemblePostprocess)
        assert(type(model_physics)==ModelPhysics)
        assert(type(model_output)==ModelOutput)
        assert(type(model_validation)==ModelValidation)

        self.time_utc = time_utc
        self.insertion_data = insertion_data
        self.meteo_data = meteo_data
        self.grid = grid
        self.species = species
        self.tephra_tgsd = tephra_tgsd
        self.radionucleides_tgsd = radionucleides_tgsd
        self.particle_aggregation = particle_aggregation
        self.source = source
        self.ensemble = ensemble
        self.emsemble_postprocess = emsemble_postprocess
        self.model_physics = model_physics
        self.model_output = model_output
        self.model_validation = model_validation
        self.file = file


    @classmethod
    def from_string(cls, string:str):
        
        lines = string.split("\n")
        
        
        string_timeutc = "".join(lines[16:38]) 
        string_insertiondata = "".join(lines[38:52]) 
        string_meteodata = "".join(lines[52:78]) 
        string_grid = "".join(lines[78:110]) 
        string_species = "".join(lines[110:152]) 
        string_tephratgsd = "".join(lines[152:186]) 
        string_radionucleidestgsd = "".join(lines[186:224]) 
        string_particleaggregation = "".join(lines[224:246]) 
        string_source = "".join(lines[246:338]) 
        string_ensemble = "".join(lines[338:440]) 
        string_ensemblepostprocess = "".join(lines[440:469]) 
        string_modelphysics = "".join(lines[469:513])
        string_modeloutput = "".join(lines[513:564]) 
        string_modelvalidation = "".join(lines[564:])
    


        time_utc = TimeUTC.from_string(
                            string_timeutc
                            )
        
        insertion_data = InsertionData.from_string(
                            string_insertiondata
                            )
        
        meteo_data = MeteoData.from_string(
                            string_meteodata
                            )
        
        grid = Grid.from_string(
                            string_grid
                            )
        
        species = Species.from_string(
                            string_species
                            )
        
        tephra_tgsd = TephraTgsd.from_string(
                            string_tephratgsd
                            )
        
        radionucleides_tgsd = RadionucleidesTgsd.from_string(
                            string_radionucleidestgsd
                            )
        
        particle_aggregation = ParticleAggregation.from_string(
                            string_particleaggregation
                            )
        
        source = Source.from_string(
                            string_source
                            )

        ensemble = Ensemble.from_string(
                            string_ensemble
                            )
        
        emsemble_postprocess =  EnsemblePostprocess.from_string(
                            string_ensemblepostprocess
                            )
        
        model_physics = ModelPhysics.from_string(
                            string_modelphysics
                            )
        
        model_output = ModelOutput.from_string(
                            string_modeloutput
                            )

        model_validation = ModelValidation.from_string(
                            string_modelvalidation
                            )

        f3dif = cls(
            time_utc = time_utc,
            insertion_data = insertion_data,
            meteo_data = meteo_data,
            grid = grid,
            species = species,
            tephra_tgsd = tephra_tgsd,
            radionucleides_tgsd = radionucleides_tgsd,
            particle_aggregation = particle_aggregation,
            source = source,
            ensemble = ensemble,
            emsemble_postprocess =  emsemble_postprocess,
            model_physics = model_physics,
            model_output = model_output,
            model_validation = model_validation,
           # file=file
        )
        
        

        return(f3dif)

    @classmethod
    def from_file(cls, file:str):
        """New Fall3DInputFile from file
        """

        with open(file) as f:
            lines = f.readlines()

    
        string_timeutc = "".join(lines[16:38]) 
        string_insertiondata = "".join(lines[38:52]) 
        string_meteodata = "".join(lines[52:78]) 
        string_grid = "".join(lines[78:110]) 
        string_species = "".join(lines[110:152]) 
        string_tephratgsd = "".join(lines[152:186]) 
        string_radionucleidestgsd = "".join(lines[186:224]) 
        string_particleaggregation = "".join(lines[224:246]) 
        string_source = "".join(lines[246:338]) 
        string_ensemble = "".join(lines[338:440]) 
        string_ensemblepostprocess = "".join(lines[440:469]) 
        string_modelphysics = "".join(lines[469:513])
        string_modeloutput = "".join(lines[513:564]) 
        string_modelvalidation = "".join(lines[564:])
    


        time_utc = TimeUTC.from_string(
                            string_timeutc
                            )
        
        insertion_data = InsertionData.from_string(
                            string_insertiondata
                            )
        
        meteo_data = MeteoData.from_string(
                            string_meteodata
                            )
        
        grid = Grid.from_string(
                            string_grid
                            )
        
        species = Species.from_string(
                            string_species
                            )
        
        tephra_tgsd = TephraTgsd.from_string(
                            string_tephratgsd
                            )
        
        radionucleides_tgsd = RadionucleidesTgsd.from_string(
                            string_radionucleidestgsd
                            )
        
        particle_aggregation = ParticleAggregation.from_string(
                            string_particleaggregation
                            )
        
        source = Source.from_string(
                            string_source
                            )

        ensemble = Ensemble.from_string(
                            string_ensemble
                            )
        
        emsemble_postprocess =  EnsemblePostprocess.from_string(
                            string_ensemblepostprocess
                            )
        
        model_physics = ModelPhysics.from_string(
                            string_modelphysics
                            )
        
        model_output = ModelOutput.from_string(
                            string_modeloutput
                            )

        model_validation = ModelValidation.from_string(
                            string_modelvalidation
                            )

        f3dif = cls(
            time_utc = time_utc,
            insertion_data = insertion_data,
            meteo_data = meteo_data,
            grid = grid,
            species = species,
            tephra_tgsd = tephra_tgsd,
            radionucleides_tgsd = radionucleides_tgsd,
            particle_aggregation = particle_aggregation,
            source = source,
            ensemble = ensemble,
            emsemble_postprocess =  emsemble_postprocess,
            model_physics = model_physics,
            model_output = model_output,
            model_validation = model_validation,
            file=file
        )
        
        

        return(f3dif)

        
    def to_string(self):
        
        string = "".join([
            fstring_boilerplate,
            self.time_utc.to_string(),
            self.insertion_data.to_string(),
            self.meteo_data.to_string(),
            self.grid.to_string(),
            self.species.to_string(),
            self.tephra_tgsd.to_string(),
            self.radionucleides_tgsd.to_string(),
            self.particle_aggregation.to_string(),
            self.source.to_string(),
            self.ensemble.to_string(),
            self.emsemble_postprocess.to_string(),
            self.model_physics.to_string(),
            self.model_output.to_string(),
            self.model_validation.to_string()
        ])

        return(string)

    def to_file(self, file):

        string = self.to_string()

        with open(file,"w+") as f:
            f.writelines(string)
            
        self.output_file = file

    def update(self,new_values:dict):
        """Update with values in dict
        """

        sections = {}

        # we iterate through each section of the file ....
        for section_name, section_type in self.types.items():
            
            section = getattr(self,section_name)

            
            # ... for each section of the file we get the current params ...
        
            params = {}
    
            for attribute_name, attribute_type in section.types.items():
                
                params[attribute_name] = getattr(section,attribute_name)

            # ...we then check to see if there are any values in that section to update ...
            if section_name in new_values.keys():

                # ... and if there are, we update them
                for name, value in new_values[section_name].items():
                    params[name] = value

            # ... finally we reinitialise the section with the new params ..
            section.__init__(**params)

            # ... and append it to our sections dict.
            sections[section_name] = section

        
        # ... finally, we reinitialise with the update parameters
        self.__init__(**sections)
        
    def get_meteodata(self):
        """Fetches Meteo data based on specification in time_utc, grid and meteo_data
        """

        source = get_MeteoSource(self)

        source.get_fall3d_input(self)


    def plot_on_map(self):
        """Plots meteodata extent, grid extent and source location on a map with a high resolution
        coastlinbe.
        """


        
        plt.figure("Test Map")
        ccrs.PlateCarree()
        crs = ccrs.PlateCarree()
        ax = plt.subplot(111, projection=crs)        
        #ax.add_feature(NaturalEarthFeature('physical', 'ocean', '50m'))
        ax.coastlines(resolution='10m',color='blue')

        ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)

        extent = self.meteo_data.plot_on_map(ax=ax)
        self.grid.plot_on_map(ax=ax)
        self.source.plot_on_map(ax=ax)

        ax.set_extent(extent, crs=crs)
        






                             

def get_MeteoSource(file:Fall3DInputFile): #time_utc:TimeUTC, grid:Grid, 
    """General class to fetch
    """

    data_format = file.meteo_data.meteo_data_format

    switch = {
        'WRF':WRFSource,
        'ERA5':ERA5Source,
        'GFS':GFSSource,
        'IFS':IFSSource,
        'CARRA':CARRASource,
        'ERA5ML':ERA5MLSource
    }

    source = switch[data_format]() #time_utc, grid, meteo_data

    return(source)
        
class ERA5MLSource:

    def __init__(self):#,time_utc:TimeUTC, grid:Grid, meteo_data:MeteoData):
        raise NotImplementedError

class ERA5Source:

    def __init__(self):#,time_utc:TimeUTC, grid:Grid, meteo_data:MeteoData):
        raise NotImplementedError

class WRFSource:

    def __init__(self):#,time_utc:TimeUTC, grid:Grid, meteo_data:MeteoData):
        raise NotImplementedError

class GFSSource:

    def __init__(self):#,time_utc:TimeUTC, grid:Grid, meteo_data:MeteoData):
        raise NotImplementedError


class IFSSource:

    def __init__(self):#,time_utc:TimeUTC, grid:Grid, meteo_data:MeteoData):
        raise NotImplementedError

class CARRASource:
    
    # we need a file on the full native CARRA West domain grid
    # so that we have the latitude and longitude grids, as well
    # as the projection information in the attributes to hand
    #file_native = "/media/talfan/Maxtor/test_interpolated.nc"
    file_native = "mnt/aux/CARRA_orography_west.nc"

    
    def __init__(self, local_storage="mnt/archive/"):
        
        self.ds_native = xr.open_mfdataset(self.file_native)['orog'].drop(['time','step','surface','valid_time'])
        self.local_storage = local_storage

    def get_fall3d_input(self, file:Fall3DInputFile):
        """
        """
        
        time_utc = file.time_utc
        grid = file.grid
        meteo_data = file.meteo_data
        
        if os.path.exists(meteo_data.meteo_data_file):
        	#raise ValueError
            print("File already exists: "+meteo_data.meteo_data_file,", skipping")
            return( xr.open_dataset(meteo_data.meteo_data_file) )

        # get extent from Grid object
        latmax = grid.latmax
        latmin = grid.latmin
        # remember! Longitudes are specified -180 to 180 in a grid object
        lonmax = Longitude(grid.lonmax,180)
        lonmin = Longitude(grid.lonmin,180)

        # get time info from TimeUTC object ...
        year = time_utc.year
        month = time_utc.month
        day = time_utc.day
        run_start = time_utc.run_start
        run_end = time_utc.run_end

        
        # ... convert to start and end datetimes ...
        start = datetime.datetime(year=year,month=month,day=day,hour=run_start)
        #start0 = datetime.datetime(year=year,month=month,day=day,hour=0)
        duration = run_end-run_start
        
        end = start + datetime.timedelta(hours=duration)
        #end0 = start + datetime.timedelta(hours=duration+4)

        # ... and get the months we need to order.
        # Fuirst, we get the difference between the two dates in seconds 
        # by differencing the unix timestamps ...
        seconds = (end.timestamp()-start.timestamp())

        # .. which we convert to decimal days ...
        days = seconds/(24*60*60)

        # ... round up to the nearest whole day ...
        days = np.ceil(days)

        # .. then convert to int for iterating over ...
        days = int(days)
        
        # ... to get a list of all daily dates between the start and end date ...
        months = []
        for day in range(days):
            date = start + datetime.timedelta(days=day)

            # ... and for each date we save the year and month ...
            months.append((date.year,date.month))

        # .. so that by performiung a set operation we can get the uniuque
        # year month pairs we need to order using the cdsapi ...
        months = set(months)

        # ... we order the data to get a list of datasets ...
        ds_months = [self.get_month(lonmin, lonmax, latmin, latmax, year, month) for year, month in months]

        # .. which we concatenate into a single one
        ds = xr.concat(ds_months,dim='time')

        # SUBSET BY DATE HERE!!!!
        ds_sub = ds.sel(
		time=(ds.time.values>= np.datetime64(start))&
		     (ds.time.values<= np.datetime64(end ))
	)

        # write to the file specified in the meteo_data section
        ds_sub.to_netcdf(meteo_data.meteo_data_file)

        return(ds)
    
 
        
    def get_month(self,lonmin:Longitude, lonmax:Longitude, latmin:float, latmax:float, year:int, month:int):
        """NOTE! We order by the month due to CDSAPI putting orders longer than a month
         to the back of the queue!!!
         """
        # first we check local storage to see if we already have a file for that
        # year and month that covers our spatial extent
        
        file = self.check_local_storage(lonmin, lonmax, latmin, latmax, year, month)
        
        if file is None:
            print("No appropriate file found in local storage for ",year, month, lonmin._180, lonmax._180, latmin,latmax)
            print("Submitting order to CDSAPI")
            file = self.make_request(lonmin, lonmax, latmin, latmax, year, month)
            
        else:
            print("Local file found:",file)
        ds = xr.open_dataset(file)
        
        return(ds)
            

    def get_local_storage(self):
        
        # search for netcdfs in the local archive
        files = glob.glob("mnt/archive/*.nc")

        # if there are no .nc files, we return an empty dataframe ..
        if len(files) ==0:
            
            df = pd.DataFrame(columns=['file','year','month','lonmin','lonmax','latmin','latmax'])

        # ... otherwise we populate the dataframe with dates and extents
        else:
            
            data = [re.findall(r"(.*(\d{4})(\d{2})_(.+)_(.+)_(.+)_(.+)__CARRA\.nc)",file)[0]  for file in files]    

            df = pd.DataFrame(
                data, 
                columns=['file','year','month','lonmin','lonmax','latmin','latmax']
            )

            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            df['lonmin'] = df['lonmin'].astype(float)
            df['lonmax'] = df['lonmax'].astype(float)
            df['latmin'] = df['latmin'].astype(float)
            df['latmax'] = df['latmax'].astype(float)

        return(df)

    
            
    
    
    def check_local_storage(self,lonmin:Longitude, lonmax:Longitude, latmin:float, latmax:float, year:int, month:int):
        """Check to see if we already have a file that matches the request
        """
       
        df = self.get_local_storage()

        # if there are no files we wil return None, otherwise ...
        if len(df)==0:
            
            file = None

        # we check to see if we have a file that meets our requirements:
        else:
        
            check = (
                    lambda r: 
                        (r['year']== year)&
                        (r['month']== month)&
                        (r['lonmin']<= lonmin._360)&
                        (r['lonmax']>= lonmax._360)&
                        (r['latmin']<= latmin)&
                        (r['latmax']>= latmax)
                        )
    
            df['match'] = df.apply(check, axis=1)

            df_match = df[df['match']]

            # if there is 1 or more files that meets our requirements we will return the first one ...
            if len(df_match)>=1:
                file = df_match['file'].item()

            # ... otherwise we return None as before
            else:
                file = None
        
            
        return(file)

      
    
    def basic_request(self, lonmin:Longitude, lonmax:Longitude, latmin:float, latmax:float, year:int, month:int)->dict:
        """Gets the part of the cdsapi request dict that is common to both single and pressure level
        requests.
        """
        #print(lonmin._180)
        #print(lonmin._360)
        
        self.lonmin = lonmin
        self.lonmax = lonmax
        self.latmin = latmin
        self.latmax = latmax
        self.year = year
        self.month = month
        
        # then we get the filenames we will use - NOTE we use 0-360 long here
        self.filename = "_".join([
                str(year) + str(month).zfill(2),
                str(self.lonmin._360),
                str(self.lonmax._360),
                str(self.latmin),
                str(self.latmax),
                '_CARRA.nc'
            ])
        
        
        
        # a visualisation of the problem:
        # we need to order a CARRA subset 
        # specified in lat lon (outer box) 
        # big enough to contain the extent
        # in CARRA x-y coordinates (middle box)
        # that encompasse our domain of interest
        # (inner box)
        #
        #    *-----------------------*
        #    |       *               |
        #    |      +   +            |        
        #    |     +       +         |
        #    |    +*----------*      |    
        #    |   + |          |  +   |
        #    |  +  |          |    * |   
        #    | +   |          |   +  |  
        #    |*    |          |  +   |   
        #    |  +  *----------* +    |     
        #    |     +           +     |             
        #    |        +       +      |          
        #    |           +   +       |      
        #    |              *        |         
        #    *-----------------------*
        #
        
        
        # In other words, we need the range of lat lon to order from cdsapi, that
        # encompases a box in native grid (x,y) that in turn
        # encmopases the range of lat lon we want for Fall3D. It's complicated!
        # (native lon is 0-360)
        domain_xy=(
                    (self.ds_native.latitude>latmin)&
                    (self.ds_native.latitude<latmax)&
                    (self.ds_native.longitude>lonmin._360)&
                    (self.ds_native.longitude<lonmax._360)
                    )
        
        # we get the min and max x and y values
        # we will use to clip the data we download after
        # we order it
        xs = np.where(domain_xy.values.any(axis=0))[0]
        xmin = xs.min()
        xmax = xs.max()

        ys = np.where(domain_xy.values.any(axis=1))[0]
        ymin = ys.min()
        ymax = ys.max()
        
        # we buffer these values to be on the safe side
        xmin -= 2
        xmax += 2
        ymin -= 2
        ymax += 2

        # Now, we need to find a range of (lat, lon) values
        # that completely enclose this range of xy values

        domain_xy = domain_xy.sel(x=slice(xmin, xmax),y=slice(ymin, ymax))
        
        # we need to save domain_xy for resampling later
        self.domain_xy = domain_xy

        latmax_for_cdsapi = domain_xy.latitude.max().values.item()
        latmin_for_cdsapi = domain_xy.latitude.min().values.item()
        # longitude goes 0-360 here for some reason
        lonmax_for_cdsapi = Longitude( domain_xy.longitude.max().values.item(), 360)
        lonmin_for_cdsapi = Longitude(domain_xy.longitude.min().values.item(),360)

        # buffer the cdsapi bounds by 0.1 of a degree just in case
        latmax_for_cdsapi += 0.1 
        latmin_for_cdsapi -= 0.1 
        lonmax_for_cdsapi = Longitude( lonmax_for_cdsapi._360 + 0.1, 360)
        lonmin_for_cdsapi = Longitude( lonmin_for_cdsapi._360 - 0.1, 360)

        # finally, remember that longitude for cdsapi is -180-180, not 0-360
        #lon360_to_180 = lambda lon:( lon + 180) % 360 - 180
        #lonmax_for_cdsapi = lon360_to_180(lonmax_for_cdsapi)
        #lonmin_for_cdsapi = lon360_to_180(lonmin_for_cdsapi)

        # the subset area for cdsapi is specified like this
        # NOTE! lon is specified -180 - 180 here
        area = [latmax_for_cdsapi, lonmin_for_cdsapi._180, latmin_for_cdsapi, lonmax_for_cdsapi._180 ]
        
        # now we need to specify the resolution
        # this needs to be a number in degrees that is the result of dividing
        # 90 by an integer:
        num_samples_in_90_degrees = 3000

        resolution = 90/num_samples_in_90_degrees

        # we need the number of days in the month, which we find by subtracting
        # the 1st of the next month from the 1st of the current month
        start = datetime.datetime(year=year,month=month,day=1)

        if month==12:
            end = datetime.datetime(year=year+1,month=1,day=1)
        else:
            end = datetime.datetime(year=year,month=month+1,day=1)

        days_in_month = (end-start).days

        days = [day for day in range(1,days_in_month+1)]


        # now we turn all the date integers to strings
        day = [str(day).zfill(2) for day in days]

        year = str(year)

        month = str(month).zfill(2)

        # we always get all three hourly data for every day
        # these will never change
        time =[
                    '00:00', '03:00', '06:00',
                    '09:00', '12:00', '15:00',
                    '18:00', '21:00',
                ]


        # ... we will make two requests, one for 
        basic_request= {
            'format': 'grib',
            'domain': 'west_domain',
            'format': 'grib',
            'product_type': 'analysis',
            'grid':[resolution,resolution],
            'area': area,
            'time': time,
            'year': year,
            'month': month,
            'day': day
        }
        
        return(basic_request)
    
    def levels_request(self, lonmin:Longitude, lonmax:Longitude, latmin:float, latmax:float, year:int, month:int)->dict:
        
        levels_request = self.basic_request(lonmin, lonmax, latmin, latmax, year, month)
        
        levels_request['variable'] = [
                    'geometric_vertical_velocity','geopotential', 
                    'relative_humidity', 'temperature',
                    'u_component_of_wind', 'v_component_of_wind',
                ]
        
        levels_request['pressure_level'] = [
                                                '800', '825', '850',
                                                '875', '900', '925',
                                                '950', '1000',
                                            ]
        
        return(levels_request)
        

    def single_request(self, lonmin:Longitude, lonmax:Longitude, latmin:float, latmax:float, year:int, month:int)->dict:
    
        single_request = self.basic_request(lonmin, lonmax, latmin, latmax, year, month)
        
        single_request['variable']= [
                    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_relative_humidity',
                    '2m_temperature', 'land_sea_mask',
                     'orography', 'surface_pressure', 'surface_roughness'
                ]
        
        single_request['level_type']= 'surface_or_atmosphere'
        
        return(single_request)
    
    def make_request(self, lonmin:Longitude, lonmax:Longitude, latmin:float, latmax:float, year:int, month:int):
        
        # first we check to see if we already have the data
        #file = local_storage(self,lonmin, lonmax, latmin, latmax, year, month)
        
        #if file is None:
        #    return(file)
                             
        # first we get the requests dicts
        levels_request = self.levels_request(lonmin, lonmax, latmin, latmax, year, month)
        
        single_request = self.single_request(lonmin, lonmax, latmin, latmax, year, month)
        
        # then we get the path to store the data
        path = os.path.join(self.local_storage, self.filename)
        
        # now we order the data. First we initialise the cdsapi client ...

        c = cdsapi.Client()

        # and we make the two requests
        c.retrieve(
            'reanalysis-carra-pressure-levels',
            levels_request,
            'pressure_levels.grib')

        c.retrieve(
            'reanalysis-carra-single-levels',
            single_request,
            'single_levels.grib')
        
        # Once we have the two gribs, we open them as xarray datasets ...
        gribs_as_ds = cfgrib.open_datasets('single_levels.grib')
        
        gribs_as_ds += cfgrib.open_datasets('pressure_levels.grib')

        # ... and iterate over them, interpolating them back to the CARRA grid
        # from the lat lon subset we we have ordered, so Fall3D can use it ...
        dss = []

        for i,ds in enumerate(gribs_as_ds):

            # resample
            ds_resampled = (
                            ds
                            .interp({
                                'latitude':self.domain_xy.latitude,
                                'longitude':self.domain_xy.longitude
                                    })
                            .drop(['surface','heightAboveGround'],errors='ignore')
                            )


            dss.append(ds_resampled)
            
        # ... merge the resampled datasets ...
        ds = xr.merge(dss)
        
        # ... and add the projection information back
        for name in ds:
            #ds[name].attrs = ds_works[name].attrs
            ds[name].attrs['GRIB_gridType']= 'lambert'
            ds[name].attrs['GRIB_gridDefinitionDescription']= 'Lambert conformal '
            ds[name].attrs['GRIB_LaDInDegrees'] = 72.0
            ds[name].attrs['GRIB_LoVInDegrees'] = 324.0
            ds[name].attrs['GRIB_DyInMetres']= 2500.0
            ds[name].attrs['GRIB_DxInMetres'] = 2500.0
            ds[name].attrs['GRIB_Latin2InDegrees'] = 72.0
            ds[name].attrs['GRIB_Latin1InDegrees']= 72.0
            #ds[name].attrs['GRIB_latitudeOfSouthernPoleInDegrees'] = 0.0
            #ds[name].attrs['GRIB_longitudeOfSouthernPoleInDegrees'] = 0.0
            
        ds.to_netcdf(path)
        
        return(path)


class Fall3DBatch:

    def __init__(self, name:str, basefile:str|Fall3DInputFile, df:pd.DataFrame, basedir="mnt/runs",n_parallel = 5,
                 path_fall3d="/home/talfan/Software/Fall3D_local/fall3d/bin/Fall3d.r8.x"):
        """Initialise a new batch run object
        """

        # name has to be a valid directory name
        # https://stackoverflow.com/questions/59672062/elegant-way-in-python-to-make-sure-a-string-is-suitable-as-a-filename
        ok = ".-_0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        if not all([ c in ok for c in name]):
            raise TypeError("Name variable contains characters illegal in a path")
        
        
        self.name = name
        self.basefile = basefile
        self.df = df
        self.basedir = basedir
        self.n_parallel = n_parallel
        self.path_fall3d = path_fall3d

    def initialise(self):
        """Initialise a new batch run - create directories
        """

        # make the path that will itself contain a directory for each run
        self.path = os.path.join(self.basedir,self.name)
        
        # we don't want to risk overwriting previous batch runs by accident
        if os.path.isdir(self.path):
        	raise ValueError("Path already exists for batch name: "+self.name)
        
        os.mkdir(self.path)
        
        # save the dataframe containing the specification for each run
        self.df_file = os.path.join(self.path, self.name+".csv")
        
        if os.path.exists(self.df_file):
        	raise ValueError("File already exists for batch name: "+self.name)
        
        self.df.to_csv(self.df_file)

        # create directory and input file for each run
        self.input_files = []
        
        # each run is a row in the dataframe, so we iterate over them...
        for i, r in self.df.iterrows():
            
            # ... convert each row to a dict ...
            rdict =r.to_dict()

            # ... now the update function for a Fall3DinputFile object
            # takes a nexted dict, with one subdict for each sectiom.
            # So we restructure the row dict into a nested dict:
            update = {}
            
            for key, value in rdict.items():
            
                section, attribute = key.split('.')
                
                if section in update.keys():
            
                    update[section][attribute] = value
            
                else:
                    update[section] = {attribute:value}

            # now we have our nested dict, we make a copy of our base file
            # and update it with the new values. This reinitialises the object
            # so any update that is of wrong type / invalid value / clashes with
            # another setting in the inputfile should be caught
        
            newf = copy.deepcopy(self.basefile)
        
            newf.update(update)

            # now we need the folder to store this run in.
            # This is just the basepath we defined earlier
            # plus the uuid, which is the indes of our row, i:
        
            new_dir = os.path.join(self.path,str(i))
            
            if os.path.isdir(new_dir):
            	raise ValueError("Path already exists: "+new_dir)
            
            os.mkdir(new_dir)
            
            # and then save the Fall3D input file to that folder:
            input_file = os.path.join(new_dir,str(i)+".inp")
            
            if os.path.exists(input_file):
            	raise ValueError("File already exists: "+input_file)
            
            newf.to_file(input_file)

            # save input file
            self.input_files.append(input_file)

    def get_meteo_data(self):
        """
        """

        for file in tqdm(self.input_files):

            f3if = Fall3DInputFile.from_file(file)
            
            meteosource = get_MeteoSource(f3if)
        
            meteosource.get_fall3d_input(
                    #time_utc=f3if.time_utc, 
                    #grid=f3if.grid, 
                    #meteo_data=f3if.meteo_data
                    f3if
                )

    def oldrun(self):
    


        for file in tqdm(self.input_files):

            subprocess.run([self.path_fall3d, "ALL",file]) 
            
    def run(self):
        

        processes = { }

        files = copy.deepcopy(self.input_files)

        # initialise first n_parallel runs

        num_starting_runs = min(self.n_parallel, len(files))

        for i in range(num_starting_runs): #self.n_parallel):
            file = files.pop()
            processes[i] = subprocess.Popen([self.path_fall3d,"All",file])

        # while there is more than 1 file left ....
        while len(files)>0:
        
            # ... iterate over each parallel slot ...
            for i, p in processes.items():
            
                # ... and when that run has finished ...
                if p.poll() is not None:
                
                    # ... get the next run in the list ...
                    file = files.pop()
                    
                    #print(len(files))
                    # ... put it in that slot and start it ...
                    processes[i] = subprocess.Popen([self.path_fall3d,"All",file])
            
        # ... and once we've used up all the files we just need to wait until the last one is finished
        finished = False
       
        while not finished:
       
            # the batch is finished when all of the last Popen objects stop returning None
            finished  = all([(p.poll() is not None) for i, p in processes.items()])
           
        print("Batch completed")
       
            
            
    def get_meteo_and_run(self): 
    
    	for file in tqdm(self.input_files):
            f3if = Fall3DInputFile.from_file(file)
            
            print("********************************************************************")
            print("FETCHING METEO DATA")
            print("********************************************************************")
            meteosource = get_MeteoSource(f3if)
            
            print("********************************************************************")
            print("RUN FALL3D")
            print("********************************************************************")
        
            meteosource.get_fall3d_input(
                    #time_utc=f3if.time_utc, 
                    #grid=f3if.grid, 
                    #meteo_data=f3if.meteo_data
                    f3if
                )
                
            subprocess.run([self.path_fall3d, "All",file]) 

            



class RawApi:
    """Python implementation of Umhverfisstonfnum's web API, see 
        https://api.ust.is/aq for details
    """

    def __init__(self):
        pass
        

    def get(self, url):
        
        with urllib.request.urlopen(url) as u:
            
            data = json.load(u)

        return(data)

    
    def getLatest(self):

        return self.get("https://api.ust.is/aq/a/getLatest")
        

    def getCurent(self, local_id='STA-IS0005A'):

        base_url = "https://api.ust.is/aq.php/a/getCurrent/id/"

        return self.get(base_url + local_id)
        
       
    def getDate(self, date="2018-01-01"):
        print("fetching date", date, "from Ust api")
        
        base_url = "https://api.ust.is/aq/a/getDate/date/"

        return self.get(base_url + date)

            
    def getStations(self):

        return self.get("https://api.ust.is/aq/a/getStations")
        

    def getStation(self, local_id='STA-IS0005A'):

        base_url = "https://api.ust.is/aq.php/a/getStation/id/"

        return self.get(base_url + local_id)


class Api:
    """Fetches data from Umhverfisstofnum's web API, facilitating search
    by date, species and latlon bounding box, and returns the data combined 
    with metadata
    """

    def __init__(self, local_storage=""):

        self.local_storage = local_storage
        
        self.rawapi = RawApi()
        
        self.get_stations()

   

    def get_stations(self):
        
        data = self.rawapi.getStations()
        
        #  ... the data is a list of dicts, so easy to put into a dataframe ...
        df  = pd.DataFrame(data)
        
        df['latitude'] = df['latitude'].astype(float)

        df['longitude'] = df['longitude'].astype(float)

        df['activity_begin'] = df['activity_begin'].apply(lambda r: pd.to_datetime(r))

        df['activity_end'] = df['activity_end'].apply(lambda r: pd.to_datetime(r))
        
        df.loc[
                np.isnat(df['activity_end']),
                'activity_end'
                ] = datetime.datetime.now()
    

        
        # ... except the species are a list in a single column - we want this
        # in a format easier to search (one column for each species with 
        # values of True or False for each station. To do this we need
        # a unique list of all the available species
        all_params = []
        
        for params in df['parameters'].values:
            
            params = params[1:-1].split(',')
            
            all_params.extend(params)
        
        # once we have a list of every occurrence of a species
        # we turn it into a set to get a list of unique elements
        all_params = set(all_params)

        self.stations = df
        
        self.parameters = all_params
        

    def json_to_xarray(self, data):
        
        dss = []

        attrs_for_names = {}
        # json data comes by day, indexed by station

        # first we iterate over the stations ...
        for station, station_data in data.items():

            # .. we pop the full name and station id ...
            
            name = station_data.pop('name')
            
            local_id = station_data.pop('local_id')

            # .. saving them for storage in dataray attributes later ..

            attrs_for_names[local_id] = name

            # ... which leaves the data, indexed by parameter
            # (SO2, PM2.5, etc...)
        
            parameters_data = station_data['parameters']

            # we iterate over the different parameters ...
            for param, param_data in parameters_data.items():

                # ... and for each we get the metadata ...
                unit = param_data.pop('unit')
                resolution = param_data.pop('resolution')

                # ... and get the other metadata we need ...
                lat = self.stations[ self.stations['local_id']==local_id]['latitude'].item()
                lon = self.stations[ self.stations['local_id']==local_id]['longitude'].item()

                # ... what remains is the data, which we put in a dataframe ...
                df = pd.DataFrame(param_data).T

                # ... convert data from strings to the correct datatype ...
                df['value'] = df['value'].astype(float)
                df['verification'] =df['verification'].astype(int)
                df['endtime'] = df['endtime'].apply(lambda r: pd.to_datetime(r))

                # ... and fix the column names so they are unique and can be merged with 
                # data from other days ...
                full_name_for_value = "#".join([local_id, param, 'value'])
                full_name_for_verification = "#".join([local_id, param, "verification"])
                
                df[full_name_for_value] = df['value']
                df[full_name_for_verification] = df['verification']

                del(df['value'])
                del(df['verification'])

                # .. we then set the index, so the data will merge correctly ...
                df = df.set_index("endtime")

                # ... convert to an xarray dataset ...
                ds = df.to_xarray()

                # ... add in the attributes ..
                attrs= {
                    'unit': unit, 
                    'resolution':resolution, 
                    'lat':lat, 
                    'lon':lon
                    }
                
                ds[full_name_for_value].attrs = attrs
                ds[full_name_for_verification].attrs = attrs

                
                dss.append(ds)

            ds_all = xr.merge(dss)

            ds_all.attrs = attrs_for_names

            ds_all.attrs['creation date'] = str(datetime.datetime.now())
        
        
        return ds_all

    def get_date(self, date):

        date_as_string = datetime.datetime.strftime(date,"%Y-%m-%d")

        output_filename = date_as_string + ".nc"
        
        output_path = os.path.join(self.local_storage, output_filename)

        if os.path.isfile(output_path):
            print(output_filename,"exists, loading from local storage.")

            data_as_xarray = xr.open_dataset(output_path)

        else:
            print(output_filename,"not found, fetching data using API")

            data_as_json = self.rawapi.getDate(date_as_string)
    
            data_as_xarray = self.json_to_xarray(data_as_json)
    
            data_as_xarray.to_netcdf(output_path)

        return(data_as_xarray)
        
        

    def search_stations(self, minlat, maxlat, minlon, maxlon, start:datetime, end:datetime, species:str):

        # raise an exception if the species isn't recognised ansd inform
        # the user of valid optionas
        if species not in self.parameters:
            
            raise Exception("species must be one of", self.parameters)

        # get a list of selected satations
        selected_stations = self.stations[
        
            (self.stations['latitude']>minlat)&
        
            (self.stations['latitude']<maxlat)&
        
            (self.stations['longitude']>minlon)&
            
            (self.stations['longitude']<maxlon)&
            
            (self.stations['activity_begin']<end)&
                
            (self.stations['activity_end']>start)&
            
            (self.stations['parameters'].apply(lambda r: species in r))            
        ]
        return selected_stations


    def get_data(self,start:datetime, end:datetime,  minlat=None, maxlat=None, minlon=None, maxlon=None, species=None):

        # raise an exception if the species isn't recognised ansd inform
        # the user of valid optionas
        if species not in self.parameters:
            
            raise Exception("species must be one of", self.parameters)

        
        # get the data covering the date interval
        duration = (end-start).days

        all_data = []

        all_attrs = {}
        
        for day in range(duration):
            
            date = start + datetime.timedelta(days=day)
            
            data_as_xarray = self.get_date(date)

            all_data.append(data_as_xarray)

            all_attrs = {**all_attrs, **data_as_xarray.attrs}

        all_data = xr.merge(all_data)

        all_data.attrs = all_attrs

        # now we select only the relevant stations

        # get a list of all the local_ids in our dataset ....
        names = list(all_data)
        num = len(set([name.split('#')[0] for name in names]))
        print(num, "stations between",start, "and", end)

        # filter for species
        if species is not None:
            names  = [ name for name in  names if  species in name.split('#')]
            num = len(set([name.split('#')[0] for name in names]))
            print(num, "stations measuring",species)


        # filter for extent
        if all([x is not None for x in [minlat, maxlat, minlon, maxlon]]):

            
            selected_stations = self.stations[
            
                (self.stations['latitude']>minlat)&
            
                (self.stations['latitude']<maxlat)&
            
                (self.stations['longitude']>minlon)&
                
                (self.stations['longitude']<maxlon)
            ]

            names = [name for name in names if name.split('#')[0] in selected_stations['local_id'].values]
            num = len(set([name.split('#')[0] for name in names]))


            print(num,"within bounding box",minlat, maxlat, minlon,maxlon)

    
        return all_data[names]




    def plot_data(self,data):
                
        plt.figure("Test Map",figsize=(10,10))
        crs = ccrs.PlateCarree()
        
        ax0 = plt.subplot(211, projection=crs)
        #ax0.set_extent(extent, crs=crs)
        ax0.coastlines(resolution='10m',color='blue')
        ax0.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
        
        ax1 = plt.subplot(212)
        
        axs = [ax1, ax0]
        
        for name in data:
            
            local_id, species, type  = name.split('#')
            
            full_name  = data.attrs[local_id]
            
            label = " ".join([local_id, full_name])
            
            if type=='value':
                
                data[name].plot(ax=axs[0],label=label)
                
                lat = data[name].attrs['lat']
        
                lon = data[name].attrs['lon']
        
                axs[1].scatter([lon],[lat],label=label)
        
        axs[0].legend(bbox_to_anchor=(1.1, 1.05))
        axs[1].legend(bbox_to_anchor=(1.1, 1.05))




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

        df['meteo_data.meteo_data_file'] = self.base_dir + "/" + self.name + "/shared_meteo_data.nc" 

        # make sure everything we don't need is off
        df['model_output.output_3d_concentration'] = YesNo('no')
        df['model_output.output_3d_concentration_bins'] = YesNo('no')
        df['model_output.output_surface_concentration'] = YesNo('yes')
        df['model_output.output_column_load'] = YesNo('no')
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
        self.batch.initialise()

    def get_meteo_data(self):
        # ... iterates over the input files and gets the appropriate met data ....
        self.batch.get_meteo_data()

    def run(self):
        # ... iterates over every input file and runs Fall3D
        self.batch.run()

    
    def build_emulator(self):
        
    
        
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
        df_runs['meteo_data.meteo_data_file'] = self.base_dir + "/" + self.name + "/shared_meteo_data.nc" 

        # and finally, we need to  make sure everything we don't need is off
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




class Forecast:

    def __init__(self, emulator, observations):

        self.emulator = emulator

        self.observations = observations

    def get_stan_data(self):

        ds_puff = self.emulator.da_puff

        result = self.observations

        concs = []
        
        for name in result:
            
            local_id, species, variable = name.split("#")
        
            if variable == 'value':
                
                concs.append(result[name])
                
                concs[-1].name = local_id
        
        df_obs  = (
                    xr
                    .merge(concs)
                    .to_dataframe()
                    .unstack()
                    .reset_index()
                    .rename(columns={'level_0':'local_id','endtime':'date',0:'conc'})
                )
        
        df_obs = df_obs[
                            (df_obs['date']>= ds_puff['date'].values[0]) & 
                            (df_obs['date']<=ds_puff['date'].values[-1])
                        ]
        
        local_ids  = ds_puff.local_id.values
        
        dates = ds_puff.date.values
        
        source_starts = ds_puff['source.source_start'].values
        
        sss = pd.to_datetime(ds_puff["date"].values[0])
        
        sss = datetime.datetime(year=sss.year,month=sss.month, day=sss.day, hour=sss.hour)
        
        source_starts = [ sss + datetime.timedelta(hours = int(hour)) for hour in source_starts]
        
        # round to nearest hour
        dates = np.array(np.array(dates, dtype='datetime64[h]'), dtype='datetime64[s]')
        
        dates  = [pd.Timestamp(d) for d in dates]
        
        lookup_local_id = dict(zip(local_ids, range(len(local_ids))))
        
        lookup_dates = dict(zip(dates, range(len(dates))))
        
        lookup_source_starts = dict(zip(source_starts, range(len(source_starts))))
        
        df_obs['i'] = df_obs['local_id'].apply(lambda r: lookup_local_id[r])
        
        df_obs['j'] = df_obs['date'].apply(lambda r: lookup_dates[r])
        
        df_obs = df_obs.dropna().reset_index()
        
        print("FILTER FOR NEGATIVE SO2 CONC DISABLED!!")
        #df_obs  = df_obs[df_obs['conc']>0].reset_index()

        self.stan_data = {
            
                # dimensions of 'eumlator' array
                'N_puf': len(ds_puff['source.source_start']),
                'N_stn': len(ds_puff.local_id),
                'N_hrs': len(ds_puff.date),
                'N_hts': len(ds_puff['source.height_above_vent']),
            
                # The 'emulator' array. Remember order! [N_puf, N_stn, N_hrs, N_hts] 
                'puffs': ds_puff.values,
            
                # offset and scale for the emulator values
                'height_min': ds_puff['source.height_above_vent'].values[0],
                'height_delta': ds_puff['source.height_above_vent'].values[1]-ds_puff['source.height_above_vent'].values[0],
            
                # number of observations
                #'N_oc':len(df_synth_obs),
                'N_oc':len(df_obs),
                # observations
                #'obs_conc':df_synth_obs['conc plus noise'].values, #/1e6, # micro g -> g
                'obs_conc':df_obs['conc'].values/1e6, # micro g -> g,
                #'ij_conc': df_synth_obs[['i','j']].values+1, # REMEMBER +1 because Stan counts from 1!!
                'ij_conc':df_obs[['i','j']].values+1,
            
                
                # limits for parameter values
                'height_lower': 100,#ds_puff['source.height_above_vent'].values[1]+10,
                'height_upper': 1000,# ds_puff['source.height_above_vent'].values[-2]-10,
            
                'height_mu_loc': 600.0,
                'height_mu_scale':500.0,
                
                'height_sigma_loc': 500.0, 
                'height_sigma_scale':500.0,
                
                
                
                'flux_lower': 0.0,
                'flux_upper':200.0,
            
                'flux_mu_loc':65.0,
                'flux_mu_scale':10.0,
            
                'flux_sigma_loc':30.0,
                'flux_sigma_scale':10.0,
                
            
                
                'sigma_conc_lower': 0.0,
                'sigma_conc_upper': 1e-4,
            
                'sigma_conc_loc':1e-6,
                'sigma_conc_scale':1e-5,
                

                
                'POSTERIOR':0
            }

    def load_stan_model(self):
        with open("model.stan") as f:
    
            stan_code = f.readlines()
        
        self.stan_code  = "".join(stan_code)
        
    def run_prior(self):

        self.stan_data['LIKELIHOOD'] = 0
        self.prior_model = stan.build(program_code=self.stan_code, data=self.stan_data)
        self.prior_fit = self.prior_model.sample(num_chains=4, num_samples=500, delta=0.99)
        

    def run_posterior(self):

        self.stan_data['LIKELIHOOD'] = 1
        self.posterior_model = stan.build(program_code=self.stan_code, data=self.stan_data)
        self.posterior_fit = self.posterior_model.sample(num_chains=4, num_samples=500, delta=0.99)

    def get_inference_data(self):

        ds_puff = self.emulator.da_puff
        
        source_start = [ pd.to_datetime(ds_puff["date"].values[0]) + datetime.timedelta(hours = int(hour)) for hour in ds_puff["source.source_start"].values]

        
        self.inference_data = az.from_pystan(
            posterior = self.posterior_fit,
            prior = self.prior_fit,
            prior_model = self.prior_model,
            posterior_model = self.posterior_model,
            posterior_predictive = ["obs_hat"], 
            prior_predictive = ["obs_hat"],
            observed_data = ["obs_conc"],
            coords = {
                "local_id":ds_puff["local_id"].values,
                "date":ds_puff["date"].values,
                "source_start": source_start,
            },
            dims = {
                "conc_matrix_hat":["chain","draw","local_id","date"],
                "conc_matrix":["chain","draw","local_id","date"],
                "height0":["chain","draw","source_start"],
                "height":["chain","draw","source_start"],
                "flux0":["chain","draw","source_start"],
                "flux":["chain","draw","source_start"],
                "puffs_interpolated":["chain","draw","local_id","date"],
                
                "heightb":["chain","draw","source_start"],
                "fluxb":["chain","draw","source_start"],
        
            }
        )
        
    def summarize_prior(self):

        return az.summary(self.inference_data,group='prior')

    def summarize_posterior(self):

        return az.summary(self.inference_data,group='posterior')

    def prior_traceplot(self):
        
        az.plot_trace(self.inference_data.prior,var_names=[
            'height',
            'flux',
            'sigma_conc',
            'height_mu',
            'height_sigma',
            'flux_mu',
            'flux_sigma'
        ])
        
        plt.tight_layout()

    def posterior_traceplot(self):
        
        az.plot_trace(self.inference_data.posterior,var_names=[
            'height',
            'flux',
            'sigma_conc',
            'height_mu',
            'height_sigma',
            'flux_mu',
            'flux_sigma'
        ])
        
        plt.tight_layout()
        


        
    
                
                

        







    


