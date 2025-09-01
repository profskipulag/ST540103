############################################################################
#
# Example use of the pyem package
#
############################################################################

# import the Emulator class from pyem
from pyem import Emulator

# we also need pyf3d (you can just copy the pyf3d directory over) ...
from pyf3d import Fall3DInputFile, YesNo

# ... various things from pycompss for running Fall3d ...
from pycompss.api.api import compss_wait_on, compss_barrier, compss_wait_on
from pycompss.api.task import task
from pycompss.api.IO import IO
from pycompss.api.constraint import constraint
from pycompss.api.binary import binary
from pycompss.api.prolog import prolog
from pycompss.api.epilog import epilog
from pycompss.api.parameter import *


# .. and a few other things
import datetime
import numpy as np
import copy
import os

############################################################################
#
#  Pycompss task definitions
#
###########################################################################


@binary(
        binary = "Fall3d.x",
        args = "ALL {{infile}}",
        working_dir="{{work_dir}}",
        fail_by_exit_value=True
    )
@task(
        infile=FILE_IN
    )
def run_fall3d(work_dir, infile):
    pass

def main():

    #############################################################################
    #
    # A simple interpolate-and-sum emulator for a linear forward model
    #
    ############################################################################

    # first we need the base Fall3D input file, which will be modified for each run used
    # to build the emulator ...

    # ... we load the basefile into memory and change a few things ..
    file = Fall3DInputFile.from_file("mnt/examples/default_so2_reykjanes2.inp")


    # ... we change the base date ...
    file.time_utc.update({
        'year':2021,
        'month':7,
        'day':18,
        'run_start':0,
        'run_end':48,
    })

    # ... update a few details about the meteorological data ...
    file.meteo_data.update({
        'dbs_end_meteo_data':48,
        'meteo_coupling_interval':3*60,
        'meteo_data_file':'/leonardo/home/userexternal/tbarnie0/test/ST540103/mnt/shared_meteo_data.nc',
        'meteo_data_dictionary_file':'/leonardo/home/userexternal/tbarnie0/test/ST540103/mnt/aux/CARRA.tbl',
        'meteo_levels_file':"/leonardo/home/userexternal/tbarnie0/test/ST540103/mnt/aux/L137_ECMWF.levels"

    })

    # ... add the file containing the locations of the specific ground stations we want to
    # keep track of ...
    file.model_output.update({
        'output_track_points_file':'/leonardo/home/userexternal/tbarnie0/test/ST540103/mnt/examples/stations.pts'
    })

    # ... some random other stuff just for the example ..
    file.model_physics.update({
        'limiter':'MINMOD'
    })

    # we want to make sure all stochastic options are set to "no" or "off", so that
    # the model will always give the same outputs for the same inputs. This is because
    # (1) uncertainty is handles by the Bayesian franmework we are using the emulator within, and
    # (2) we want to be able to compare emulator performance with that of a reall Fal3D run
    file.emsemble_postprocess.update({
        'postprocess_median':YesNo('no')
    })

    # ... and we set the mass flow rate to 1.0, because the emulator will scale this by the flux we want.
    file.source.update({
        'mass_flow_rate':"1.0"
    })


    # ... and lastly we need to order the metdata
    file.get_meteodata()


    # Now we have a starting file, we create an emulator object ...
    em = Emulator(
        # ... this is the base Fall3D file we just created - it sets the values for options the
        # emulatror doesn't change (everything except, heigh flux and source_start, basically)
        basefile=file,

        # the starting date for our emulator ...
        start=datetime.datetime(year=2021,month=7, day=18),

        # ... the duration that our emulatort will simulate, in hours ...
        hours=24,

        # ... the heights for whichy our emulatior will be run -
        # This gives the range and resolution of the look up table used
        # for interpolating the ground concentrations associated with a 'puf'
        # emitted at a specific height
        heights = np.arange(
            0, # start
            2550, # stop
            200 # interval (not the number of heights!)
        ).astype(float),

        # the name of the emulator specifies the directory within /mnt/runs/ in which the
        # emulator runs and the dataframe  describing them are saved
        name = "library_test2",

        # path to Fall3D
        path_fall3d = "Fall3d.x"
    )

    # First we initiualise the emulator, which creates a folder for every run
    # needed to build the look up table in mnt/runs/<name>
    print("Initialising runs ...")
    em.initialise()

    # fetches meteo data for each run
    print("Fetching meteodata ...")
    em.get_meteo_data()

    # run everything
    print("Running Fall3D")
    #em.run()
    MAX_SIMULTANEOUS_RUNS=10

    files = copy.deepcopy(em.batch.input_files)

    active_tasks = []
    for i, file in enumerate(files):

        file = "/leonardo/home/userexternal/tbarnie0/test/ST540103/"+file
        work_dir = os.path.dirname(file)


        print("work_dir", work_dir)
        print("file", file)
        active_tasks.append(run_fall3d(work_dir, file))

    compss_barrier() 
    active_tasks = compss_wait_on(active_tasks)
    print("Batch completed")
    
    # processes the data needed for emulating ground concentration on a lat lon grid
    print("Building surface emulator")
    em.build_surface_emulator()

    # processes the data needed for emulating total column gas content on a lat lon grid
    print("Building column mass emulator")
    em.build_col_mass_emulator()

    # runs Fall£d with a random timeseries of Eruption Source Parameters (ESPs) for comparison with emulator output
    print("Getting emulator diagnostics")
    em.get_emulator_diagnostics(height_low=125.0, height_high=500.0)
   
    # process data needed for emulating ground concentration at each station
    print("Building station emulator")
    em.build_station_emulator()

    # save emulator to netcdf
    print("Saving emulator to netcdf")
    em.to_netcdf("dt5404_example2.nc")


if  __name__ =="__main__":
   
    main()
