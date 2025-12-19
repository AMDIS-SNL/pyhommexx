#!/usr/bin/env python3

import argparse
import sys
import pathlib
import numpy as np
import xarray as xr

PYHOMMEXX_LIB_PATH="/home/lbertag/workdir/e3sm/e3sm-homme-build/amdis/serial/debug/src/theta-l_kokkos/pyhommexx/"
sys.path.append(PYHOMMEXX_LIB_PATH)
import pyhommexx

from mpi4py import MPI

###############################################################################
def parse_command_line(args):
###############################################################################
    exec_name = pathlib.Path(args[0]).name
    parser = argparse.ArgumentParser(
        usage=f"""\n{exec_name} <ARGS> [--verbose]
OR
{exec_name} --help

EXAMPLES:

    Run theta model using homme library /path/to/theta.so and namelist /path/to/namelist.nl

        > ./{exec_name} -l /path/to/theta.so -n /path/to/namelist.nl
""")

    # The timestep
    parser.add_argument("-dt","--dt", type=float, required=True,
                        help="Dynamics timestep")
    parser.add_argument("-n","--nstep", type=int, required=True,
                        help="Numver of dynamics timestep")
    #  # The name of the nc files where to grab data from
    parser.add_argument("-nml","--namelist", type=str, required=True,
                        help="Path to the runtime namelist file")

    return parser.parse_args(args[1:])

###############################################################################
def run_theta(dt,nstep,namelist):
###############################################################################

    comm = MPI.COMM_WORLD

    pyhommexx.init(comm,"namelist.nl")
    params = pyhommexx.get_params()

    nelemd = params['nelemd']
    ngp = params['np']
    nlev = params['nlev']

    # Get info needed to save unique points only
    num_unique_pts = np.ndarray([nelemd],dtype=np.int32)
    unique_i = np.ndarray([nelemd,ngp*ngp],dtype=np.int32)
    unique_j = np.ndarray([nelemd,ngp*ngp],dtype=np.int32)

    pyhommexx.get_num_unique_pts(num_unique_pts)
    pyhommexx.get_unique_pts(unique_i,unique_j)

    ncol = np.sum(num_unique_pts)

    v = np.ndarray([nelemd,2,ngp,ngp,nlev],dtype=np.float64)
    vthdp = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
    dp = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)

    print(f"nelemd: {nelemd}")
    print(f"ngp: {ngp}")
    print(f"nlev: {nlev}")
    pyhommexx.get_state(v,vthdp,dp)
    for n in range(nstep):
        pyhommexx.forward(dt)
    pyhommexx.get_state(v,vthdp,dp)
    pyhommexx.finalize()

    v_unique = np.ndarray([2,nlev,ncol],dtype=np.float64)
    icol = 0
    for ie in range(nelemd):
        for n in range (num_unique_pts[ie]):
            ip = unique_i[ie,n]
            jp = unique_j[ie,n]
            v_unique[:,:,icol] = v[ie,:,ip,jp,:]
            icol += 1

    v_with_time = np.expand_dims(v_unique,0)

    ds = xr.Dataset()

    ds['u'] = xr.DataArray(v_with_time[:,0,:,:],
                           dims=['time', 'lev', 'ncol'],
                           coords={'time': np.arange(1),
                                   'lev': np.arange(nlev),
                                   'ncol': np.arange(ncol)})
    ds['v'] = xr.DataArray(v_with_time[:,1,:,:],
                           dims=['time', 'lev', 'ncol'],
                           coords={'time': np.arange(1),
                                   'lev': np.arange(nlev),
                                   'ncol': np.arange(ncol)})
    ds.to_netcdf('pyhommexx.nc')

###############################################################################
if (__name__ == "__main__"):
    run_theta(**vars(parse_command_line(sys.argv)))
