#!/usr/bin/env python3

import argparse
import sys
import pathlib
import numpy as np
import xarray as xr

# IMPORTANT: edit this path to point to the folder where you built pyhommexx
PYHOMMEXX_LIB_PATH="/home/lbertag/workdir/e3sm/e3sm-homme-build/amdis/gcc/serial/release/src/theta-l_kokkos/pyhommexx/"
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

    Run 10 theta model steps using timestep 300s and namelist /path/to/namelist.nl

        > ./{exec_name} -n 10 -dt 300 -n /path/to/namelist.nl
""")

    # The timestep
    parser.add_argument("-dt","--dt", type=float, required=True,
                        help="Dynamics timestep")
    parser.add_argument("-n","--nstep", type=int, required=True,
                        help="Numver of dynamics timestep")
    #  # The name of the nc files where to grab data from
    parser.add_argument("-nml","--namelist", type=str, required=True,
                        help="Path to the runtime namelist file")
    parser.add_argument("-p","--perturb", type=float, default=0,
                        help="Relative perturbation level to add to the IC")

    return parser.parse_args(args[1:])

###############################################################################
def run_theta(dt,nstep,namelist,perturb):
###############################################################################

    pyhommexx.init_session()

    # Read namelist parameters
    pyhommexx.read_params(namelist)
    params = pyhommexx.get_params()

    ne = params['ne']
    ngp = params['np']
    nlev = params['nlev']

    # Initialize model
    pyhommexx.model_init()
    nelemd = pyhommexx.get_nelemd(); # Not available until prim_init decomposes the grid

    u = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
    v = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)

    # Get info needed to save unique points only
    num_unique_pts = np.ndarray([nelemd],dtype=np.int32)
    unique_i = np.ndarray([nelemd,ngp*ngp],dtype=np.int32)
    unique_j = np.ndarray([nelemd,ngp*ngp],dtype=np.int32)

    pyhommexx.get_num_unique_pts(num_unique_pts)
    pyhommexx.get_unique_pts(unique_i,unique_j)

    ncol = np.sum(num_unique_pts)
    # Get initial state, perturb it, then send it back
    pyhommexx.get_state_var(u,"u")
    pyhommexx.get_state_var(v,"v")
    if perturb>0:
        factor = 1 +  perturb * np.random.normal(size=u.shape)
        u *= factoperturb
        v *= factor
        pyhommexx.set_state_var(u,"u")
        pyhommexx.set_state_var(v,"v")

    # Run hommexx
    print ("Running hommexx. Grid specs:")
    print(f" ne: {ne}")
    print(f" ngp: {ngp}")
    print(f" nlev: {nlev}")
    print(f" nelemd: {nelemd}")
    print(f" ncol: {ncol}")


    for n in range(nstep):
        pyhommexx.forward(dt)

    # Retrieve final state
    pyhommexx.get_state_var(u,"u")
    pyhommexx.get_state_var(v,"v")

    u_unique = np.ndarray([nlev,ncol],dtype=np.float64)
    v_unique = np.ndarray([nlev,ncol],dtype=np.float64)
    icol = 0
    for ie in range(nelemd):
        for n in range (num_unique_pts[ie]):
            ip = unique_i[ie,n]
            jp = unique_j[ie,n]
            u_unique[:,icol] = u[ie,ip,jp,:]
            v_unique[:,icol] = v[ie,ip,jp,:]
            icol += 1

    u_with_time = np.expand_dims(u_unique,0)
    v_with_time = np.expand_dims(v_unique,0)

    ds = xr.Dataset()

    ds['u'] = xr.DataArray(u_with_time[:,:,:],
                           dims=['time', 'lev', 'ncol'],
                           coords={'time': np.arange(1),
                                   'lev': np.arange(nlev),
                                   'ncol': np.arange(ncol)})
    ds['v'] = xr.DataArray(v_with_time[:,:,:],
                           dims=['time', 'lev', 'ncol'],
                           coords={'time': np.arange(1),
                                   'lev': np.arange(nlev),
                                   'ncol': np.arange(ncol)})
    ds.to_netcdf('pyhommexx.nc')

    # Finalize hommexx
    pyhommexx.finalize()

###############################################################################
if (__name__ == "__main__"):
    run_theta(**vars(parse_command_line(sys.argv)))
