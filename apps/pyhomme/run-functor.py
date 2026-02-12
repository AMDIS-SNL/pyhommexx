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

    #  # The name of the nc files where to grab data from
    parser.add_argument("-nml","--namelist", type=str, default='./namelist.nl',
                        help="Path to the runtime namelist file")
    parser.add_argument("-p","--perturb", type=float, default=0,
                        help="Max of a gaussian perturbation centered at (30N,0) to add to the zonal velocity")
    parser.add_argument("-s","--sigma", type=float, default=1,
                        help="Sigma of a gaussian perturbation centered at (30N,0) to add to the zonal velocity (in km)")
    parser.add_argument("-f","--functor", type=str, default='caar',
                        help="Functor to run")
    parser.add_argument("-dt","--dt", type=float, default=0.0,
                        help="Time step to use when running the functor")

    return parser.parse_args(args[1:])

###############################################################################
def run_theta(namelist,perturb,sigma,functor,dt):
###############################################################################

    # Sanity checks
    if sigma<=0:
        raise ValueError(f"Invalid value for sigma ({sigma}). It should be strictly positive.")
    if sigma<=0:
        raise ValueError(f"Invalid value for dt ({dt}). It should be strictly positive.")

    pyhommexx.init_session(do_print_to_screen=False)

    # Read namelist parameters
    pyhommexx.read_params(namelist)
    pyhommexx.set_params({"alloc_sphere_coords" : True})
    params = pyhommexx.get_params()

    ne = params['ne']
    ngp = params['np']
    nlev = params['nlev']

    # Initialize model
    pyhommexx.model_init()
    nelemd = pyhommexx.get_nelemd(); # Not available until prim_init decomposes the grid

    # Get info needed to save unique points only
    num_unique_pts = np.ndarray([nelemd],dtype=np.int32)
    unique_i = np.ndarray([nelemd,ngp*ngp],dtype=np.int32)
    unique_j = np.ndarray([nelemd,ngp*ngp],dtype=np.int32)

    pyhommexx.get_num_unique_pts(num_unique_pts)
    pyhommexx.get_unique_pts(unique_i,unique_j)

    ncol = np.sum(num_unique_pts)

    # Retrieve initial state
    u0 = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
    v0 = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
    u  = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
    v  = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
    du = np.ndarray([nelemd,ngp,ngp,nlev,1],dtype=np.float64)
    dv = np.ndarray([nelemd,ngp,ngp,nlev,1],dtype=np.float64)
    pyhommexx.get_state_var(u0,"u")
    pyhommexx.get_state_var(v0,"v")

    if perturb>0 and sigma>0:
        # Perturb initial meridional velocity with gaussian centered at lat=30N, lon=0,
        # with max_perturbation 5% and decay as a gaussian with sigma=10km
        pyhommexx.perturb_state_var("u",0.5,0,perturb,sigma*1e3)

        # Check perturbed state
        pyhommexx.get_state_var(u,"u")
        pyhommexx.get_state_var(v,"v")

        print (f"max(u-u0): {np.max(u-u0)}")
        print (f"max(v-v0): {np.max(v-v0)}") # Should be 0, as we only perturbed u

        # Check perturbed state initial sens
        pyhommexx.get_state_var_sens(du,"u")
        print (f"max(du): {np.max(du)}")

        pyhommexx.get_state_var_sens(dv,"u")
        print (f"max(dv): {np.max(dv)}") # Should be 0, as we only perturbed u

    # Run hommexx
    print ("Grid specs:")
    print(f" ne: {ne}")
    print(f" ngp: {ngp}")
    print(f" nlev: {nlev}")
    print(f" nelemd: {nelemd}")
    print(f" ncol: {ncol}")

    # Init dp3d
    pyhommexx.init_dp3d_from_ps()

    # Run functor
    rkparams = {'dt' : dt, 'update_tl' : True}
    pyhommexx.run_functor(functor,rkparams)

    pyhommexx.get_state_var(u,"u")
    pyhommexx.get_state_var(v,"v")
    pyhommexx.get_state_var_sens(du,"u")
    pyhommexx.get_state_var_sens(dv,"v")

    # Extract unique columns values for u/v and save them
    u_unique = np.ndarray([nlev,ncol],dtype=np.float64)
    v_unique = np.ndarray([nlev,ncol],dtype=np.float64)
    du_unique = np.ndarray([nlev,ncol,1],dtype=np.float64)
    dv_unique = np.ndarray([nlev,ncol,1],dtype=np.float64)
    icol = 0
    for ie in range(nelemd):
        for n in range (num_unique_pts[ie]):
            ip = unique_i[ie,n]
            jp = unique_j[ie,n]
            u_unique[:,icol] = u[ie,ip,jp,:]
            v_unique[:,icol] = v[ie,ip,jp,:]
            du_unique[:,icol,0] = du[ie,ip,jp,:,0]
            dv_unique[:,icol,0] = dv[ie,ip,jp,:,0]
            #  print(f"u[{icol},:]: {u_unique[:,icol]}")
            icol += 1

    u_with_time = np.expand_dims(u_unique,0)
    v_with_time = np.expand_dims(v_unique,0)
    du_with_time = np.expand_dims(du_unique,0)
    dv_with_time = np.expand_dims(dv_unique,0)

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
    ds['du'] = xr.DataArray(du_with_time[:,:,:],
                            dims=['time', 'lev', 'ncol', 'nsens'],
                            coords={'time': np.arange(1),
                                    'lev': np.arange(nlev),
                                    'ncol': np.arange(ncol),
                                    'nsens': np.arange(1)})
    ds['dv'] = xr.DataArray(dv_with_time[:,:,:],
                            dims=['time', 'lev', 'ncol', 'nsens'],
                            coords={'time': np.arange(1),
                                    'lev': np.arange(nlev),
                                    'ncol': np.arange(ncol),
                                    'nsens': np.arange(1)})
    ds.to_netcdf('pyhommexx.nc')

    # Finalize hommexx
    pyhommexx.finalize()

###############################################################################
if (__name__ == "__main__"):
    run_theta(**vars(parse_command_line(sys.argv)))
