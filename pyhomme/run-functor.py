#!/usr/bin/env python3

import argparse
import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# Point PYHOMMEXX_LIB_PATH at the build directory that contains pyhommexx.*.so
sys.path.append(os.environ["PYHOMMEXX_LIB_PATH"])
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
                        help="Max of a gaussian perturbation centered at (30N,0) to add to the zonal velocity (in m/s)")
    parser.add_argument("-s","--sigma", type=float, default=1,
                        help="Sigma of a gaussian perturbation centered at (30N,0) to add to the zonal velocity (in km)")
    parser.add_argument("-f","--functor", type=str, default='caar',
                        help="Functor to run")
    parser.add_argument("-dt","--dt", type=float, default=0.0,
                        help="Time step to use when running the functor")

    return parser.parse_args(args[1:])


###############################################################################
def distance(lat, lon, lat0, lon0, rearth):
###############################################################################
    dx = lat - lat0
    dy = np.fmod(lon - lon0 + np.pi, 2 * np.pi) - np.pi

    a = np.sin(dx / 2) ** 2 + np.cos(lat) * np.cos(lat0) * (np.sin(dy / 2) ** 2)
    c = 2 * np.arcsin(np.sqrt(a))

    return rearth * c

###############################################################################
def gaussian_perturb(delta,pmax,sigma,lat,lon,lat0,lon0,rearth):
###############################################################################
    for ie in range(delta.shape[0]):
        for ip in range(delta.shape[1]):
            for jp in range(delta.shape[2]):
                d = distance(lat[ie,ip,jp],lon[ie,ip,jp],lat0,lon0,rearth) / 1000
                delta[ie,ip,jp,...] = pmax*np.exp(-np.power(d, 2) / (2 * np.power(sigma, 2)))

###############################################################################
def run_functor(namelist,perturb,sigma,functor,dt):
###############################################################################

    # Sanity checks
    if sigma<=0:
        raise ValueError(f"Invalid value for sigma ({sigma}). It should be strictly positive.")
    if sigma<=0:
        raise ValueError(f"Invalid value for dt ({dt}). It should be strictly positive.")

    # If you want to see homme's output during init, set arg to True
    pyhommexx.init_session(do_print_to_screen=False)

    # Ensure both real and dpfad are enabled
    # This will ensure that model_init also creates all versions of data structures and functors
    # templated on all requested scalar types
    pyhommexx.enable_scalar_type("real")
    pyhommexx.enable_scalar_type("dpfad")

    # Read namelist parameters
    pyhommexx.read_params(namelist)
    pyhommexx.set_params({"alloc_sphere_coords" : True})
    params = pyhommexx.get_params()

    ne = params['ne']
    ngp = params['np']
    nlev = params['nlev']

    # Initialize model
    pyhommexx.model_init()

    # Now that homme's long init output is over, you can toggle any homme output back ON (to see errors)
    #  pyhommexx.toggle_screen_output(True)

    # Print grid specs
    nelemd = pyhommexx.get_nelemd(); # Not available until prim_init decomposes the grid

    print ("Grid specs:")
    print(f" ne: {ne}")
    print(f" ngp: {ngp}")
    print(f" nlev: {nlev}")
    print(f" nelemd: {nelemd}")

    if perturb==0 or sigma==0 or dt==0:
        print("Input values for -dt/--dt, -p/--perturb, or -s/--sigma is 0. We won't perform any test.")
        pyhommexx.finalize()
        return

    lat = np.ndarray([nelemd,ngp,ngp],dtype=np.float64)
    lon = np.ndarray([nelemd,ngp,ngp],dtype=np.float64)
    pyhommexx.get_dyn_latlon(lat,lon)
    rearth = pyhommexx.get_phys_constant('rearth')

    # Vector used for perturbations
    lat0 = 0.5 # radians
    lon0 = 0   # radians
    delta = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
    gaussian_perturb(delta,perturb,sigma,lat,lon,lat0,lon0,rearth)

    #  print(delta)
    #  print (f"max(delta): {np.max(delta)}")
    #  pyhommexx.finalize()
    #  return
    # Init dp3d to ref values, and copy real state into dpfad state
    pyhommexx.init_dp3d_from_ps()
    pyhommexx.copy_state("real","dpfad")

    # We'll need to reset state later to u_ref*perturb, so keep a copy here
    u_ref = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
    pyhommexx.get_state_var(u_ref,"u","real",0)

    # Run functor with dpfad first, and with different real perturb later, to check derivs
    rkparams = {'dt' : dt}
    pyhommexx.perturb_state_var("u",delta,0,"dpfad") # perturb=0, but this inits Fad derivs
    pyhommexx.run_functor(functor,rkparams,"dpfad")

    # Retrieve sacado sensitivity
    dudp_fad = np.ndarray([nelemd,ngp,ngp,nlev,1],dtype=np.float64)
    dvdp_fad = np.ndarray([nelemd,ngp,ngp,nlev,1],dtype=np.float64)
    pyhommexx.get_state_var_dp_sens(dudp_fad,"u")
    pyhommexx.get_state_var_dp_sens(dvdp_fad,"v")

    # Run with real scalar, unperturbed and then perturbed
    pyhommexx.run_functor(functor,rkparams,"real")
    u0 = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
    v0 = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
    pyhommexx.get_state_var(u0,"u")
    pyhommexx.get_state_var(v0,"v")

    dudp_fd = []   # FD sensitivities
    dvdp_fd = []
    N = 8 # How many FD intervals
    factors = np.logspace(0,-N, num=N+1)
    for factor in factors:
        du  = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)
        dv  = np.ndarray([nelemd,ngp,ngp,nlev],dtype=np.float64)

        # Perturb initial meridional velocity with gaussian centered at lat=30N (0.5 rad), lon=0,
        # with max_perturbation factor*perturb and decay as a gaussian with std_dev=sigma
        pyhommexx.set_state_var(u_ref,"u","real",0)
        print(np.max(factor*delta))
        pyhommexx.perturb_state_var("u",delta,factor*perturb,"real")

        pyhommexx.run_functor(functor,rkparams,"real")

        # Get state and compute FD sens approx
        pyhommexx.get_state_var(du,"u","real",1)
        pyhommexx.get_state_var(dv,"v","real",1)

        du -= u0
        dv -= v0
        du /= factor*perturb
        dv /= factor*perturb
        dudp_fd.append(np.copy(du))
        dvdp_fd.append(np.copy(dv))

    print(f"max_u_perturb_factor: {[float(f*perturb) for f in factors]}")
    print(f"||DpFad(u)|| = {float(np.linalg.norm(dudp_fad))}")
    print(f"||FD(u)|| = {[float(np.linalg.norm(du)) for du in dudp_fd]}")
    print(f"||DpFad(u)-FD(u)|| = {[float(np.linalg.norm(dudp_fad[...,0]-du)) for du in dudp_fd]}")
    print(f"||DpFad(v)|| = {float(np.linalg.norm(dvdp_fad))}")
    print(f"||FD(v)|| = {[float(np.linalg.norm(dv)) for dv in dvdp_fd]}")
    print(f"||DpFad(v)-FD(v)|| = {[float(np.linalg.norm(dvdp_fad[...,0]-dv)) for dv in dvdp_fd]}")

    # Plot error
    err_u = [float(np.linalg.norm(dudp_fad[...,0]-du)) for du in dudp_fd]
    err_v = [float(np.linalg.norm(dvdp_fad[...,0]-dv)) for dv in dvdp_fd]
    plt.figure(figsize=(10, 6))
    plt.loglog(perturb*factors, err_u, label='Zonal', marker='o')
    plt.loglog(perturb*factors, err_v, label='Meridional', marker='s')

    plt.gca().invert_xaxis()

    # Add labels and title
    plt.xlabel('Perturb')
    plt.ylabel('||FAD - FD||')
    plt.title('Zonal and Meridional winds sensitivity error')
    plt.legend()

    # Show the plot
    plt.grid()
    plt.show()

    # Finalize hommexx
    pyhommexx.finalize()

###############################################################################
if (__name__ == "__main__"):
    run_functor(**vars(parse_command_line(sys.argv)))
