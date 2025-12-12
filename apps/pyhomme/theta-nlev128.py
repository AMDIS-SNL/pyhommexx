#!/usr/bin/env python3

import argparse
import sys
import pathlib

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

\033[1mEXAMPLES:\033[0m

    \033[1;32m# Run theta model using homme library /path/to/theta.so and namelist /path/to/namelist.nl

        > ./{exec_name} -l /path/to/theta.so -n /path/to/namelist.nl
""")

    #  # The name of the nc files where to grab data from
    #  parser.add_argument("-n","--namelist", type=str, required=True,
    #                      help="Path to the runtime namelist file")

    return parser.parse_args(args[1:])

###############################################################################
def run_theta():
###############################################################################

    comm = MPI.COMM_WORLD

    pyhommexx.init(comm)
    pyhommexx.forward()
    pyhommexx.finalize()

###############################################################################
if (__name__ == "__main__"):
    run_theta(**vars(parse_command_line(sys.argv)))
