#!/usr/bin/env python3

# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import argparse
from distutils.util import strtobool
from qe_toolkit.io import read_pwi
from qe_toolkit.pp import write_pp_input, read_bader_charges

# -------------------------------------------------------------------------------------
# PARSE ARGUMENTS
# -------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--change_dir",
    "-c",
    type=fbool,
    nargs=1,
    required=False,
    default=True,
)

parser.add_argument(
    "--write_input",
    "-w",
    type=fbool,
    nargs=1,
    required=False,
    default=True,
)

parser.add_argument(
    "--run_qe_bin",
    "-r",
    type=fbool,
    nargs=1,
    required=False,
    default=True,
)

parser.add_argument(
    "--mpi_cmd",
    "-mp",
    type=str,
    required=False,
    default="module load intel/2020.1 openmpi/4.0.3 && mpirun",
)

parser.add_argument(
    "--postprocess",
    "-p",
    type=fbool,
    nargs=1,
    required=False,
    default=True,
)

parser.add_argument(
    "--vacuum",
    "-v",
    type=str,
    required=False,
    default="off",
)

parser.add_argument(
    "--pw_pwi",
    "-pwi",
    type=str,
    required=False,
    default="pw.pwi",
    help="Quantum Espresso input file.",
)

parsed_args = parser.parse_args()

# -------------------------------------------------------------------------------------
# READ INPUT FILE
# -------------------------------------------------------------------------------------

if parsed_args.postprocess is True:
    atoms = read_pwi(filename=parsed_args.pw_pwi)

# -------------------------------------------------------------------------------------
# CHANGE DIRECTORY
# -------------------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.makedirs("bader", exist_ok=True)
    os.chdir("bader")
    pw_out_dir = "../"
else:
    pw_out_dir = "./"

# -------------------------------------------------------------------------------------
# PRINT BADER INPUT
# -------------------------------------------------------------------------------------

if parsed_args.write_input is True:
    # Write pp and plot input files.
    pp_data = {
        "prefix": "pwscf",
        "outdir": pw_out_dir + "calc",
        "filplot": "filplot_val",
        "plot_num": 0,
    }
    plot_data = {
        "nfile": 1,
        "iflag": 3,
        "output_format": 6,
        "fileout": "pp_val.cube",
    }
    write_pp_input(pp_data, plot_data, filename="pp_val.pwi")
    pp_data.update({"filplot": "filplot_all", "plot_num": 21})
    plot_data.update({"fileout": "pp_all.cube"})
    write_pp_input(pp_data, plot_data, filename="pp_all.pwi")

# -------------------------------------------------------------------------------------
# RUN PP AND BADER
# -------------------------------------------------------------------------------------

if parsed_args.run_qe_bin is True:
    os.system(parsed_args.mpi_cmd + " pp.x < pp_val.pwi > pp_val.pwo")
    os.system(parsed_args.mpi_cmd + " pp.x < pp_all.pwi > pp_all.pwo")

# -------------------------------------------------------------------------------------
# POSTPROCESS
# -------------------------------------------------------------------------------------

if parsed_args.postprocess is True:
    # Run bader calculation.
    os.system(
        f"bader pp_val.cube -ref pp_all.cube -vac {parsed_args.vacuum} > bader.out"
    )
    # Read Bader charges.
    read_bader_charges(atoms=atoms, filename="ACF.dat", filename_out="charges.txt")

# -------------------------------------------------------------------------------------
# CHANGE DIRECTORY
# -------------------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.chdir("..")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
