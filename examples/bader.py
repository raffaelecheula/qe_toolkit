#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import argparse
from distutils.util import strtobool
from qe_toolkit.io import ReadQeInp
from qe_toolkit.pp import write_pp_input, read_bader_charges

# -----------------------------------------------------------------------------
# PARSE ARGUMENTS
# -----------------------------------------------------------------------------

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
    "--qe_path",
    "-qe",
    type=str,
    required=False,
    default="",
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
    nargs=1,
    required=False,
    default="off",
)

parser.add_argument(
    "--espresso_pwi",
    "-pwi",
    type=str,
    required=False,
    default="pw.pwi",
    help="Quantum Espresso input file.",
)

parsed_args = parser.parse_args()

# -----------------------------------------------------------------------------
# READ INPUT FILE
# -----------------------------------------------------------------------------

if parsed_args.postprocess is True:
    qe_inp = ReadQeInp(parsed_args.espresso_pwi)
    atoms = qe_inp.get_atoms()

# -----------------------------------------------------------------------------
# CHANGE DIRECTORY
# -----------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.makedirs("bader", exist_ok=True)
    os.chdir("bader")
    pw_out_dir = "../"
else:
    pw_out_dir = "./"

# -----------------------------------------------------------------------------
# PRINT BADER INPUT
# -----------------------------------------------------------------------------

if parsed_args.write_input is True:

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

# -----------------------------------------------------------------------------
# RUN PP AND BADER
# -----------------------------------------------------------------------------

if parsed_args.run_qe_bin is True:
    os.system(f"{parsed_args.qe_path}pp.x < pp_val.pwi > pp_val.pwo")
    os.system(f"{parsed_args.qe_path}pp.x < pp_all.pwi > pp_all.pwo")

# -----------------------------------------------------------------------------
# POSTPROCESS
# -----------------------------------------------------------------------------

if parsed_args.postprocess is True:
    os.system(
        f"bader pp_val.cube -ref pp_all.cube -vac {parsed_args.vacuum} > bader.out"
    )
    read_bader_charges(atoms=atoms, filename="ACF.dat")

# -----------------------------------------------------------------------------
# CHANGE DIRECTORY
# -----------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.chdir("..")

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
