# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import argparse
import matplotlib.pyplot as plt
from distutils.util import strtobool
from ase.calculators.espresso import Espresso
from qe_toolkit.io import read_pwi, read_pwo

# -------------------------------------------------------------------------------------
# PARSE ARGUMENTS
# -------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--kpts",
    "-kp",
    type=str,
    required=False,
    default="",
)

parser.add_argument(
    "--qe_pwi",
    "-pwi",
    type=str,
    required=False,
    default="pw.pwi",
    help="Quantum Espresso input file.",
)

parser.add_argument(
    "--qe_pwo",
    "-pwo",
    type=str,
    required=False,
    default="pw.pwo",
    help="Quantum Espresso output file.",
)

parser.add_argument(
    "--mpi_cmd",
    "-mc",
    type=str,
    required=False,
    default="module load intel/2020.1 openmpi/4.0.3 && mpirun",
)

parser.add_argument(
    "--run_qe_bin",
    "-r",
    type=fbool,
    required=False,
    default=False,
)

parsed_args = parser.parse_args()

# -------------------------------------------------------------------------------------
# NSCF
# -------------------------------------------------------------------------------------

# Read pwi input file.
atoms = read_pwi(filename=parsed_args.qe_pwi)
input_data = atoms.info["input_data"]
pseudopotentials = atoms.info["pseudopotentials"]
kpts = atoms.info["kpts"]
koffset = atoms.info["koffset"]
# Read pwo output file.
atoms = read_pwo(filename=parsed_args.qe_pwo, filepwi=parsed_args.qe_pwi)
# Update kpts.
if parsed_args.kpts == "auto":
    kpts_new = [int(60 / atoms.cell.lengths()[ii]) for ii in range(3)]
elif parsed_args.kpts != "":
    kpts_new = [int(ii) for ii in parsed_args.kpts.split(",")]
else:
    kpts_new = kpts
# Update input data.
input_data_nscf = input_data.copy()
input_data_nscf.update({
    "restart_mode": "from_scratch",
    "calculation": "nscf",
    "occupations": "tetrahedra_opt",
    "conv_thr": 1e-8,
    "nosym": True,
    "verbosity": "high",
    "diago_full_acc": True,
})
input_data_nscf.pop("smearing", None)
input_data_nscf.pop("degauss", None)
input_data_nscf.pop("max_seconds", None)
# Write input file for nscf.
calc = Espresso(
    input_data=input_data,
    pseudopotentials=pseudopotentials,
    kpts=kpts,
    koffset=koffset,
    label="nscf",
)
calc.set(input_data=input_data_nscf)
calc.set(kpts=kpts_new)
calc.write_input(atoms)

# -------------------------------------------------------------------------------------
# RUN PROJWFC
# -------------------------------------------------------------------------------------

if parsed_args.run_qe_bin is True:
    os.system(parsed_args.mpi_cmd + " pw.x < nscf.pwi > nscf.pwo")

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
