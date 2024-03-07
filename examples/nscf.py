# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import argparse
import matplotlib.pyplot as plt
from distutils.util import strtobool
from ase.calculators.espresso import Espresso
from qe_toolkit.io import ReadQeOut, ReadQeInp

# -----------------------------------------------------------------------------
# PARSE ARGUMENTS
# -----------------------------------------------------------------------------

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
    "--espresso_pwi",
    "-pwi",
    type=str,
    required=False,
    default="pw.pwi",
    help="Quantum Espresso input file.",
)

parser.add_argument(
    "--espresso_pwo",
    "-pwo",
    type=str,
    required=False,
    default="pw.pwo",
    help="Quantum Espresso output file.",
)

parser.add_argument(
    "--run_qe_bin",
    "-r",
    type=fbool,
    required=False,
    default=False,
)

parsed_args = parser.parse_args()

# -----------------------------------------------------------------------------
# NSCF
# -----------------------------------------------------------------------------

qe_inp = ReadQeInp(parsed_args.espresso_pwi)
input_data, pseudos, kpts, koffset = qe_inp.get_data_pseudos_kpts()
atoms = qe_inp.get_atoms()
if "calculation" in input_data:
    qe_out = ReadQeOut(parsed_args.espresso_pwo)
    atoms = qe_inp.update_atoms(qe_out.get_atoms())

if parsed_args.kpts != "":
    kpts_new = [int(i) for i in parsed_args.kpts.split(",")]
else:
    kpts_new = kpts

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

calc = Espresso(
    input_data=input_data,
    pseudopotentials=pseudos,
    kpts=kpts,
    koffset=koffset,
    label="nscf",
)

calc.set(input_data=input_data_nscf)
calc.set(kpts=kpts_new)
calc.write_input(atoms)

# -----------------------------------------------------------------------------
# RUN PROJWFC
# -----------------------------------------------------------------------------

if parsed_args.run_qe_bin is True:
    os.system(f"srun pw.x < nscf.pwi > nscf.pwo")

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
