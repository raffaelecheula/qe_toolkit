# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import signal
import argparse
import numpy as np
from time import strftime
from distutils.util import strtobool
from ase.units import kB
from ase.vibrations import Vibrations
from ase.thermochemistry import HarmonicThermo
from ase.calculators.espresso import Espresso
from ase.calculators.socketio import SocketIOCalculator
from qe_toolkit.io import ReadQeInp, ReadQeOut
from qe_toolkit.utils import get_atoms_not_fixed

# -----------------------------------------------------------------------------
# CONTROL
# -----------------------------------------------------------------------------

MAX_SECONDS = 55050
signal.alarm(MAX_SECONDS)

# 'all' | 'only_relaxed' | indices separated by , | slices of indices
atoms_to_vib_default = "only_relaxed"

repetitions_default = "1,1,1"

# -----------------------------------------------------------------------------
# PARSER
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--run_vib",
    "-rv",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--write_hessian",
    "-wh",
    type=fbool,
    required=False,
    default=False,
)

parser.add_argument(
    "--harmonic_thermo",
    "-ht",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--temperature",
    "-te",
    type=float,
    required=False,
    default=500+273.15,
)

parser.add_argument(
    "--atoms_to_vib",
    "-av",
    type=str,
    required=False,
    default=atoms_to_vib_default,
)

parser.add_argument(
    "--repetitions",
    "-re",
    type=str,
    required=False,
    default=repetitions_default,
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

parsed_args = parser.parse_args()

atoms_to_vib = parsed_args.atoms_to_vib
repetitions = [int(ii) for ii in parsed_args.repetitions.split(",")]

# -----------------------------------------------------------------------------
# READ QE INPUTS
# -----------------------------------------------------------------------------

qe_inp = ReadQeInp(parsed_args.espresso_pwi)
input_data, pseudos, kpts, koffset = qe_inp.get_data_pseudos_kpts()
atoms = qe_inp.get_atoms()
if "calculation" in input_data:
    qe_out = ReadQeOut(parsed_args.espresso_pwo)
    atoms = qe_inp.update_atoms(qe_out.get_atoms())

if parsed_args.repetitions != "1,1,1":
    atoms *= repetitions
    kpts = [int(np.ceil(kpts[ii] / repetitions[ii])) for ii in range(len(kpts))]

input_data.update({
    "calculation": "scf",
    "restart_mode": "from_scratch",
    "tprnfor": True,
    "tstress": True,
})

calc = Espresso(
    input_data=input_data, pseudopotentials=pseudos, kpts=kpts, koffset=koffset,
)
calc.label = "vib"
calc.command = "mpirun pw.x -in PREFIX.pwi > PREFIX.pwo"

atoms.calc = calc

# -----------------------------------------------------------------------------
# ATOMS TO VIB
# -----------------------------------------------------------------------------

axis_dict = {"x": 0, "y": 1, "z": 2}

if atoms_to_vib == "all":
    indices = [a.index for a in atoms]

elif atoms_to_vib == "only_relaxed":
    indices = get_atoms_not_fixed(atoms)

elif "," in atoms_to_vib:
    indices = [int(ii) for ii in atoms_to_vib.split(",")]

elif ":" in atoms_to_vib:
    slices = atoms_to_vib.split(":")
    for ii in (0, 1):
        slices[ii] = int(slices[ii]) if len(slices[ii]) > 0 else None
    indices = [a.index for a in atoms][slices[0] : slices[1]]

elif ">" in atoms_to_vib:
    axis, dist = atoms_to_vib.split(">")
    ii = axis_dict[axis]
    indices = [a.index for a in atoms if a.position[ii] > float(dist)]

elif "<" in atoms_to_vib:
    axis, dist = atoms_to_vib.split("<")
    ii = axis_dict[axis]
    indices = [a.index for a in atoms if a.position[ii] < float(dist)]

atoms.set_constraint()

print(f"indices of atoms to vibrate = {indices}")

# -----------------------------------------------------------------------------
# RUN VIB
# -----------------------------------------------------------------------------

vib = Vibrations(atoms=atoms, indices=indices, delta=0.01, nfree=2)

if parsed_args.run_vib is True:

    logfile = "vib.log"
    vib.clean(empty_files=True)
    vib.run()
    open(logfile, "w").close()
    vib.summary(method="standard", log=logfile)

    print("Vibrations Calculation Finished")

# -----------------------------------------------------------------------------
# WRITE HESSIAN
# -----------------------------------------------------------------------------

if parsed_args.write_hessian is True and parsed_args.run_vib is True:

    with open("Hessian.txt", "w+") as fileobj:
        for line in vib.H:
            for num in line:
                print("{:7.2f}".format(num), end="", file=fileobj)
            print("", file=fileobj)

# -----------------------------------------------------------------------------
# HARMONIC THERMO
# -----------------------------------------------------------------------------

if parsed_args.harmonic_thermo is True and parsed_args.run_vib is True:

    temperature = parsed_args.temperature
    print(f"\nHarmonicThermo calculation at T = {temperature} K")

    vib_energies = []
    for ii, vib_energy in enumerate(vib.get_energies()):
        if np.iscomplex(vib_energy):
            print(f"Mode {ii:02d} has imaginary frequency")
        else:
            vib_energies.append(vib_energy)

    thermo = HarmonicThermo(vib_energies=vib_energies)
    thermo.get_helmholtz_energy(temperature=temperature, verbose=True)

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
