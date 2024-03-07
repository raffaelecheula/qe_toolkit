#!/usr/bin/env python

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import argparse
import pickle
import subprocess
import numpy as np
from distutils.util import strtobool
from mendeleev import element
from dscribe.descriptors import SOAP
from qe_toolkit.io import ReadQeInp, ReadQeOut
from qe_toolkit.pp import write_features_out

# -----------------------------------------------------------------------------
# PARSE ARGUMENTS
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--structure",
    "-s",
    type=str,
    required=True,
    default="slab",
)

parser.add_argument(
    "--save_pickle",
    "-sp",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--write_out",
    "-w",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--nodes",
    "-n",
    type=str,
    required=False,
    default="none",
)

parser.add_argument(
    "--time",
    "-t",
    type=str,
    required=False,
    default="none",
)

parser.add_argument(
    "--partition",
    "-p",
    type=str,
    required=False,
    default="none",
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

# -----------------------------------------------------------------------------
# WRITE OUT
# -----------------------------------------------------------------------------

if parsed_args.write_out is True:

    features_names = [
        "e_affinity",
        "en_pauling",
        "ion_pot",
        "d_radius",
        "d_filling",
        "d_center",
        "d_width",
        "d_skewness",
        "d_kurtosis",
        "sp_filling",
        "d_density",
        "sp_density",
        "work_func",
        "n_coord",
        "n_bonds",
        "HOMO",
        "LUMO",
    ]
    features_names += [f"SOAP_{i+1:02d}" for i in range(70)]

    if os.path.exists("features.out"):
        os.remove("features.out")

# -----------------------------------------------------------------------------
# RUN CALCULATIONS
# -----------------------------------------------------------------------------

args_str = ""
if parsed_args.nodes != "none":
    args_str += f" -n={parsed_args.nodes}"
if parsed_args.time != "none":
    args_str += f" -t={parsed_args.time}"
if parsed_args.partition != "none":
    args_str += f" -p={parsed_args.partition}"

if parsed_args.structure == "slab":
    finished = True
    if not os.path.isfile("features_bands.pickle"):
        finished = False
        sp = subprocess.Popen(["/bin/bash", "-i", "-c", f"run projwfc {args_str}"])
        sp.communicate()
    if not os.path.isfile("workfunction.pickle"):
        finished = False
        sp = subprocess.Popen(["/bin/bash", "-i", "-c", f"run potential {args_str}"])
        sp.communicate()
    if finished is False:
        exit()

elif parsed_args.structure == "bulk":
    finished = True
    if not os.path.isfile("features_bands.pickle"):
        finished = False
        sp = subprocess.Popen(["/bin/bash", "-i", "-c", f"run projwfc {args_str}"])
        sp.communicate()
    if finished is False:
        exit()

# -----------------------------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------------------------

qe_inp = ReadQeInp(parsed_args.espresso_pwi)
input_data, pseudos, kpts, koffset = qe_inp.get_data_pseudos_kpts()
atoms = qe_inp.get_atoms()
if "calculation" in input_data:
    qe_out = ReadQeOut(parsed_args.espresso_pwo)
    atoms = qe_inp.update_atoms(qe_out.get_atoms())

features_const = np.zeros((len(atoms), 4))
for i, atom in enumerate(atoms):
    elem = element(atom.symbol)
    features_const[i, 0] = elem.electron_affinity
    features_const[i, 1] = elem.en_pauling
    features_const[i, 2] = elem.ionenergies[1]
    if parsed_args.structure in ("slab", "bulk"):
        features_const[i, 3] = elem.metallic_radius * 0.01
    else:
        features_const[i, 3] = np.nan

# -----------------------------------------------------------------------------
# PROJWFC AND POTENTIAL
# -----------------------------------------------------------------------------

if parsed_args.structure == "slab":

    with open("features_bands.pickle", "rb") as fileobj:
        features_bands = pickle.load(fileobj)

    with open("workfunction.pickle", "rb") as fileobj:
        workfunction = pickle.load(fileobj)
    features_workfunction = np.zeros((len(atoms), 1))
    features_workfunction[:] = workfunction

    features_homolumo = np.zeros((len(atoms), 2))
    features_homolumo[:] = np.nan

elif parsed_args.structure == "bulk":

    with open("features_bands.pickle", "rb") as fileobj:
        features_bands = pickle.load(fileobj)

    features_workfunction = np.zeros((len(atoms), 1))
    features_workfunction[:] = np.nan

    features_homolumo = np.zeros((len(atoms), 2))
    features_homolumo[:] = np.nan

else:

    features_bands = np.zeros((len(atoms), 8))
    features_bands[:] = np.nan

    features_workfunction = np.zeros((len(atoms), 1))
    features_workfunction[:] = np.nan

    qe_out.read_bands(scale_band_energies=True)
    e_fermi = qe_out.e_fermi

    bands = []
    for kpoint in qe_out.e_bands_dict:
        bands += qe_out.e_bands_dict[kpoint]
    homo = max([b for b in bands if b <= 0])
    lumo = min([b for b in bands if b > 0])

    features_homolumo = np.zeros((len(atoms), 2))
    features_homolumo[:, 0] = homo
    features_homolumo[:, 1] = lumo

features_site = np.zeros((len(atoms), 2))
features_site[:] = np.nan

# -----------------------------------------------------------------------------
# SOAP
# -----------------------------------------------------------------------------

atoms_copy = atoms.copy()
atoms_copy.symbols = ["X" for _ in atoms_copy]

soap_desc = SOAP(
    species=["X"],
    periodic=True,
    rcut=3,
    nmax=4,
    lmax=6,
    sigma=0.35,
    sparse=False,
)

features_soap = soap_desc.create(atoms_copy)

features = np.hstack((
    features_const,
    features_bands,
    features_workfunction,
    features_site,
    features_homolumo,
    features_soap,
))

if parsed_args.save_pickle is True:
    with open("features.pickle", "wb") as fileobj:
        pickle.dump(features, fileobj)

if parsed_args.write_out is True:
    write_features_out(
        atoms=atoms,
        features_names=features_names,
        features=features,
        filename="features.out",
    )

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
