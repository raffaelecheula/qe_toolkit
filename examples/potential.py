#!/usr/bin/env python

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import os
import argparse
import pickle
import matplotlib.pyplot as plt
from distutils.util import strtobool
from ase.units import Ry
from ase.calculators.espresso import Espresso
from qe_toolkit.io import ReadQeInp, ReadQeOut, write_pp_input

# -----------------------------------------------------------------------------
# PARSE ARGUMENTS
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser()

fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--change_dir",
    "-c",
    type=fbool, required=False, default=True,
)

parser.add_argument(
    "--write_input",
    "-w",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--run_qe_bin",
    "-r",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--postprocess",
    "-p",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--save_images",
    "-si",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--save_pickle",
    "-sp",
    type=fbool,
    required=False,
    default=True,
)

parser.add_argument(
    "--label",
    "-l",
    type=str,
    required=False,
    default="pw",
)

parser.add_argument(
    "--qe_path",
    "-qe",
    type=str,
    required=False,
    default="",
)

parser.add_argument(
    "--run_scf",
    "-scf",
    type=fbool,
    required=False,
    default=True,
)

parsed_args = parser.parse_args()

# -----------------------------------------------------------------------------
# CHANGE DIRECTORY
# -----------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.makedirs("./potential", exist_ok=True)
    os.chdir("potential")
    pwout_dir = ".."
else:
    pwout_dir = "."

# -----------------------------------------------------------------------------
# PRINT PP AND AVE INPUTS
# -----------------------------------------------------------------------------

label = parsed_args.label

if parsed_args.write_input is True:

    qe_inp = ReadQeInp(f"{pwout_dir}/{label}.pwi")
    atoms = qe_inp.get_atoms()
    input_data, pseudos, kpts, koffset = qe_inp.get_data_pseudos_kpts()
    outdir = input_data["outdir"]

    if parsed_args.run_scf is True:
        
        qe_out = ReadQeOut(f"{pwout_dir}/{label}.pwo")
        atoms = qe_inp.update_atoms(qe_out.get_atoms())
        
        input_data.update({
            "restart_mode": "from_scratch",
            "calculation": "scf",
        })
        calc = Espresso(
            input_data=input_data,
            pseudopotentials=pseudos,
            kpts=kpts,
            koffset=koffset,
        )
        calc.label = "scf"
        calc.set(input_data=input_data)
        calc.write_input(atoms)

    else:
        outdir = f"{pwout_dir}/{outdir}"

    pp_data = {
        "outdir": outdir,
        "filplot": "filplot",
        "plot_num": 11,
    }
    plot_data = {}

    write_pp_input(pp_data, plot_data, filename="pp.pwi")

    n_files = 1
    filplots = [pp_data["filplot"]]
    weigths = [1.0]
    n_points = 1000
    plane = 3
    window = 1.0

    with open("ave.pwi", "w+") as fileobj:
        print(n_files, file=fileobj)
        for i in range(n_files):
            print(filplots[i], file=fileobj)
            print(weigths[i], file=fileobj)
        print(n_points, file=fileobj)
        print(plane, file=fileobj)
        print(window, file=fileobj)

# -----------------------------------------------------------------------------
# RUN PP
# -----------------------------------------------------------------------------

if parsed_args.run_qe_bin is True:
    qe_path = parsed_args.qe_path
    if parsed_args.run_scf is True:
        os.system(f"srun {qe_path}pw.x < scf.pwi > scf.pwo")
    os.system(f"srun {qe_path}pp.x < pp.pwi > pp.pwo")
    os.system(f"{qe_path}average.x < ave.pwi > ave.pwo")

# -----------------------------------------------------------------------------
# PLOT POTENTIAL
# -----------------------------------------------------------------------------

if parsed_args.postprocess is True:

    if parsed_args.run_scf is True:
        filename = "scf.pwo"
    else:
        filename = f"{pwout_dir}/{label}.pwo"
    qe_out = ReadQeOut(filename=filename)
    qe_out.read_bands(scale_band_energies=True)
    e_fermi = qe_out.e_fermi

    with open("avg.dat", "r") as fileobj:
        lines = fileobj.readlines()

    x_vec = []
    y_tot = []
    y_ave = []

    for line in lines:
        line_split = line.split()
        x_vec += [float(line_split[0])]
        y_tot += [float(line_split[1]) * Ry]
        y_ave += [float(line_split[2]) * Ry]

    workfunction = max(y_tot) - e_fermi

# -----------------------------------------------------------------------------
# PLOT POTENTIAL
# -----------------------------------------------------------------------------

if parsed_args.postprocess is True and parsed_args.save_images is True:

    plt.xlim([min(x_vec), max(x_vec)])
    plt.xlabel("z [A]")
    plt.ylabel("V [eV]")
    plt.plot(x_vec, y_tot)
    plt.savefig("potential.png", dpi=300)

# -----------------------------------------------------------------------------
# CHANGE DIRECTORY
# -----------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.chdir("..")

# -----------------------------------------------------------------------------
# SAVE PICKLE
# -----------------------------------------------------------------------------

if parsed_args.postprocess is True and parsed_args.save_pickle is True:

    with open("workfunction.pickle", "wb") as fileobj:
        pickle.dump(workfunction, fileobj)

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
