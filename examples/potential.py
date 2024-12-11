#!/usr/bin/env python

# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import os
import argparse
import pickle
import matplotlib.pyplot as plt
from distutils.util import strtobool
from ase import units
from ase.calculators.espresso import Espresso
from qe_toolkit.io import read_pwi, read_pwo, write_pp_input, read_pw_bands

# -------------------------------------------------------------------------------------
# PARSE ARGUMENTS
# -------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
fbool = lambda x: bool(strtobool(x))

parser.add_argument(
    "--change_dir",
    "-c",
    type=fbool,
    required=False,
    default=True,
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
    "--run_scf",
    "-scf",
    type=fbool,
    required=False,
    default=True,
)

parsed_args = parser.parse_args()

# -------------------------------------------------------------------------------------
# CHANGE DIRECTORY
# -------------------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.makedirs("./potential", exist_ok=True)
    os.chdir("potential")
    path_head = ".."
else:
    path_head = "."

# -------------------------------------------------------------------------------------
# WRITE PP AND AVE INPUTS
# -------------------------------------------------------------------------------------

if parsed_args.write_input is True:
    # Read pwi input file.
    atoms = read_pwi(filename=parsed_args.qe_pwi, path_head=path_head)
    input_data = atoms.info["input_data"]
    pseudopotentials = atoms.info["pseudopotentials"]
    kpts = atoms.info["kpts"]
    koffset = atoms.info["koffset"]
    # Get outdir and input file for scf.
    outdir = input_data["outdir"]
    if parsed_args.run_scf is True:
        # Read pwo output file.
        atoms = read_pwo(
            filename=parsed_args.qe_pwo,
            filepwi=parsed_args.qe_pwi,
            path_head=path_head,
        )
        # Update input data.
        input_data.update({
            "restart_mode": "from_scratch",
            "calculation": "scf",
        })
        # Write input data for scf.
        calc = Espresso(
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=kpts,
            koffset=koffset,
        )
        calc.label = "scf"
        calc.set(input_data=input_data)
        calc.write_input(atoms)
    else:
        outdir = os.path.join(path_head, outdir)
    # Write pp and ave input files.
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

# -------------------------------------------------------------------------------------
# RUN PP
# -------------------------------------------------------------------------------------

if parsed_args.run_qe_bin is True:
    if parsed_args.run_scf is True:
        os.system(parsed_args.mpi_cmd + " pw.x < scf.pwi > scf.pwo")
    os.system(parsed_args.mpi_cmd + " pp.x < pp.pwi > pp.pwo")
    os.system("average.x < ave.pwi > ave.pwo")

# -------------------------------------------------------------------------------------
# POSTPROCESS
# -------------------------------------------------------------------------------------

if parsed_args.postprocess is True:
    if parsed_args.run_scf is True:
        filename = "scf.pwo"
    else:
        filename = os.path.join(path_head, parsed_args.qe_pwo)
    e_bands_dict, e_fermi = read_pw_bands(
        filename=filename,
        scale_band_energies=True,
    )
    with open("avg.dat", "r") as fileobj:
        lines = fileobj.readlines()
    x_vec = []
    y_tot = []
    y_ave = []
    for line in lines:
        line_split = line.split()
        x_vec += [float(line_split[0])]
        y_tot += [float(line_split[1]) * units.Ry]
        y_ave += [float(line_split[2]) * units.Ry]
    workfunction = max(y_tot) - e_fermi

# -------------------------------------------------------------------------------------
# PLOT POTENTIAL
# -------------------------------------------------------------------------------------

if parsed_args.postprocess is True and parsed_args.save_images is True:
    plt.xlim([min(x_vec), max(x_vec)])
    plt.xlabel("z [A]")
    plt.ylabel("V [eV]")
    plt.plot(x_vec, y_tot)
    plt.savefig("potential.png", dpi=300)

# -------------------------------------------------------------------------------------
# CHANGE DIRECTORY
# -------------------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.chdir("..")

# -------------------------------------------------------------------------------------
# SAVE PICKLE
# -------------------------------------------------------------------------------------

if parsed_args.postprocess is True and parsed_args.save_pickle is True:
    with open("workfunction.pickle", "wb") as fileobj:
        pickle.dump(workfunction, fileobj)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
