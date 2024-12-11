# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from distutils.util import strtobool
from ase.calculators.espresso import Espresso
from qe_toolkit.io import read_pwi, read_pwo, read_pw_bands, write_projwfc_input
from qe_toolkit.pp import get_pdos, get_pdos_vect, get_features_bands

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
    "--kpts",
    "-kp",
    type=str,
    required=False,
    default="",
)

parser.add_argument(
    "--run_nscf",
    "-nscf",
    type=fbool,
    required=False,
    default=True,
)

parsed_args = parser.parse_args()

# -------------------------------------------------------------------------------------
# CHANGE DIRECTORY
# -------------------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.makedirs("./projwfc", exist_ok=True)
    os.chdir("projwfc")
    path_head = ".."
else:
    path_head = "."

# -------------------------------------------------------------------------------------
# WRITE PROJWFC INP
# -------------------------------------------------------------------------------------

if parsed_args.write_input is True:
    # Read pwi input file.
    atoms = read_pwi(filename=parsed_args.qe_pwi, path_head=path_head)
    input_data = atoms.info["input_data"]
    pseudopotentials = atoms.info["pseudopotentials"]
    kpts = atoms.info["kpts"]
    koffset = atoms.info["koffset"]
    # Get outdir and input files for scf and nscf.
    outdir = input_data["outdir"]
    if parsed_args.run_nscf is True:
        # Update kpts.
        if parsed_args.kpts == "auto":
            kpts_new = [int(60 / atoms.cell.lengths()[ii]) for ii in range(3)]
        elif parsed_args.kpts != "":
            kpts_new = [int(ii) for ii in parsed_args.kpts.split(",")]
        else:
            kpts_new = kpts
        # Read pwo output file.
        atoms = read_pwo(
            filename=parsed_args.qe_pwo,
            filepwi=parsed_args.qe_pwi,
            path_head=path_head,
        )
        # Update input data for scf.
        input_data_scf = input_data.copy()
        input_data_scf.update({
            "restart_mode": "from_scratch",
            "calculation": "scf",
        })
        input_data_scf.pop("max_seconds", None)
        # Update input data for nscf.
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
        # Write input files for scf and nscf.
        calc = Espresso(
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=kpts,
            koffset=koffset,
        )
        calc.label = "scf"
        calc.set(input_data=input_data_scf)
        calc.write_input(atoms)
        calc.label = "nscf"
        calc.set(input_data=input_data_nscf)
        calc.set(kpts=kpts_new)
        calc.write_input(atoms)
    else:
        outdir = os.path.join(path_head, outdir)
    # Write projwfc input file.
    proj_data = {
        "outdir": outdir,
        "ngauss": 0,
        "Emin": -50.0,
        "Emax": +50.0,
        "DeltaE": 0.01,
        "filpdos": "pdos",
    }
    write_projwfc_input(proj_data=proj_data, filename="projwfc.pwi")

# -------------------------------------------------------------------------------------
# RUN PROJWFC
# -------------------------------------------------------------------------------------

if parsed_args.run_qe_bin is True:
    if parsed_args.run_nscf is True:
        os.system(parsed_args.mpi_cmd + " pw.x < scf.pwi > scf.pwo")
        os.system(parsed_args.mpi_cmd + " pw.x < nscf.pwi > nscf.pwo")
    os.system(parsed_args.mpi_cmd + " projwfc.x < projwfc.pwi > projwfc.pwo")

# -------------------------------------------------------------------------------------
# POSTPROCESS
# -------------------------------------------------------------------------------------

if parsed_args.postprocess is True:
    if parsed_args.run_nscf is True:
        filename = "nscf.pwo"
    else:
        filename = os.path.join(path_head, parsed_args.qe_pwo)
    e_bands_dict, e_fermi = read_pw_bands(
        filename=filename,
        scale_band_energies=True,
    )
    energy, dos = get_pdos(filename="pdos.pdos_tot", e_fermi=e_fermi)
    energy, pdos_vect = get_pdos_vect(
        atoms=atoms,
        e_fermi=e_fermi,
        filename="projwfc.pwo",
    )

# -------------------------------------------------------------------------------------
# SAVE IMAGES
# -------------------------------------------------------------------------------------

if parsed_args.postprocess is True and parsed_args.save_images is True:
    x_max_dos = 10.0 * len(atoms)
    x_max_pdos = 10.0
    color_dict = {
        "s": "limegreen",
        "p": "darkorange",
        "d": "royalblue",
        "f": "crimson",
    }
    fig = plt.figure(0)
    plt.xlim([0.0, x_max_dos])
    plt.ylim([-15.0, +10.0])
    plt.xlabel("pdos [a.u.]")
    plt.ylabel("energy [eV]")
    plt.plot(dos, energy)
    plt.plot([0.0, x_max_dos], [0.0] * 2, color="red")
    plt.savefig("dos.png", dpi=300)
    plt.close()
    for i, atom in enumerate(atoms):
        fig = plt.figure(i + 1)
        plt.xlim([0.0, x_max_pdos])
        plt.ylim([-15.0, +10.0])
        plt.xlabel("pdos [a.u.]")
        plt.ylabel("energy [eV]")
        pdos_dict = pdos_vect[i]
        for orbital_type in pdos_dict:
            pdos = pdos_dict[orbital_type]
            color = color_dict[orbital_type]
            plt.plot(pdos, energy, color=color)
        plt.plot([0.0, x_max_pdos], [0.0] * 2, color="red")
        plt.savefig(f"pdos_atm#{i+1}({atom.symbol}).png", dpi=300)
        plt.close()

# -------------------------------------------------------------------------------------
# CHANGE DIRECTORY
# -------------------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.chdir("..")

# -------------------------------------------------------------------------------------
# SAVE PICKLE
# -------------------------------------------------------------------------------------

if parsed_args.postprocess is True and parsed_args.save_pickle is True:
    get_features_bands(
        atoms=atoms,
        energy=energy,
        pdos_vect=pdos_vect,
        delta_e=0.1,
        save_pickle=True,
    )

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
