# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from distutils.util import strtobool
from ase.calculators.espresso import Espresso
from qe_toolkit.io import ReadQeOut, ReadQeInp, write_projwfc_input
from qe_toolkit.pp import get_pdos, get_pdos_vect, get_features_bands

# -----------------------------------------------------------------------------
# PARSE ARGUMENTS
# -----------------------------------------------------------------------------

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
    "--label",
    "-l",
    type=str,
    required=False,
    default="pw",
)

parser.add_argument(
    "--kpts",
    "-kp",
    type=str,
    required=False,
    default="",
)

parser.add_argument(
    "--qe_path",
    "-qe",
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

# -----------------------------------------------------------------------------
# CHANGE DIRECTORY
# -----------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.makedirs("./projwfc", exist_ok=True)
    os.chdir("projwfc")
    pwout_dir = ".."
else:
    pwout_dir = "."

# -----------------------------------------------------------------------------
# PRINT PROJWFC INP
# -----------------------------------------------------------------------------

label = parsed_args.label

if parsed_args.write_input is True:

    qe_inp = ReadQeInp(f"{pwout_dir}/{label}.pwi")
    atoms = qe_inp.get_atoms()
    input_data, pseudos, kpts, koffset = qe_inp.get_data_pseudos_kpts()
    outdir = input_data["outdir"]

    if parsed_args.kpts == "auto":
        kpts_new = [int(60 / atoms.cell.lengths()[ii]) for ii in range(3)]
    elif parsed_args.kpts != "":
        kpts_new = [int(i) for i in parsed_args.kpts.split(",")]
    else:
        kpts_new = kpts

    if parsed_args.run_nscf is True:

        qe_out = ReadQeOut(f"{pwout_dir}/{label}.pwo")
        atoms = qe_inp.update_atoms(qe_out.get_atoms())

        input_data_scf = input_data.copy()
        input_data_scf.update(
            {"restart_mode": "from_scratch", "calculation": "scf",}
        )
        input_data_scf.pop("max_seconds", None)

        input_data_nscf = input_data.copy()
        input_data_nscf.update(
            {
                "restart_mode": "from_scratch",
                "calculation": "nscf",
                "occupations": "tetrahedra_opt",
                "conv_thr": 1e-8,
                "nosym": True,
                "verbosity": "high",
                "diago_full_acc": True,
            }
        )
        input_data_nscf.pop("smearing", None)
        input_data_nscf.pop("degauss", None)
        input_data_nscf.pop("max_seconds", None)

        calc = Espresso(
            input_data=input_data,
            pseudopotentials=pseudos,
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
        outdir = f"{pwout_dir}/{outdir}"

    proj_data = {
        "outdir": outdir,
        "ngauss": 0,
        "Emin": -50.0,
        "Emax": +50.0,
        "DeltaE": 0.01,
        "filpdos": "pdos",
    }

    write_projwfc_input(proj_data=proj_data, filename="projwfc.pwi")

# -----------------------------------------------------------------------------
# RUN PROJWFC
# -----------------------------------------------------------------------------

if parsed_args.run_qe_bin is True:
    qe_path = parsed_args.qe_path
    if parsed_args.run_nscf is True:
        os.system(f"srun {qe_path}pw.x < scf.pwi > scf.pwo")
        os.system(f"srun {qe_path}pw.x < nscf.pwi > nscf.pwo")
    os.system(f"srun {qe_path}projwfc.x < projwfc.pwi > projwfc.pwo")

# -----------------------------------------------------------------------------
# POSTPROCESS
# -----------------------------------------------------------------------------

if parsed_args.postprocess is True:

    if parsed_args.run_nscf is True:
        filename = "nscf.pwo"
    else:
        filename = f"{pwout_dir}/{label}.pwo"
    qe_out = ReadQeOut(filename=filename)
    atoms = qe_out.get_atoms()
    qe_out.read_bands(scale_band_energies=True)
    e_fermi = qe_out.e_fermi

    energy, dos = get_pdos(filename="pdos.pdos_tot", e_fermi=e_fermi,)

    energy, pdos_vect = get_pdos_vect(
        atoms=atoms, e_fermi=e_fermi, filename="projwfc.pwo",
    )

# -----------------------------------------------------------------------------
# SAVE IMAGES
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# CHANGE DIRECTORY
# -----------------------------------------------------------------------------

if parsed_args.change_dir is True:
    os.chdir("..")

# -----------------------------------------------------------------------------
# SAVE PICKLE
# -----------------------------------------------------------------------------

if parsed_args.postprocess is True and parsed_args.save_pickle is True:

    get_features_bands(
        atoms=atoms,
        energy=energy,
        pdos_vect=pdos_vect,
        delta_e=0.1,
        save_pickle=True,
    )

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
