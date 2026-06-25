#!/usr/bin/env python3

# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.calculators.espresso import Espresso
from qe_toolkit.io import read_pwo, read_bands, write_projwfc_input
from qe_toolkit.pp import get_dos, get_pdos_list, plot_dos, plot_pdos_list

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    # Parameters.
    filename_pwo = "pw.pwo"
    outdir_pwo = "calc"
    energy_range = [-50.0, +50.0]
    # Quantum Espresso bin.
    qe_bindir = "/home/rcheula/espresso/qe-7.5/bin/"
    mpi_cmds = ""
    # Read atoms.
    atoms = read_pwo(filename=filename_pwo)
    band_list = read_bands(filename=filename_pwo, scale_band_energies=True)
    e_fermi = atoms.calc.eFermi
    # Make projwfc directory.
    os.makedirs("./projwfc", exist_ok=True)
    os.chdir("projwfc")
    # Prepare projwfc input.
    proj_data = {
        "outdir": f"../{outdir_pwo}",
        "ngauss": 0,
        "Emin": energy_range[0],
        "Emax": energy_range[1],
        "DeltaE": 0.01,
        "filpdos": "pdos",
    }
    write_projwfc_input(
        proj_data=proj_data,
        filename="projwfc.pwi",
    )
    # Run projwfc calculation.
    os.system(f"{mpi_cmds} {qe_bindir}projwfc.x < projwfc.pwi > projwfc.pwo")
    # Read DOS.
    energy, dos = get_dos(filename="pdos.pdos_tot", e_fermi=e_fermi)
    # Plot DOS.
    plot_dos(
        energy=energy,
        dos=dos,
        x_max=10.0 * len(atoms),
        filename="dos.png",
    )
    # Read pDOS.
    energy, pdos_list = get_pdos_list(atoms=atoms, filename="projwfc.pwo")
    # Plot pDOS.
    plot_pdos_list(
        energy=energy,
        pdos_list=pdos_list,
        x_max=10.0,
        filename="pdos.png",
    )
    # Return to previous directory.
    os.chdir("..")

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------