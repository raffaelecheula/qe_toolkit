#!/usr/bin/env python3

# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
from qe_toolkit.io import read_pwo, read_Bader_charges
from qe_toolkit.pp import write_pp_input

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    # Parameters.
    filename_pwo = "pw.pwo"
    outdir_pwo = "calc"
    vacuum = "off"
    # Quantum Espresso bin.
    qe_bindir = "/home/rcheula/espresso/qe-7.5/bin/"
    mpi_cmds = ""
    # Read atoms.
    atoms = read_pwo(filename=filename_pwo)
    # Make bader directory.
    os.makedirs("bader", exist_ok=True)
    os.chdir("bader")
    # Prepare PP input files for valence and all-electron densities.
    pp_data = {"prefix": "pwscf", "outdir": f"../{outdir_pwo}"}
    plot_data = {"nfile": 1, "iflag": 3, "output_format": 6}
    write_pp_input(
        pp_data={**pp_data, "plot_num": 0, "filplot": "filplot_val"},
        plot_data={**plot_data, "fileout": "pp_val.cube"},
        filename="pp_val.pwi",
    )
    write_pp_input(
        pp_data={**pp_data, "plot_num": 21, "filplot": "filplot_all"},
        plot_data={**plot_data, "fileout": "pp_all.cube"},
        filename="pp_all.pwi",
    )
    # Run PP calculations.
    os.system(f"{mpi_cmds} {qe_bindir}pp.x < pp_val.pwi > pp_val.pwo")
    os.system(f"{mpi_cmds} {qe_bindir}pp.x < pp_all.pwi > pp_all.pwo")
    # Run Bader analysis.
    os.system(f"bader pp_val.cube -ref pp_all.cube -vac {vacuum} > bader.out")
    # Read Bader charges.
    read_Bader_charges(atoms=atoms, filename="ACF.dat")
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