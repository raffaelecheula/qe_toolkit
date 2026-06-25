#!/usr/bin/env python3

# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import matplotlib
matplotlib.use("Agg")
import yaml
import numpy as np
import matplotlib.pyplot as plt
from ase import units

from qe_toolkit.io import read_cube, read_Fermi_energy

# -------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------

def main():
    # Parameters.
    filename_pwo = "pw.pwo"
    # Read Fermi energy.
    e_fermi = read_Fermi_energy(filename=filename_pwo)
    # Get potential from cube file.
    origin, a_vect, b_vect, c_vect, v_data = read_cube("velectrostatic.cube")
    # Average potential in xy plane.
    v_vect = v_data.mean(axis=(0, 1)) * units.Ha
    # Get z coordinates.
    step = np.linalg.norm(c_vect) * units.Bohr
    z_vect = np.arange(len(v_vect)) * step + step / 2
    # Plot potential.
    plt.xlim([0, max(z_vect) + step / 2])
    plt.xlabel("z [Å]")
    plt.ylabel("V [eV]")
    plt.plot(z_vect, v_vect)
    plt.tight_layout()
    plt.savefig("potential_environ.png", dpi=300)
    # Calculate work function.
    e_vacuum = float(np.mean(v_vect[:3] + v_vect[-3:]))
    workfunction = e_vacuum - e_fermi
    yaml_data = {"workfunction": workfunction}
    with open("workfunction.yaml", "w") as fileobj:
        yaml.dump(yaml_data, fileobj)

# -------------------------------------------------------------------------------------
# IF NAME MAIN
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
