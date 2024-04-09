# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import os
import numpy as np
import copy as cp
from ase import Atom, Atoms
from ase.units import create_units
from ase.calculators.singlepoint import SinglePointDFTCalculator
from .utils import get_symbols_list

# -----------------------------------------------------------------------------
# WRITE NEB DAT
# -----------------------------------------------------------------------------

def write_neb_dat(neb_data, filename="neb.dat", mode="w+"):
    """Write the neb.dat input file for a NEB calculation."""
    neb_dict = {
        "string_method": None,
        "restart_mode": None,
        "nstep_path": None,
        "num_of_images": None,
        "opt_scheme": None,
        "CI_scheme": None,
        "first_last_opt": None,
        "minimum_image": None,
        "temp_req": None,
        "ds": None,
        "k_max": None,
        "k_min": None,
        "path_thr": None,
        "use_masses": None,
        "use_freezing": None,
        "lfcpopt": None,
        "fcp_mu": None,
        "fcp_tot_charge_first": None,
        "fcp_tot_charge_last": None,
    }

    for arg in neb_data:
        neb_dict[arg] = neb_data[arg]

    with open(filename, mode) as f:

        f.write("&PATH\n")
        for arg in [arg for arg in neb_dict if neb_dict[arg] is not None]:
            if isinstance(neb_dict[arg], str):
                neb_dict[arg] = "'" + neb_dict[arg] + "'"
            elif neb_dict[arg] is True:
                neb_dict[arg] = ".true."
            elif neb_dict[arg] is False:
                neb_dict[arg] = ".false."
            f.write("   {0} = {1}\n".format(str(arg).ljust(16), neb_dict[arg]))
        f.write("/")

# -----------------------------------------------------------------------------
# WRITE NEB INP
# -----------------------------------------------------------------------------

def write_neb_inp(neb_data, images, calc, filename="neb.pwi"):
    """Write the input file for a NEB calculation."""
    calc = cp.deepcopy(calc)
    calc.label = "tmp"

    with open(filename, "w+") as f:
        f.write("BEGIN\n")
        f.write("BEGIN_PATH_INPUT\n")

    write_neb_dat(neb_data, filename, mode="a+")
    with open(filename, "a+") as f:
        f.write("\nEND_PATH_INPUT\n")
        f.write("BEGIN_ENGINE_INPUT\n")
        for i in range(len(images)):
            calc.write_input(images[i])
            with open("tmp.pwi", "rU") as g:
                lines = g.readlines()
            for n, line in enumerate(lines):
                if "ATOMIC_POSITIONS" in line:
                    break
            if i == 0:
                for line in lines[:n]:
                    f.write(line)
                f.write("BEGIN_POSITIONS\n")
                f.write("FIRST_IMAGE\n")
            elif i == len(images) - 1:
                f.write("LAST_IMAGE\n")
            else:
                f.write("INTERMEDIATE_IMAGE\n")
            for line in lines[n:]:
                f.write(line)

        os.remove("tmp.pwi")

        f.write("END_POSITIONS\n")
        f.write("END_ENGINE_INPUT\n")
        f.write("END\n")

# -----------------------------------------------------------------------------
# READ NEB CRD
# -----------------------------------------------------------------------------

def read_neb_crd(images, filename="pwscf.crd"):
    """Read pwscf.crd file from a NEB calculation."""
    with open(filename, "rU") as fileobj:
        lines = fileobj.readlines()
    n_atoms = len(images[0])
    num = 2
    for image in images:
        positions = []
        for line in lines[num : num + n_atoms]:
            positions.append(line.split()[1:4])
        image.set_positions(positions)
        num += n_atoms + 2
    return images

# -----------------------------------------------------------------------------
# READ NEB PATH
# -----------------------------------------------------------------------------

def read_neb_path(images, filename="pwscf.path", return_index_ts=False):
    """Read pwscf.path file from a NEB calculation."""
    units = create_units("2006")
    n_atoms = len(images[0])
    with open(filename, "r") as fileobj:
        lines = fileobj.readlines()
    indices = []
    for ii, line in enumerate(lines):
        if "Image:" in line:
            indices.append(ii)
        if "QUICK-MIN FIELDS" in line:
            break
    for ii, index in enumerate(indices):
        # Read energy.
        energy = float(lines[index + 1]) * units["Hartree"]
        # Read positions.
        positions = []
        for jj in range(n_atoms):
            positions.append([float(pp) for pp in lines[index+2+jj].split()[:3]])
        positions = np.array(positions)*units["Bohr"]
        # Read forces.
        forces = []
        for jj in range(n_atoms):
            forces.append([float(ff) for ff in lines[index+2+jj].split()[3:7]])
        forces = np.array(forces) * units['Ry'] / units['Bohr']
        # Update parameters.
        images[ii].positions = positions
        images[ii].calc = SinglePointDFTCalculator(images[ii])
        images[ii].calc.results.update({"energy": energy, "forces": forces})
    if return_index_ts is True:
        return np.argsort([atoms.get_potential_energy() for atoms in images])[-1]
    else:
        return images

# -----------------------------------------------------------------------------
# PRINT AXSF
# -----------------------------------------------------------------------------

def print_axsf(filename, animation, variable_cell=False):
    """Print file for visualization of an animation in Xcrysden."""
    f = open(filename, "w+")

    print(" ANIMSTEP", len(animation), file=f)
    print(" CRYSTAL", file=f)

    if variable_cell is False:
        cell = animation[0].cell
        print(" PRIMVEC", file=f)
        print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[0]), file=f)
        print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[1]), file=f)
        print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[2]), file=f)

    for i, atoms in enumerate(animation):
        if variable_cell is True:
            cell = atoms.cell
            print(" PRIMVEC", i + 1, file=f)
            print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[0]), file=f)
            print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[1]), file=f)
            print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[2]), file=f)

        print(" PRIMCOORD", i + 1, file=f)
        print(len(atoms), len(get_symbols_list(atoms)), file=f)
        for a in atoms:
            print(
                "{0:3s} {1:14.8f} {2:14.8f} {3:14.8f}".format(
                    a.symbol, a.position[0], a.position[1], a.position[2]
                ),
                file=f,
            )
    f.close()

# -----------------------------------------------------------------------------
# READ AXSF
# -----------------------------------------------------------------------------

def read_axsf(filename):
    """Read Xcrysden animation file."""
    with open(filename, "rU") as fileobj:
        lines = fileobj.readlines()

    for line in lines:
        if "PRIMCOORD" in line:
            key = "PRIMCOORD"
            break
        elif "ATOMS" in line:
            key = "ATOMS"
            break

    if key == "PRIMCOORD":
        for n, line in enumerate(lines):
            if "PRIMVEC" in line:
                break
        cell_vectors = np.zeros((3, 3))
        for i, line in enumerate(lines[n + 1 : n + 4]):
            entries = line.split()
            cell_vectors[i][0] = float(entries[0])
            cell_vectors[i][1] = float(entries[1])
            cell_vectors[i][2] = float(entries[2])
        atoms_zero = Atoms(cell=cell_vectors, pbc=(True, True, True))
        increment = 2

    elif key == "ATOMS":
        atoms_zero = Atoms(pbc=(False, False, False))
        increment = 1

    key = "PRIMCOORD"
    animation = []
    for n, line in enumerate(lines):
        if key in line:
            atoms = Atoms(cell=cell_vectors, pbc=(True, True, True))
            for line in lines[n + increment :]:
                entr = line.split()
                if entr[0] == key:
                    break
                symbol = entr[0]
                position = (float(entr[1]), float(entr[2]), float(entr[3]))
                atoms += Atom(symbol, position=position)
            animation += [atoms]

    return animation

# -----------------------------------------------------------------------------
# REORDER ATOMS NEB
# -----------------------------------------------------------------------------

def reorder_atoms_neb(atoms_first, atoms_last):
    """Reorder atoms for a NEB calculation."""
    # Calculate the distances from each atom in atoms_first to each atom
    # in atoms_last. An infinite distance is assigned between atoms of
    # different species.
    n_atoms = len(atoms_first)
    dist_matrix = np.zeros([n_atoms, n_atoms])
    dist_argmin = np.zeros([n_atoms])
    for ii, a in enumerate(atoms_first):
        dist = np.linalg.norm(a.position - atoms_last.get_positions(), axis=1)
        dist[[not jj for jj in atoms_last.symbols == a.symbol]] = np.inf
        dist_matrix[ii, :] = dist
        dist_argmin[ii] = np.argmin(dist)

    # If each atom in atoms_first has an unique match in atoms_last,
    # we found the indices to sort atoms_last. Otherwise, we have to analyze
    # the unassigned atoms (the ones that have a shared match atom).
    uniques, count = np.unique(dist_argmin, return_counts=True)
    if len(uniques) < n_atoms:
        # We set to infinite the distances with the atoms already assigned.
        matches = [int(ii) for ii in uniques[count == 1]]
        dist_matrix[:, matches] = np.inf
        unassigned_mask = np.isin(dist_argmin, uniques[count > 1])
        unassigned_atoms = np.arange(n_atoms)[unassigned_mask]
        while len(unassigned_atoms) > 0:
            # We get the atom closer to its match between all the
            # unassigned atoms, and we assign it to its match, setting the
            # distance between the match and all the remaining unassigned
            # atoms equal to infinite.
            dist_min = []
            for uu in unassigned_atoms:
                dist_min += [np.min(dist_matrix[uu, :])]
            index_new = np.argmin(dist_min)
            assigned_new = unassigned_atoms[index_new]
            unassigned_atoms = np.delete(unassigned_atoms, index_new)
            match_new = int(dist_argmin[assigned_new])
            dist_matrix[:, match_new] = np.inf
            for uu in unassigned_atoms:
                dist_argmin[uu] = np.argmin(dist_matrix[uu, :])

    atoms_last = atoms_last[[int(ii) for ii in dist_argmin]]

    return atoms_first, atoms_last

# -----------------------------------------------------------------------------
# GET ATOMS TS FROM NEB
# -----------------------------------------------------------------------------

def get_atoms_ts_from_neb(images, index_ts=None, mult_factor=0.01):
    """Get transition state atoms from a NEB calculation."""
    if index_ts is None:
        index_ts = np.argsort([atoms.get_potential_energy() for atoms in images])[-1]
    atoms_ts = images[index_ts].copy()
    vector = np.zeros((len(atoms_ts), 3))
    if index_ts > 0:
        vector += atoms_ts.positions - images[index_ts-1].positions
    if index_ts < len(images)-1:
        vector += images[index_ts+1].positions - atoms_ts.positions
    vector /= np.linalg.norm(vector)
    vector *= mult_factor
    atoms_ts.info["vector"] = vector
    return atoms_ts

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
