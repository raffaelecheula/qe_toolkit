# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
from ase import units
from ase import Atom, Atoms

# -----------------------------------------------------------------------------
# GET SYMBOLS LIST
# -----------------------------------------------------------------------------

def get_symbols_list(atoms, check_magmoms=True):
    """Get a list of unique elements in an Atoms object."""
    symbols = atoms.get_chemical_symbols()
    magmoms = [a.magmom for a in atoms]
    if check_magmoms is True:
        magmoms = [a.magmom for a in atoms]
    else:
        magmoms = [0.0 for a in atoms]
    if len(symbols) > 1:
        for i in range(len(symbols)-1, 0, -1):
            for j in range(i):
                if symbols[j] == symbols[i] and magmoms[i] == magmoms[j]:
                    del symbols[i]
                    break
    return symbols

# -----------------------------------------------------------------------------
# GET SYMBOLS DICT
# -----------------------------------------------------------------------------

def get_symbols_dict(atoms):
    """Get a dictionary of unique elements in an Atoms object."""
    symbols_dict = {}
    symbols = atoms.get_chemical_symbols()
    for symbol in symbols:
        if symbol in symbols_dict:
            symbols_dict[symbol] += 1
        else:
            symbols_dict[symbol] = 1
    return symbols_dict

# -----------------------------------------------------------------------------
# GET FORMULA REPETITIONS
# -----------------------------------------------------------------------------

def get_formula_repetitions(atoms):
    """Get the number of repetitions in the formula of an Atoms object."""
    symbols_dict = get_symbols_dict(atoms)
    formula_repetitions = min([symbols_dict[i] for i in symbols_dict])
    return formula_repetitions

# -----------------------------------------------------------------------------
# GET ATOMS FIXED
# -----------------------------------------------------------------------------

def get_atoms_fixed(atoms, return_mask=False):
    """Get the indices of the atoms with FixAtoms constraints."""
    from ase.constraints import FixAtoms
    indices = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices += list(constraint.get_indices())
    
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -----------------------------------------------------------------------------
# GET ATOMS NOT FIXED
# -----------------------------------------------------------------------------

def get_atoms_not_fixed(atoms, return_mask=False):
    """Get the indices of the atoms without FixAtoms constraints."""
    indices = [ii for ii in range(len(atoms)) if ii not in get_atoms_fixed(atoms)]
    if return_mask:
        return [True if ii in indices else False for ii in range(len(atoms))]
    else:
        return indices

# -----------------------------------------------------------------------------
# GET VALENCE ELECTRONS
# -----------------------------------------------------------------------------

def get_valence_electrons(atoms):
    """Get the valence electrons of an Atoms object."""
    from ase.io.espresso import SSSP_VALENCE
    n_electrons = 0
    for a in atoms:
        n_electrons += SSSP_VALENCE[a.number]
    return n_electrons

# -----------------------------------------------------------------------------
# READ VIB ENERGIES
# -----------------------------------------------------------------------------

def read_vib_energies(filename="vib.log", imaginary=False):
    """Read the vibrational energies from a log file."""
    vib_energies = []
    fileobj = open(filename, "rU")
    lines = fileobj.readlines()
    fileobj.close()
    for i in range(3, len(lines)):
        if lines[i][0] == "-":
            break
        string = lines[i].split()[1]
        if string[-1] == "i":
            if imaginary is True:
                vib_energies.append(complex(0.0, float(string[:-1]) * 1e-3))
        else:
            vib_energies.append(complex(float(string) * 1e-3))
    return vib_energies

# -----------------------------------------------------------------------------
# GET MOMENT OF INERTIA XYZ
# -----------------------------------------------------------------------------

def get_moments_of_inertia_xyz(atoms, center=None):
    """Get the moments of inertia of a molecule."""
    if center is None:
        center = atoms.get_center_of_mass()
    positions = atoms.get_positions() - center
    masses = atoms.get_masses()
    inertia_moments = np.zeros(3)
    for ii in range(len(atoms)):
        xx, yy, zz = positions[ii]
        inertia_moments[0] += masses[ii] * (yy ** 2 + zz ** 2)
        inertia_moments[1] += masses[ii] * (xx ** 2 + zz ** 2)
        inertia_moments[2] += masses[ii] * (xx ** 2 + yy ** 2)
    return inertia_moments

# -----------------------------------------------------------------------------
# SWAP ATOMS
# -----------------------------------------------------------------------------

def swap_atoms(atoms, indices_swap):
    """Swap two atoms in an Atoms object."""
    indices = list(range(len(atoms)))
    for ind in indices_swap:
        indices[ind[0]], indices[ind[1]] = indices[ind[1]], indices[ind[0]]
    atoms = atoms[indices]
    return atoms

# -----------------------------------------------------------------------------
# GET VIB MODES ANIMATION
# -----------------------------------------------------------------------------

def get_vib_modes_animation(vib, kT=units.kB * 300, nimages=30):
    """Get an animation of the vibrational modes."""
    animations = []
    for index, energy in enumerate(vib.get_energies()):
        if abs(energy) > 1e-5:
            animation = []
            mode = vib.get_mode(index) * np.sqrt(kT / abs(vib.hnu[index]))
            p = vib.atoms.positions.copy()
            index %= 3 * len(vib.indices)
            for x in np.linspace(0, 2 * np.pi, nimages, endpoint=False):
                vib.atoms.set_positions(p + np.sin(x) * mode)
                animation += [vib.atoms.copy()]
            vib.atoms.set_positions(p)
            animations.append(animation)
    return animations

# -----------------------------------------------------------------------------
# WRITE ATOMS PICKLE
# -----------------------------------------------------------------------------

def write_atoms_pickle(atoms, filename):
    """Write atoms data into pickle file."""
    import pickle
    with open(filename, "wb") as fileobj:
        pickle.dump(atoms, fileobj)

# -----------------------------------------------------------------------------
# READ ATOMS PICKLE
# -----------------------------------------------------------------------------

def read_atoms_pickle(filename):
    """Read atoms data from pickle file."""
    import pickle
    with open(filename, "rb") as fileobj:
        atoms = pickle.load(fileobj)
    return atoms

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
