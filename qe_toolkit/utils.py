# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
from ase import Atoms, units

# -------------------------------------------------------------------------------------
# GET SYMBOLS LIST
# -------------------------------------------------------------------------------------

def get_symbols_list(
    atoms: Atoms,
    check_magmoms: bool = True,
):
    """
    Get a list of unique elements in an Atoms object.
    """
    symbols = atoms.get_chemical_symbols()
    if check_magmoms is True:
        magmoms = [aa.magmom for aa in atoms]
    else:
        magmoms = [0.0 for aa in atoms]
    if len(symbols) > 1:
        for ii in range(len(symbols) - 1, 0, -1):
            for jj in range(ii):
                if symbols[jj] == symbols[ii] and magmoms[ii] == magmoms[jj]:
                    del symbols[ii]
                    break
    return symbols

# -------------------------------------------------------------------------------------
# GET SYMBOLS DICT
# -------------------------------------------------------------------------------------

def get_symbols_dict(
    atoms: Atoms,
):
    """
    Get a dictionary of unique elements in an Atoms object.
    """
    symbols_dict = {}
    symbols = atoms.get_chemical_symbols()
    for ss in symbols:
        if ss in symbols_dict:
            symbols_dict[ss] += 1
        else:
            symbols_dict[ss] = 1
    return symbols_dict

# -------------------------------------------------------------------------------------
# GET FORMULA REPETITIONS
# -------------------------------------------------------------------------------------

def get_formula_repetitions(
    atoms: Atoms
):
    """
    Get the number of repetitions in the formula of an Atoms object.
    """
    symbols_dict = get_symbols_dict(atoms)
    return min([symbols_dict[ii] for ii in symbols_dict])

# -------------------------------------------------------------------------------------
# GET ATOMS FIXED
# -------------------------------------------------------------------------------------

def get_atoms_fixed(
    atoms: Atoms,
):
    """
    Get the indices of the atoms with FixAtoms constraints.
    """
    from ase.constraints import FixAtoms
    indices = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices += list(constraint.get_indices())
    # Return indices.
    return indices

# -------------------------------------------------------------------------------------
# GET ATOMS NOT FIXED
# -------------------------------------------------------------------------------------

def get_atoms_not_fixed(
    atoms: Atoms,
):
    """
    Get the indices of the atoms without FixAtoms constraints.
    """
    return [ii for ii in range(len(atoms)) if ii not in get_atoms_fixed(atoms)]

# -------------------------------------------------------------------------------------
# GET VALENCE ELECTRONS
# -------------------------------------------------------------------------------------

def get_valence_electrons(
    atoms: Atoms,
):
    """
    Get the valence electrons of an Atoms object.
    """
    from ase.io.espresso import SSSP_VALENCE
    n_electrons = 0
    for a in atoms:
        n_electrons += SSSP_VALENCE[a.number]
    return n_electrons

# -------------------------------------------------------------------------------------
# SWAP ATOMS
# -------------------------------------------------------------------------------------

def swap_atoms(
    atoms: Atoms,
    indices_swap: list,
):
    """
    Swap two atoms in an Atoms object.
    """
    indices = list(range(len(atoms)))
    for ii in indices_swap:
        indices[ii[0]], indices[ii[1]] = indices[ii[1]], indices[ii[0]]
    # Return reordered atoms.
    return atoms[indices]

# -------------------------------------------------------------------------------------
# OPTIMAL REORDER INDICES
# -------------------------------------------------------------------------------------

def optimal_reorder_indices(
    atoms: Atoms,
    atoms_ref: Atoms,
) -> list:
    """
    Calculate indices to reorder atoms to best match reference atoms.
    """
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    numbers = np.array([aa.number for aa in atoms])
    numbers_ref = np.array([aa.number for aa in atoms_ref])
    array = np.hstack([atoms.positions, numbers.reshape(-1, 1)])
    array_ref = np.hstack([atoms_ref.positions, numbers_ref.reshape(-1, 1)])
    # Compute pairwise Euclidean distance cost matrix between rows.
    cost_matrix = cdist(XA=array_ref, XB=array, metric="euclidean")
    # Solve the optimal assignment problem (Hungarian algorithm).
    indices_ref, indices = linear_sum_assignment(cost_matrix)
    # Return indices.
    return indices

# -------------------------------------------------------------------------------------
# REORDER ATOMS
# -------------------------------------------------------------------------------------

def reorder_atoms(
    atoms: Atoms,
    atoms_ref: Atoms = None,
) -> None:
    """
    Reorder atoms to best match reference atoms.
    """
    indices = optimal_reorder_indices(atoms=atoms, atoms_ref=atoms_ref)
    # Reorder the atoms.
    atoms.positions = atoms.positions[indices]
    atoms.symbols = atoms.symbols[indices]
    # Reassign atoms to calculator to avoid new calculation.
    if atoms.calc:
        atoms.calc.atoms = atoms

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
