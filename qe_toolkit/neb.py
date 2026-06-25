# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import numpy as np
import copy as cp
from ase import Atoms
from ase.units import create_units
from ase.io.espresso import write_espresso_in
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator

from qe_toolkit.utils import get_symbols_list

# -------------------------------------------------------------------------------------
# WRITE NEB DAT
# -------------------------------------------------------------------------------------

def write_neb_dat(
    fileobj: object,
    neb_data: dict,
):
    """
    Write a neb.dat input file for a NEB calculation.
    """
    neb_data = {
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
        **neb_data,
    }
    # Write to file.
    fileobj.write("&PATH\n")
    for arg in [arg for arg in neb_data if neb_data[arg] is not None]:
        if isinstance(neb_data[arg], str):
            neb_data[arg] = "'" + neb_data[arg] + "'"
        elif neb_data[arg] is True:
            neb_data[arg] = ".true."
        elif neb_data[arg] is False:
            neb_data[arg] = ".false."
        fileobj.write("   {0} = {1}\n".format(str(arg).ljust(16), neb_data[arg]))
    fileobj.write("/")

# -------------------------------------------------------------------------------------
# WRITE NEB INP
# -------------------------------------------------------------------------------------

def write_neb_inp(
    images: list,
    neb_data: dict,
    input_data: dict,
    pseudopotentials: list,
    kpts: list,
    koffset: list = None,
    filename: str = "neb.pwi",
    **kwargs: dict,
):
    """
    Write the input file for a NEB calculation.
    """
    import io
    buffer = io.StringIO()
    # Write path input.
    write_neb_dat(fileobj=buffer, neb_data=neb_data)
    text = "BEGIN\nBEGIN_PATH_INPUT\n" + buffer.getvalue() + "\nEND_PATH_INPUT\n"
    # Write engine input.
    for ii, atoms in enumerate(images):
        buffer.seek(0)
        buffer.truncate(0)
        write_espresso_in(
            buffer,
            atoms=atoms,
            input_data=input_data,
            pseudopotentials=pseudopotentials,
            kpts=kpts,
            koffset=koffset,
            **kwargs,
        )
        if ii == 0:
            text += "BEGIN_ENGINE_INPUT\n"
            text += buffer.getvalue().split("ATOMIC_POSITIONS", 1)[0]
            text += "BEGIN_POSITIONS\nFIRST_IMAGE\nATOMIC_POSITIONS"
        elif ii == len(images) - 1:
            text += "LAST_IMAGE\nATOMIC_POSITIONS"
        else:
            text += "INTERMEDIATE_IMAGE\nATOMIC_POSITIONS"
        text += buffer.getvalue().split("ATOMIC_POSITIONS", -1)[-1]
    text += "END_POSITIONS\nEND_ENGINE_INPUT\nEND\n"
    # Write to file.
    with open(filename, "w") as fileobj:
        fileobj.write(text)

# -------------------------------------------------------------------------------------
# READ NEB CRD
# -------------------------------------------------------------------------------------

def read_neb_crd(
    images: list,
    filename: str = "pwscf.crd",
):
    """
    Read .crd file from a NEB calculation.
    """
    with open(filename, "r") as fileobj:
        lines = fileobj.readlines()
    n_atoms = len(images[0])
    num = 2
    for image in images:
        positions = []
        for line in lines[num : num + n_atoms]:
            positions.append(line.split()[1:4])
        image.set_positions(positions)
        num += n_atoms + 2
    # Return images.
    return images

# -------------------------------------------------------------------------------------
# READ NEB PATH
# -------------------------------------------------------------------------------------

def read_neb_path(
    images: list,
    filename: str = "pwscf.path",
):
    """
    Read .path file from a NEB calculation.
    """
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
            positions.append([float(pp) for pp in lines[index + 2 + jj].split()[:3]])
        positions = np.array(positions) * units["Bohr"]
        # Read forces.
        forces = []
        for jj in range(n_atoms):
            forces.append([float(ff) for ff in lines[index + 2 + jj].split()[3:7]])
        forces = np.array(forces) * units["Ry"] / units["Bohr"]
        # Update parameters.
        images[ii].positions = positions
        images[ii].calc = SinglePointCalculator(atoms=images[ii])
        images[ii].calc.results.update({"energy": energy, "forces": forces})
    # Return images.
    return images

# -------------------------------------------------------------------------------------
# READ AXSF
# -------------------------------------------------------------------------------------

def read_axsf(
    filename: str = "pwscf.axsf",
):
    """
    Read .axsf Xcrysden file.
    """
    # Read file.
    with open(filename, "r") as fileobj:
        lines = fileobj.readlines()
    # Recognize format.
    for line in lines:
        if "PRIMCOORD" in line:
            key = "PRIMCOORD"
            break
        elif "ATOMS" in line:
            key = "ATOMS"
            break
    # Parse cell.
    if key == "PRIMCOORD":
        for n, line in enumerate(lines):
            if "PRIMVEC" in line:
                break
        cell_vectors = np.zeros((3, 3))
        for ii, line in enumerate(lines[n + 1 : n + 4]):
            entries = line.split()
            cell_vectors[ii][0] = float(entries[0])
            cell_vectors[ii][1] = float(entries[1])
            cell_vectors[ii][2] = float(entries[2])
        atoms_zero = Atoms(cell=cell_vectors, pbc=True)
        increment = 2
    elif key == "ATOMS":
        atoms_zero = Atoms(pbc=False)
        increment = 1
    # Parse atoms.
    key = "PRIMCOORD"
    images = []
    for n, line in enumerate(lines):
        if key in line:
            atoms = Atoms(cell=cell_vectors, pbc=True)
            for line in lines[n + increment :]:
                entry = line.split()
                if entry[0] == key:
                    break
                symbol = entry[0]
                positions = [float(entry[1]), float(entry[2]), float(entry[3])]
                atoms += Atoms(symbols=[symbol], positions=positions)
            images += [atoms]
    # Return images.
    return images

# -------------------------------------------------------------------------------------
# WRITE AXSF
# -------------------------------------------------------------------------------------

def write_axsf(
    filename: str = "images.axsf",
    images: list = None,
    variable_cell=False,
):
    """
    Write file for visualization of an animation in Xcrysden.
    """
    with open(filename, "w+") as fileobj:
        print(" ANIMSTEP", len(images), file=fileobj)
        print(" CRYSTAL", file=fileobj)
        if variable_cell is False:
            cell = images[0].cell
            print(" PRIMVEC", file=fileobj)
            print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[0]), file=fileobj)
            print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[1]), file=fileobj)
            print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[2]), file=fileobj)
        for ii, atoms in enumerate(images):
            if variable_cell is True:
                cell = atoms.cell
                print(" PRIMVEC", ii + 1, file=fileobj)
                print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[0]), file=fileobj)
                print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[1]), file=fileobj)
                print("{0:14.8f} {1:14.8f} {2:14.8f}".format(*cell[2]), file=fileobj)
            print(" PRIMCOORD", ii + 1, file=fileobj)
            print(len(atoms), len(get_symbols_list(atoms)), file=fileobj)
            for aa in atoms:
                vv = [aa.symbol] + list(aa.position)
                print("{0:3s} {1:14.8f} {2:14.8f} {3:14.8f}".format(*vv), file=fileobj)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
