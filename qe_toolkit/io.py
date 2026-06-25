# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import re
import os
import warnings
import numpy as np
import copy as cp
from ase import Atoms
from ase.io import read
from ase.constraints import FixAtoms, FixCartesian

from qe_toolkit.utils import get_symbols_list, get_symbols_dict

# -------------------------------------------------------------------------------------
# READ PWO
# -------------------------------------------------------------------------------------

def read_pwo(
    filename: str = "pw.pwo",
    index: int = None,
    filepwi: str = "pw.pwi",
    path_head: str = None,
    initial_magmoms: bool = True,
    **kwargs: dict,
):
    """
    Read Quantum Espresso output file.
    """
    # Get files paths.
    if path_head is None:
        path_head, filename = os.path.split(filename)
    filepwo = os.path.join(path_head, filename)
    if filepwi is not None:
        filepwi = os.path.join(path_head, filepwi)
    # Read espresso output file.
    atoms_pwo = read(filename=filepwo, index=index, format="espresso-out", **kwargs)
    is_list = True
    if not isinstance(atoms_pwo, list):
        atoms_pwo = [atoms_pwo]
        is_list = False
    # Override cell and constraints if filepwi is available.
    if filepwi is not None and os.path.isfile(filepwi):
        atoms_pwi = read_pwi(filename=filepwi)
        for atoms in atoms_pwo:
            # Copy constraints and info from input file.
            atoms.constraints = atoms_pwi.constraints.copy()
            atoms.info = atoms_pwi.info.copy()
            # Set initial magmoms as input file.
            if initial_magmoms is True:
                magmoms = atoms_pwi.get_initial_magnetic_moments()
                atoms.set_initial_magnetic_moments(magmoms=magmoms)
            # Update cell for vc-relax and vc-md calculations.
            if "vc" not in atoms.info["input_data"].get("calculation", "scf"):
                atoms.set_cell(atoms_pwi.get_cell())
            # This is to not require a new calculation.
            atoms.calc.atoms = atoms
    else:
        warnings.warn(f"file {filepwi} not found!")
        # Get constraints.
        constraints = read_pwo_constraints(filename=filepwo)
        for atoms in atoms_pwo:
            atoms.constraints = constraints.copy()
    if is_list is False:
        atoms_pwo = atoms_pwo[0]
    # Return atoms.
    return atoms_pwo

# -------------------------------------------------------------------------------------
# READ PWI
# -------------------------------------------------------------------------------------

def read_pwi(
    filename: str = "pw.pwi",
    path_head: str = None,
    apply_constraints: bool = True,
    **kwargs: dict,
):
    """
    Read Quantum Espresso input file.
    """
    from ase.io.espresso import get_atomic_species, read_fortran_namelist
    if path_head is not None:
        filename = os.path.join(path_head, filename)
    # Get atoms.
    atoms = read(filename=filename, format="espresso-in", **kwargs)
    # Get constraints.
    if apply_constraints is True:
        atoms.constraints = read_pwi_constraints(filename=filename)
    # Get input_data.
    data, card_lines = read_fortran_namelist(fileobj=open(filename))
    input_data = {}
    for key in data:
        input_data.update(data[key])
    atoms.info["input_data"] = input_data
    # Get pseudopotentials.
    species_card = get_atomic_species(card_lines, n_species=data["system"]["ntyp"])
    pseudopotentials = {}
    for species in species_card:
        pseudopotentials[species[0]] = species[2]
    atoms.info["pseudopotentials"] = pseudopotentials
    # Get kpts and koffset (gamma and automatic).
    kpts = None
    koffset = None
    for ii, line in enumerate(card_lines):
        if "K_POINTS" in line and "automatic" in line:
            kpts = [int(ii) for ii in card_lines[ii + 1].split()[:3]]
            koffset = [int(ii) for ii in card_lines[ii + 1].split()[3:]]
    atoms.info["kpts"] = kpts
    atoms.info["koffset"] = koffset
    # Return atoms.
    return atoms

# -------------------------------------------------------------------------------------
# FORCE MULTS TO CONSTRAINTS
# -------------------------------------------------------------------------------------

def force_mults_to_constraints(
    force_mults: list,
):
    """
    Convert force multipliers into ASE constrainsts.
    """
    constraints = []
    fix_atoms = []
    fix_cartesian = []
    for aa, force_mult in enumerate(force_mults):
        if force_mult is None:
            continue
        mask = [True if mm == 0.0 else False for mm in force_mult]
        if mask == [True, True, True]:
            fix_atoms += [aa]
        else:
            fix_cartesian.append(FixCartesian(a=aa, mask=mask))
    constraints.append(FixAtoms(indices=fix_atoms))
    constraints += fix_cartesian
    # Return constraints.
    return constraints

# -------------------------------------------------------------------------------------
# READ PWI CONSTRAINTS
# -------------------------------------------------------------------------------------

def read_pwi_constraints(
    filename: str = "pw.pwi",
):
    """
    Read constraints from Quantum Espresso input.
    """
    from ase.io.espresso import read_fortran_namelist, get_atomic_positions
    data, lines = read_fortran_namelist(fileobj=open(filename, "r"))
    n_atoms = data["system"]["nat"]
    positions_card = get_atomic_positions(lines=lines, n_atoms=n_atoms)
    force_mults = [position[2] for position in positions_card]
    constraints = force_mults_to_constraints(force_mults=force_mults)
    # Return constraints.
    return constraints

# -------------------------------------------------------------------------------------
# READ PWO CONSTRAINTS
# -------------------------------------------------------------------------------------

def read_pwo_constraints(
    filename: str = "pw.pwo",
):
    """
    Read constraints from Quantum Espresso output.
    """
    from ase.io.espresso import parse_pwo_start, get_atomic_positions
    with open(filename, "r") as fileobj:
        lines = fileobj.readlines()
    for number, line in reversed(list(enumerate(lines))):
        if "ATOMIC_POSITIONS" in line:
            break
    n_atoms = parse_pwo_start(lines=lines)["nat"]
    positions_card = get_atomic_positions(lines=lines[number:], n_atoms=n_atoms)
    force_mults = [position[2] for position in positions_card]
    constraints = force_mults_to_constraints(force_mults=force_mults)
    # Return constraints.
    return constraints

# -------------------------------------------------------------------------------------
# READ FERMI ENERGY
# -------------------------------------------------------------------------------------

def read_Fermi_energy(
    filename: str = "pw.pwo",
):
    """
    Read Fermi energy from Quantum Espresso output.
    """
    atoms = read(filename=filename, index=-1, format="espresso-out")
    # Return Fermi energy.
    return atoms.calc.eFermi

# -------------------------------------------------------------------------------------
# READ BANDS
# -------------------------------------------------------------------------------------

def read_bands(
    filename: str = "pw.pwo",
    scale_energies: bool = False,
):
    """
    Read bands from Quantum Espresso output.
    """
    atoms = read(filename=filename, index=-1, format="espresso-out")
    band_list = []
    for kpt in atoms.calc.kpts:
        energies = list(kpt.eps_n)
        if scale_energies is True:
            energies = [ee - atoms.calc.eFermi for ee in energies]
        band_list.append({
            "kpt": kpt.k,
            "energies": energies,
            "spin": int(kpt.s),
            "weight": float(kpt.weight),
        })
    # Return list of bands.
    return band_list

# -------------------------------------------------------------------------------------
# READ FILP
# -------------------------------------------------------------------------------------

def read_filp(
    filename: str = "vmat.dat",
):
    """
    Read filp Quantum Espresso output.
    """
    # Read file.
    with open(filename, "r") as fileobj:
        lines = fileobj.readlines()
    # Get number of bands and k-points.
    n_bands = int(re.search(r"nbnd=\s*(\d+)", lines[0]).group(1))
    n_kpts  = int(re.search(r"nks=\s*(\d+)", lines[0]).group(1))
    n_cond = int(lines[1].split()[3])
    n_val = n_bands - n_cond
    # Calculate block sizes.
    len_block_cond = n_cond // 5 + (1 if n_cond % 5 != 0 else 0)
    len_block_bands = len_block_cond * n_val + 1
    len_block_kpt = 1 + 3 * len_block_bands
    # Read data.
    data = []
    for kk in range(n_kpts):
        ii_kpt = 1 + kk * len_block_kpt
        pieces = lines[ii_kpt].split()
        kpt = [float(pp) for pp in pieces[:3]]
        bands = {}
        for jj, xyz in enumerate(["x", "y", "z"]):
            ii_band = ii_kpt + 1 + jj * len_block_bands
            values = []
            for qq in range(ii_band + 1, ii_band + len_block_bands):
                values.extend([float(xx) for xx in lines[qq].split()])
            bands[xyz] = np.array(values).reshape((n_val, n_cond))
        data.append({"kpt": kpt, "n_cond": n_cond, "n_val": n_val, "bands": bands})
    # Return results.
    return {"n_bands": n_bands, "n_kpts": n_kpts, "data": data}

# -------------------------------------------------------------------------------------
# READ BADER CHARGES
# -------------------------------------------------------------------------------------

def read_Bader_charges(
    atoms: Atoms,
    filename: str = "ACF.dat",
    filename_out: str = "charges.txt",
):
    """
    Read Bader output and calculate charges
    """
    from ase.io.espresso import SSSP_VALENCE
    charges = []
    with open(filename, "r") as fileobj:
        lines = fileobj.readlines()
        for ii, line in enumerate(lines[2:2 + len(atoms)]):
            charges.append(SSSP_VALENCE[atoms[ii].number] - float(line.split()[4]))
    # Write output file.
    with open(filename_out, "w") as fileobj:
        for ii, atom in enumerate(atoms):
            print(f"{ii:4d} {atom.symbol:4s} {charges[ii]:+7.4f}", file=fileobj)
    # Return charges.
    return charges

# -------------------------------------------------------------------------------------
# ASSIGN HUBBARD U
# -------------------------------------------------------------------------------------

def assign_hubbard_U(
    atoms: Atoms,
    input_data: dict,
    hubbard_U_dict: dict,
    hubbard_J0_dict: dict = None,
    not_in_dict_ok: bool = True,
):
    """
    Assign Hubbard U parameters in Quantum Espresso input data.
    """
    symbols_list = get_symbols_list(atoms=atoms)
    for ii in range(len(symbols_list)):
        symbol = symbols_list[ii]
        if symbol not in hubbard_U_dict and not_in_dict_ok is True:
            input_data[f"Hubbard_U({ii + 1})"] = 0.
        else:
            input_data[f"Hubbard_U({ii + 1})"] = hubbard_U_dict[symbol]
        if hubbard_J0_dict is not None:
            if symbol not in hubbard_J0_dict and not_in_dict_ok is True:
                input_data[f"Hubbard_J0({ii + 1})"] = 0.
            else:
                input_data[f"Hubbard_J0({ii + 1})"] = hubbard_J0_dict[symbol]
    input_data["lda_plus_u"] = True
    # Return updated input data.
    return input_data

# -------------------------------------------------------------------------------------
# WRITE QE INPUT BLOCK
# -------------------------------------------------------------------------------------

def write_qe_input_block(
    fileobj: object,
    block_name: str,
    block_data: dict,
    col: int = 23,
):
    """
    Write Quantum Espresso input block.
    """
    print("&" + block_name, file=fileobj)
    for arg in [arg for arg in block_data if block_data[arg] is not None]:
        if type(block_data[arg]) == str:
            string = "'" + block_data[arg] + "'"
        elif block_data[arg] is True:
            string = ".true."
        elif block_data[arg] is False:
            string = ".false."
        else:
            string = block_data[arg]
        print("   {0} = {1}".format(arg.ljust(col), string), file=fileobj)
    print("/", file=fileobj)

# -------------------------------------------------------------------------------------
# WRITE DOS INPUT
# -------------------------------------------------------------------------------------

def write_dos_input(
    dos_data: dict,
    filename: str = "dos.pwi",
    col: int = 23,
):
    """
    Write input for DOS calculation.
    """
    with open(filename, "w+") as fileobj:
        write_qe_input_block(
            fileobj=fileobj,
            block_name="DOS",
            block_data=dos_data,
            col=col,
        )

# -------------------------------------------------------------------------------------
# WRITE PP INPUT
# -------------------------------------------------------------------------------------

def write_pp_input(
    pp_data: dict,
    plot_data: dict,
    filename: str = "pp.pwi",
    col: int = 23,
):
    """
    Write input for pp calculation.
    """
    with open(filename, "w+") as fileobj:
        write_qe_input_block(
            fileobj=fileobj,
            block_name="INPUTPP",
            block_data=pp_data,
            col=col,
        )
        write_qe_input_block(
            fileobj=fileobj,
            block_name="PLOT",
            block_data=plot_data,
            col=col,
        )

# -------------------------------------------------------------------------------------
# WRITE PROJWFC INPUT
# -------------------------------------------------------------------------------------

def write_projwfc_input(
    proj_data: dict,
    filename: str = "projwfc.pwi",
    col: int = 23,
):
    """
    Write input for Projected Wavefunction calculation.
    """
    with open(filename, "w+") as fileobj:
        write_qe_input_block(
            fileobj=fileobj,
            block_name="PROJWFC",
            block_data=proj_data,
            col=col,
        )

# -------------------------------------------------------------------------------------
# WRITE AVERAGE INPUT
# -------------------------------------------------------------------------------------

def write_average_input(
    filplots: list = ["filplot"],
    weigths: list = [1.0],
    n_points: int = 1000,
    plane: int = 3,
    window: float = 1.0,
    filename: str = "ave.pwi",
):
    """
    Write input for average calculation.
    """
    n_files = len(filplots)
    with open(filename, "w+") as fileobj:
        print(n_files, file=fileobj)
        for ii in range(n_files):
            print(filplots[ii], file=fileobj)
            print(weigths[ii], file=fileobj)
        print(n_points, file=fileobj)
        print(plane, file=fileobj)
        print(window, file=fileobj)

# -------------------------------------------------------------------------------------
# WRITE ENVIRON INPUT
# -------------------------------------------------------------------------------------

def write_environ_input(
    env_dict: dict,
    bon_dict: dict,
    ele_dict: dict,
    filename: str = "environ.in",
    regions_list: list = [],
    col: int = 23,
):
    """
    Write input for Environ calculation.
    """
    with open(filename, "w+") as fileobj:
        write_qe_input_block(
            fileobj=fileobj,
            block_name="ENVIRON",
            block_data=env_dict,
            col=col,
        )
        write_qe_input_block(
            fileobj=fileobj,
            block_name="BOUNDARY",
            block_data=bon_dict,
            col=col,
        )
        write_qe_input_block(
            fileobj=fileobj,
            block_name="ELECTROSTATIC",
            block_data=ele_dict,
            col=col,
        )
        if len(regions_list) > 0:
            print("\nDIELECTRIC_REGIONS angstrom", file=fileobj)
            for reg in regions_list:
                print("{:.4f}".format(reg["eps_stat"]), end=" ", file=fileobj)
                print("{:.4f}".format(reg["eps_opt"]), end=" ", file=fileobj)
                print("{:.14f}".format(reg["position"][0]), end=" ", file=fileobj)
                print("{:.14f}".format(reg["position"][1]), end=" ", file=fileobj)
                print("{:.14f}".format(reg["position"][2]), end=" ", file=fileobj)
                print("{:.14f}".format(reg["width"]), end=" ", file=fileobj)
                print("{:.4f}".format(reg["spread"]), end=" ", file=fileobj)
                print("{}".format(reg["dim"]), end=" ", file=fileobj)
                print("{}".format(reg["axis"]), file=fileobj)

# -------------------------------------------------------------------------------------
# READ CUBE
# -------------------------------------------------------------------------------------

def read_cube(
    filename: str,
):
    """
    Read a cube file and return the origin, lattice vectors, and data.
    """
    with open(filename, "r") as fileobj:
        lines = fileobj.readlines()
    # Cube header.
    n_atoms = int(lines[2].split()[0])
    origin = np.array([float(xx) for xx in lines[2].split()[1:4]])
    # Lattice vectors and grid dimensions.
    nx, ax, ay, az = lines[3].split()
    ny, bx, by, bz = lines[4].split()
    nz, cx, cy, cz = lines[5].split()
    nx, ny, nz = int(nx), int(ny), int(nz)
    a_vect = np.array([float(ax), float(ay), float(az)])
    b_vect = np.array([float(bx), float(by), float(bz)])
    c_vect = np.array([float(cx), float(cy), float(cz)])
    # Get data.
    values = []
    for line in lines[6 + n_atoms:]:
        values.extend([float(xx) for xx in line.split()])
    data = np.array(values).reshape((nx, ny, nz))
    # Return origin, lattice vectors, and data.
    return origin, a_vect, b_vect, c_vect, data

# -------------------------------------------------------------------------------------
# CHECK FINISHED
# -------------------------------------------------------------------------------------

def check_finished(
    filename: str,
    calculation: str,
    finish_keys: dict = {},
):
    """
    Check if the calculation is finished.
    """
    finish_keys_all = {
        "scf": "End of self-consistent calculation",     
        "nscf": "End of band structure calculation",
        "bands": "End of band structure calculation",
        "relax": "Final energy",
        "md": "End of molecular dynamics calculation",
        "vc-relax": "Final enthalpy",
        "vc-md": "End of molecular dynamics calculation",
    }
    finish_keys_all.update(finish_keys)
    key = finish_keys_all[calculation]
    # Search the file.
    with open(filename, "r") as file:
        for line in reversed(file.readlines()):
            if key in line:
                return True
    return False

# -------------------------------------------------------------------------------------
# GET PSEUDOPOTENTIALS NAMES
# -------------------------------------------------------------------------------------

def get_pseudopotentials_names(
    library: str = "SSSP efficiency",
):
    """
    Pseudopotentials names from the SSSP library.
    """
    if library == "SSSP efficiency":
        pseudopotentials = {
            "Ag": "Ag_ONCV_PBE-1.0.oncvpsp.upf",
            "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
            "Ar": "Ar_ONCV_PBE-1.1.oncvpsp.upf",
            "As": "As.pbe-n-rrkjus_psl.0.2.UPF",
            "Au": "Au_ONCV_PBE-1.0.oncvpsp.upf",
            "B": "b_pbe_v1.4.uspp.F.UPF",
            "Ba": "Ba.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Be": "be_pbe_v1.4.uspp.F.UPF",
            "Bi": "Bi_pbe_v1.uspp.F.UPF",
            "Br": "br_pbe_v1.4.uspp.F.UPF",
            "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
            "Ca": "Ca_pbe_v1.uspp.F.UPF",
            "Cd": "Cd.pbe-dn-rrkjus_psl.0.3.1.UPF",
            "Ce": "Ce.GGA-PBE-paw-v1.0.UPF",
            "Cl": "cl_pbe_v1.4.uspp.F.UPF",
            "Co": "Co_pbe_v1.2.uspp.F.UPF",
            "Cr": "cr_pbe_v1.5.uspp.F.UPF",
            "Cs": "Cs_pbe_v1.uspp.F.UPF",
            "Cu": "Cu_pbe_v1.2.uspp.F.UPF",
            "Dy": "Dy.GGA-PBE-paw-v1.0.UPF",
            "Er": "Er.GGA-PBE-paw-v1.0.UPF",
            "Eu": "Eu.GGA-PBE-paw-v1.0.UPF",
            "F": "f_pbe_v1.4.uspp.F.UPF",
            "Fe": "Fe.pbe-spn-kjpaw_psl.0.2.1.UPF",
            "Ga": "Ga.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "Gd": "Gd.GGA-PBE-paw-v1.0.UPF",
            "Ge": "ge_pbe_v1.4.uspp.F.UPF",
            "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
            "He": "He_ONCV_PBE-1.0.oncvpsp.upf",
            "Hf": "Hf-sp.oncvpsp.upf",
            "Hg": "Hg_ONCV_PBE-1.0.oncvpsp.upf",
            "Ho": "Ho.GGA-PBE-paw-v1.0.UPF",
            "I": "I.pbe-n-kjpaw_psl.0.2.UPF",
            "In": "In.pbe-dn-rrkjus_psl.0.2.2.UPF",
            "Ir": "Ir_pbe_v1.2.uspp.F.UPF",
            "K": "K.pbe-spn-kjpaw_psl.1.0.0.UPF",
            "Kr": "Kr_ONCV_PBE-1.0.oncvpsp.upf",
            "La": "La.GGA-PBE-paw-v1.0.UPF",
            "Li": "li_pbe_v1.4.uspp.F.UPF",
            "Lu": "Lu.GGA-PBE-paw-v1.0.UPF",
            "Mg": "Mg.pbe-n-kjpaw_psl.0.3.0.UPF",
            "Mn": "mn_pbe_v1.5.uspp.F.UPF",
            "Mo": "Mo_ONCV_PBE-1.0.oncvpsp.upf",
            "N": "N.pbe-n-radius_5.UPF",
            "Na": "na_pbe_v1.5.uspp.F.UPF",
            "Nb": "Nb.pbe-spn-kjpaw_psl.0.3.0.UPF",
            "Nd": "Nd.GGA-PBE-paw-v1.0.UPF",
            "Ne": "Ne_ONCV_PBE-1.0.oncvpsp.upf",
            "Ni": "ni_pbe_v1.4.uspp.F.UPF",
            "O": "O.pbe-n-kjpaw_psl.0.1.UPF",
            "Os": "Os_pbe_v1.2.uspp.F.UPF",
            "P": "P.pbe-n-rrkjus_psl.1.0.0.UPF",
            "Pb": "Pb.pbe-dn-kjpaw_psl.0.2.2.UPF",
            "Pd": "Pd_ONCV_PBE-1.0.oncvpsp.upf",
            "Pm": "Pm.GGA-PBE-paw-v1.0.UPF",
            "Po": "Po.pbe-dn-rrkjus_psl.1.0.0.UPF",
            "Pr": "Pr.GGA-PBE-paw-v1.0.UPF",
            "Pt": "pt_pbe_v1.4.uspp.F.UPF",
            "Rb": "Rb_ONCV_PBE-1.0.oncvpsp.upf",
            "Re": "Re_pbe_v1.2.uspp.F.UPF",
            "Rh": "Rh_ONCV_PBE-1.0.oncvpsp.upf",
            "Rn": "Rn.pbe-dn-kjpaw_psl.1.0.0.UPF",
            "Ru": "Ru_ONCV_PBE-1.0.oncvpsp.upf",
            "S": "s_pbe_v1.4.uspp.F.UPF",
            "Sb": "sb_pbe_v1.4.uspp.F.UPF",
            "Sc": "Sc_ONCV_PBE-1.0.oncvpsp.upf",
            "Se": "Se_pbe_v1.uspp.F.UPF",
            "Si": "Si.pbe-n-rrkjus_psl.1.0.0.UPF",
            "Sm": "Sm.GGA-PBE-paw-v1.0.UPF",
            "Sn": "Sn_pbe_v1.uspp.F.UPF",
            "Sr": "Sr_pbe_v1.uspp.F.UPF",
            "Ta": "Ta_pbe_v1.uspp.F.UPF",
            "Tb": "Tb.GGA-PBE-paw-v1.0.UPF",
            "Tc": "Tc_ONCV_PBE-1.0.oncvpsp.upf",
            "Te": "Te_pbe_v1.uspp.F.UPF",
            "Ti": "ti_pbe_v1.4.uspp.F.UPF",
            "Tl": "Tl_pbe_v1.2.uspp.F.UPF",
            "Tm": "Tm.GGA-PBE-paw-v1.0.UPF",
            "V": "v_pbe_v1.4.uspp.F.UPF",
            "W": "W_pbe_v1.2.uspp.F.UPF",
            "Xe": "Xe_ONCV_PBE-1.1.oncvpsp.upf",
            "Y": "Y_pbe_v1.uspp.F.UPF",
            "Yb": "Yb.GGA-PBE-paw-v1.0.UPF",
            "Zn": "Zn_pbe_v1.uspp.F.UPF",
            "Zr": "Zr_pbe_v1.uspp.F.UPF",
        }
    else:
        raise NameError("implemented libraries: SSSP efficiency")
    # Return pseudopotentials.
    return pseudopotentials

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
