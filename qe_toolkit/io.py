# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import ast
import re
import pickle
import numpy as np
import copy as cp
import os
import warnings
from ase import Atom, Atoms
from ase.io import read
from ase.io.espresso import read_fortran_namelist, get_atomic_species, SSSP_VALENCE
from ase.data import atomic_numbers
from ase.units import create_units
from ase.constraints import FixAtoms, FixCartesian
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.calculators.espresso import Espresso
from .utils import get_symbols_list, get_symbols_dict


# -----------------------------------------------------------------------------
# READ QUANTUM ESPRESSO PWO
# -----------------------------------------------------------------------------

def read_pwo(
    filename="pw.pwo",
    index=None,
    filepwi="pw.pwi",
    path_head=None,
    same_path_head=False,
    **kwargs,
):
    """Read Quantum Espresso output file."""
    if path_head is not None:
        filename = os.path.join(path_head, filename)
        filepwi = os.path.join(path_head, filepwi)
    atoms_pwo = read(filename=filename, index=index, **kwargs)
    is_list = True
    if not isinstance(atoms_pwo, list):
        atoms_pwo = [atoms_pwo]
        is_list = False
    # Override cell and constraints if filepwi is available.
    if filepwi and os.path.isfile(filepwi):
        for atoms in atoms_pwo:
            atoms_pwi = read(filename=filepwi)
            data, card_lines = read_fortran_namelist(fileobj=open(filepwi))
            atoms.constraints = atoms_pwi.constraints
            if "vc" not in data["control"].get("calculation", ""):
                atoms.set_cell(atoms_pwi.get_cell())
            # This is to not require a new calculation.
            if atoms.calc:
                atoms.calc.atoms = atoms
    else:
        warnings.warn(f"file {filepwi} not found!")
    if is_list is False:
        atoms_pwo = atoms_pwo[0]
    return atoms_pwo

# -----------------------------------------------------------------------------
# READ QUANTUM ESPRESSO PWI
# -----------------------------------------------------------------------------

def read_pwi(filename="pw.pwi", path_head=None, **kwargs):
    """Read Quantum Espresso input file."""
    if path_head is not None:
        filename = os.path.join(path_head, filename)
    # Get atoms.
    atoms = read(filename=filename, **kwargs)
    # Get input_data.
    data, card_lines = read_fortran_namelist(fileobj=open(filename))
    input_data = {}
    for key in data:
        input_data.update(data[key])
    atoms.info["input_data"] = input_data
    # Get pseudopotentials.
    species_card = get_atomic_species(card_lines, n_species=data['system']['ntyp'])
    pseudopotentials = {}
    for species in species_card:
        pseudopotentials[species[0]] = species[2]
    atoms.info["pseudopotentials"] = pseudopotentials
    # Get kpts and koffset (gamma and automatic).
    kpts = None
    koffset = None
    for ii, line in enumerate(card_lines):
        if "K_POINTS" in line and "automatic" in line:
            kpts = [int(ii) for ii in card_lines[ii+1].split()[:3]]
            koffset = [int(ii) for ii in card_lines[ii+1].split()[3:]]
    atoms.info["kpts"] = kpts
    atoms.info["koffset"] = koffset
    return atoms

# -----------------------------------------------------------------------------
# READ QUANTUM ESPRESSO OUT
# -----------------------------------------------------------------------------

def read_qe_out(filename="pw.pwo"):
    """Read Quantum Espresso output."""
    units = create_units("2006")

    atoms = Atoms(pbc=True)
    cell = np.zeros((3, 3))
    energy = None
    forces = None
    spin_pol_inp = False
    spin_pol_out = False

    with open(filename, "rU") as fileobj:
        lines = fileobj.readlines()

    n_nrg = None
    n_fin = None
    n_forces = None
    n_mag_inp = []
    for nn, line in enumerate(lines):
        if "positions (alat units)" in line:
            atomic_pos_units = "alat"
            n_pos = nn + 1
        elif "ATOMIC_POSITIONS" in line and "crystal" in line:
            atomic_pos_units = "crystal"
            n_pos = nn + 1
        elif "ATOMIC_POSITIONS" in line and "angstrom" in line:
            atomic_pos_units = "angstrom"
            n_pos = nn + 1
        elif "celldm(1)" in line:
            celldm = float(line.split()[1]) * units["Bohr"]
        elif "crystal axes: (cart. coord. in units of alat)" in line:
            cell_units = "alat"
            n_cell = nn + 1
        elif "CELL_PARAMETERS" in line and "angstrom" in line:
            cell_units = "angstrom"
            n_cell = nn + 1
        elif "!" in line:
            n_nrg = nn
        elif "Final energy" in line:
            n_fin = nn
        elif "Starting magnetic structure" in line and spin_pol_out is False:
            spin_pol_out = True
            n_mag_out = nn + 2
        elif "starting_magnetization" in line:
            spin_pol_inp = True
            n_mag_inp += [nn]
        elif "ATOMIC_SPECIES" in line:
            n_atom_list = nn + 1
        elif "Forces acting on atoms" in line:
            n_forces = nn + 2

    for ii in range(3):
        line = lines[n_cell + ii]
        if cell_units == "alat":
            cell[ii] = [float(cc) * celldm for cc in line.split()[3:6]]
        elif cell_units == "angstrom":
            cell[ii] = [float(cc) for cc in line.split()[:3]]

    atoms.set_cell(cell)
    energy = None
    if n_nrg is not None:
        energy = float(lines[n_nrg].split()[4]) * units["Ry"]
    if n_fin is not None:
        energy = float(lines[n_fin].split()[3]) * units["Ry"]

    index = 0
    indices = []
    constraints = []
    translate_constraints = {0: True, 1: False}

    magmoms_dict = {}
    atoms_list = []
    if spin_pol_inp is True:
        for line in lines[n_atom_list:]:
            if len(line.split()) == 0:
                break
            atoms_list += [line.split()[0]]
        for nn in n_mag_inp:
            num = ""
            read = False
            for ii in lines[nn]:
                if ii == ")":
                    read = False
                if read is True:
                    num += ii
                if ii == "(":
                    read = True
            magmoms_dict[atoms_list[int(num) - 1]] = float(lines[nn].split()[2])

    if spin_pol_out is True:
        for line in lines[n_mag_out:]:
            if len(line.split()) == 0 or line.split()[0] == "End":
                break
            magmoms_dict[line.split()[0]] = float(line.split()[1])

    for line in lines[n_pos:]:
        if len(line.split()) == 0 or line.split()[0] == "End":
            break
        if atomic_pos_units == "alat":
            name = line.split()[1]
            positions = [[float(ii) * celldm for ii in line.split()[6:9]]]
            fix = [False, False, False]
        else:
            name = line.split()[0]
            positions = [[float(ii) for ii in line.split()[1:4]]]
            fix = [translate_constraints[int(ii)] for ii in line.split()[4:]]
        symbol = ""
        magmom_tag = ""
        for ii in range(len(name)):
            if name[ii].isdigit():
                magmom_tag += name[ii]
            else:
                symbol += name[ii]
        if spin_pol_inp is True or spin_pol_out is True:
            magmom = magmoms_dict[name]
            magmom *= SSSP_VALENCE[atomic_numbers[symbol]]
            magmoms = [magmom]
        else:
            magmoms = [0.0] * len(positions)
        if atomic_pos_units == "crystal":
            atoms += Atoms(
                symbols=symbol,
                scaled_positions=positions,
                magmoms=magmoms,
            )
        else:
            atoms += Atoms(
                symbols=symbol,
                positions=positions,
                magmoms=magmoms,
            )
        if fix == [True, True, True]:
            indices.append(index)
        elif True in fix:
            constraints.append(FixCartesian([index], fix))
        index += 1

    if n_forces is not None:
        forces = []
        for ii in range(len(atoms)):
            line = lines[n_forces + ii]
            forces.append([float(ff) for ff in line.split()[6:9]])
        forces = np.array(forces) * units['Ry'] / units['Bohr']

    constraints.append(FixAtoms(indices=indices))
    atoms.set_constraint(constraints)
    atoms.calc = SinglePointDFTCalculator(atoms)
    atoms.calc.results.update({"energy": energy})
    if forces is not None:
        atoms.calc.results.update({"forces": forces})

    return atoms

# -----------------------------------------------------------------------------
# READ QUANTUM ESPRESSO OUT
# -----------------------------------------------------------------------------

class ReadQeOut:
    """Class to read Quantum Espresso output files."""
    def __init__(self, filename):
        self.filename = filename
        self.atoms = None

    def get_atoms(self, cell=None):
        atoms = read_qe_out(self.filename)
        if cell is not None:
            atoms.set_cell(cell)
        self.atoms = atoms
        return atoms

    def get_potential_energy(self):
        if self.atoms is None:
            atoms = self.get_atoms()
        else:
            atoms = self.atoms
        return atoms.get_potential_energy()

    def read_bands(self, scale_band_energies=True):
        e_bands_dict, e_fermi = read_pw_bands(
            filename=self.filename,
            scale_band_energies=scale_band_energies,
        )
        self.e_bands_dict = e_bands_dict
        self.e_fermi = e_fermi
        return e_bands_dict, e_fermi

# -----------------------------------------------------------------------------
# READ QUANTUM ESPRESSO INP
# -----------------------------------------------------------------------------

def read_qe_inp(filename="pw.pwi"):
    """Read Quantum Espresso input."""
    with open(filename, "rU") as fileobj:
        lines = fileobj.readlines()
    n_as = 0
    n_kp = 0
    gamma = False
    for n, line in enumerate(lines):
        if "ATOMIC_SPECIES" in line:
            n_as = n
        elif "K_POINTS" in line:
            if "gamma" in line:
                gamma = True
            n_kp = n
    input_data = {}
    for n, line in enumerate(lines):
        if (
            "ATOMIC_SPECIES" in line
            or "ATOMIC_POSITIONS" in line
            or "K_POINTS" in line
            or "CELL_PARAMETERS" in line
        ):
            break
        if len(line.strip()) == 0 or line == "\n":
            pass
        elif line[0] in ("&", "/"):
            pass
        else:
            keyword, argument = line.split("=")
            keyword = re.sub(re.compile(r"\s+"), "", keyword)
            argument = re.sub(re.compile(r"\s+"), "", argument)
            if ".true." in argument:
                argument = True
            elif ".false." in argument:
                argument = False
            else:
                argument = ast.literal_eval(argument)
            if type(argument) is tuple:
                argument = argument[0]
            input_data[keyword] = argument

    pseudos = {}
    for n, line in enumerate(lines[n_as + 1 :]):
        if len(line.strip()) == 0 or line == "\n":
            break
        element, MW, pseudo = line.split()
        pseudos[element] = pseudo

    if gamma:
        kpts = (1, 1, 1)
        koffset = (0, 0, 0)
    else:
        kpts = [int(i) for i in lines[n_kp + 1].split()[:3]]
        koffset = [int(i) for i in lines[n_kp + 1].split()[3:]]

    return input_data, pseudos, kpts, koffset

# -----------------------------------------------------------------------------
# READ QUANTUM ESPRESSO INP
# -----------------------------------------------------------------------------

class ReadQeInp:
    """Class to read Quantum Espresso input files."""
    def __init__(self, filename):
        self.filename = filename
        self.atoms = None
        self.input_data = None
        self.pseudos = None
        self.kpts = None
        self.koffset = None
        self.label = filename.split(".")[0]

    def get_data_pseudos_kpts(self):
        input_data, pseudos, kpts, koffset = read_qe_inp(self.filename)
        self.input_data = input_data
        self.pseudos = pseudos
        self.kpts = kpts
        self.koffset = koffset
        return input_data, pseudos, kpts, koffset

    def get_calculator(self):
        input_data, pseudos, kpts, koffset = self.get_data_pseudos_kpts()
        calc = Espresso(
            input_data=input_data,
            pseudopotentials=pseudos,
            kpts=kpts,
            koffset=koffset,
            label=self.label,
        )
        return calc

    def get_atoms(self):
        atoms = read_qe_out(self.filename)
        self.atoms = atoms
        return atoms

    def update_atoms(self, atoms_new):
        if self.atoms is None:
            self.get_atoms()
        if self.input_data is None:
            self.get_data_pseudos_kpts()
        if "calculation" in self.input_data:
            if self.input_data["calculation"] in ("relax", "md", "vc-relax", "vc-md"):
                self.atoms.set_positions(atoms_new.get_positions())
            if self.input_data["calculation"] in ("vc-relax", "vc-md"):
                self.atoms.set_cell(atoms_new.get_cell())
        return self.atoms

    def get_input_data_dicts(self, remove_keywords=True):
        hubbard_U_dict = {}
        hubbard_J0_dict = {}
        init_charges_dict = {}
        if self.atoms is None:
            self.get_atoms()
        symbols_list = get_symbols_list(self.atoms)
        del_keywords = []
        for keyword in self.input_data:
            if "Hubbard_U" in keyword:
                n = int(keyword.split("(", ")")[1])
                symbol = symbols_list[n - 1]
                hubbard_U_dict[symbol] = self.input_data[keyword]
                del_keywords += [keyword]

            elif "Hubbard_J0" in keyword:
                n = int(keyword.split("(", ")")[1])
                symbol = symbols_list[n - 1]
                hubbard_J0_dict[symbol] = self.input_data[keyword]
                del_keywords += [keyword]

            elif "starting_charge" in keyword:
                n = int(re.split(r"\(|\)", keyword)[1])
                symbol = symbols_list[n - 1]
                init_charges_dict[symbol] = self.input_data[keyword]
                del_keywords += [keyword]
                if (
                    "tot_charge" in self.input_data and 
                    "tot_charge" not in del_keywords
                ):
                    del_keywords += ["tot_charge"]

        if remove_keywords is True:
            for keyword in del_keywords:
                del self.input_data[keyword]
        self.hubbard_U_dict = hubbard_U_dict
        self.hubbard_J0_dict = hubbard_J0_dict
        self.init_charges_dict = init_charges_dict

# -----------------------------------------------------------------------------
# UPDATE PSEUDOS
# -----------------------------------------------------------------------------

def update_pseudos(pseudos, filename):
    """Update pseudopotentials names."""
    pseudos_new = read_qe_inp(filename)[1]
    pseudos.copy()
    pseudos.update(pseudos_new)
    return pseudos

# -----------------------------------------------------------------------------
# READ PW BANDS
# -----------------------------------------------------------------------------

def read_pw_bands(filename="pw.pwo", scale_band_energies=True):
    """Read bands from Quantum Espresso output."""
    with open(filename, "rU") as fileobj:
        lines = fileobj.readlines()
    kpt = 0
    e_bands_dict = {}
    n_kpts = 0
    kpts_list = []
    n_spin = 1
    read_bands = False
    for line in lines:
        if "End of self-consistent calculation" in line:
            kpt = 0
            e_bands_dict = {}
            kpts_list = []
        if "number of k points" in line:
            n_kpts = int(line.split()[4])
        if "SPIN UP" in line:
            n_spin = 2
        if " k =" in line:
            read_bands = True
            count = 0
            kpt += 1
            e_bands_dict[kpt] = []
            kpts_list += [kpt]
        if read_bands is True:
            if count == 1:
                for i in range(8):
                    if len(line) > 9*(i+1)+2:
                        e_bands_dict[kpt] += [float(line[9*i+2:9*(i+1)+2])]
            if len(line.strip()) == 0 or line == "\n":
                count += 1
        if "the Fermi energy is" in line:
            e_fermi = float(line.split()[4])
    n_kpts *= n_spin
    if scale_band_energies is True:
        for kpt in e_bands_dict:
            for i in range(len(e_bands_dict[kpt])):
                e_bands_dict[kpt][i] -= e_fermi
    return e_bands_dict, e_fermi

# -----------------------------------------------------------------------------
# ASSIGN HUBBARD U
# -----------------------------------------------------------------------------

def assign_hubbard_U(
    atoms,
    pw_data,
    hubbard_U_dict,
    hubbard_J0_dict=None,
    not_in_dict_ok=True,
):
    """Assign Hubbard U parameters in pw parameters data."""
    symbols_list = get_symbols_list(atoms)
    for ii in range(len(symbols_list)):
        symbol = symbols_list[ii]
        if symbol not in hubbard_U_dict and not_in_dict_ok is True:
            pw_data["Hubbard_U({})".format(ii + 1)] = 0.
        else:
            pw_data["Hubbard_U({})".format(ii + 1)] = hubbard_U_dict[symbol]
        if hubbard_J0_dict is not None:
            if symbol not in hubbard_J0_dict and not_in_dict_ok is True:
                pw_data["Hubbard_J0({})".format(ii + 1)] = 0.
            else:
                pw_data["Hubbard_J0({})".format(ii + 1)] = hubbard_J0_dict[symbol]
    pw_data["lda_plus_u"] = True
    return pw_data

# -----------------------------------------------------------------------------
# ASSIGN INIT CHARGES
# -----------------------------------------------------------------------------

def assign_init_charges(atoms, pw_data, init_charges_dict):
    """Assign initial charges parameters in pw parameters data."""
    symbols_list = get_symbols_list(atoms)
    symbols_dict = get_symbols_dict(atoms)
    i = 0
    charge_dict = {}
    tot_charge = 0.0
    for symbol in symbols_list:
        charge = init_charges_dict[symbol]
        if charge != 0.0:
            charge_dict["starting_charge({})".format(i + 1)] = charge
            tot_charge += symbols_dict[symbol] * charge
        i += 1
    pw_data["tot_charge"] = tot_charge
    if "system" in pw_data:
        pw_data["system"].update(charge_dict)
    else:
        pw_data["system"] = charge_dict
    return pw_data

# -----------------------------------------------------------------------------
# WRITE QE INPUT BLOCK
# -----------------------------------------------------------------------------

def write_qe_input_block(fileobj, block_name, block_data, col=23):
    """Write Quantum Espresso input block."""
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

# -----------------------------------------------------------------------------
# WRITE DOS INPUT
# -----------------------------------------------------------------------------

def write_dos_input(dos_data, filename="dos.pwi", col=23):
    """Write input for dos calculation."""
    with open(filename, "w+") as fileobj:
        write_qe_input_block(
            fileobj=fileobj, block_name="DOS", block_data=dos_data, col=col,
        )

# -----------------------------------------------------------------------------
# WRITE PP INPUT
# -----------------------------------------------------------------------------

def write_pp_input(pp_data, plot_data, filename="pp.pwi", col=23):
    """Write input for pp calculation."""
    with open(filename, "w+") as fileobj:
        write_qe_input_block(
            fileobj=fileobj, block_name="INPUTPP", block_data=pp_data, col=col,
        )
        write_qe_input_block(
            fileobj=fileobj, block_name="PLOT", block_data=plot_data, col=col,
        )

# -----------------------------------------------------------------------------
# WRITE PROJWFC INPUT
# -----------------------------------------------------------------------------

def write_projwfc_input(proj_data, filename="projwfc.pwi", col=23):
    """Write input for projwfc calculation."""
    with open(filename, "w+") as fileobj:
        write_qe_input_block(
            fileobj=fileobj, block_name="PROJWFC", block_data=proj_data, col=col,
        )

# -----------------------------------------------------------------------------
# WRITE ENVIRON INPUT
# -----------------------------------------------------------------------------

def write_environ_input(
    env_dict, bon_dict, ele_dict, filename="environ.in", reg_list=[], col=23,
):
    """Write input for Environ calculation."""
    fileobj = open(filename, "w+")
    block_name = "ENVIRON"
    write_qe_input_block(
        fileobj=fileobj, block_name=block_name, block_data=env_dict, col=col
    )
    block_name = "BOUNDARY"
    write_qe_input_block(
        fileobj=fileobj, block_name=block_name, block_data=bon_dict, col=col
    )
    block_name = "ELECTROSTATIC"
    write_qe_input_block(
        fileobj=fileobj, block_name=block_name, block_data=ele_dict, col=col
    )
    if len(reg_list) > 0:
        print("\nDIELECTRIC_REGIONS angstrom", file=fileobj)
        for reg in reg_list:
            print("{:.4f}".format(reg.eps_stat), end=" ", file=fileobj)
            print("{:.4f}".format(reg.eps_opt), end=" ", file=fileobj)
            print("{:.14f}".format(reg.position[0]), end=" ", file=fileobj)
            print("{:.14f}".format(reg.position[1]), end=" ", file=fileobj)
            print("{:.14f}".format(reg.position[2]), end=" ", file=fileobj)
            print("{:.14f}".format(reg.width), end=" ", file=fileobj)
            print("{:.4f}".format(reg.spread), end=" ", file=fileobj)
            print(reg.dim, end=" ", file=fileobj)
            print(reg.axis, file=fileobj)
    fileobj.close()

# -----------------------------------------------------------------------------
# DIELECTRIC REGION
# -----------------------------------------------------------------------------

class DielectricRegion:
    def __init__(
        self,
        eps_stat=None,
        eps_opt=None,
        position=None,
        width=None,
        spread=None,
        dim=None,
        axis=None,
    ):
        self.eps_stat = eps_stat
        self.eps_opt = eps_opt
        self.position = position
        self.width = width
        self.spread = spread
        self.dim = dim
        self.axis = axis

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
# GET PSEUDOPOTENTIALS NAMES
# -----------------------------------------------------------------------------

def get_pseudopotentials_names(library="SSSP efficiency"):
    """Pseudopotentials names from the SSSP library."""
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

    return pseudopotentials

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
