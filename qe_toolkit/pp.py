# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import numpy as np
import copy as cp
from ase import Atoms

from qe_toolkit.io import read_bands, write_pp_input
from qe_toolkit.utils import get_symbols_list, get_symbols_dict

# -------------------------------------------------------------------------------------
# MERGE CHARGE FILES
# -------------------------------------------------------------------------------------

def merge_charge_files(
    files_in: list,
    file_out: str,
):
    """
    Merge two charge files.
    """
    # Read files and sum densities.
    for num, filename in enumerate(files_in):
        with open(filename, "r") as fileobj:
            lines = fileobj.readlines()
        if num == 0:
            new_lines = cp.deepcopy(lines)
            for nn, line in enumerate(lines):
                if "BEGIN_BLOCK_DATAGRID_3D" in line:
                    n_den = nn
                if "END_DATAGRID_3D" in line:
                    n_end = nn
            n_grid = [int(nn) for nn in lines[n_den + 3].split()]
            n_points = n_grid[0] * n_grid[1] * n_grid[2]
            density = np.zeros(n_points)
        ii = 0
        for line in lines[n_den + 8 : n_end]:
            for ll in line.split():
                density[ii] += float(ll)
                ii += 1
    # Write output file.
    with open(file_out, "w+") as fileobj:
        for line in new_lines[: n_den + 8]:
            fileobj.write(line)
        for ii in range(n_points):
            if ii != 0 and ii % 6 == 0:
                fileobj.write("\n")
            fileobj.write("{:14.6E}".format(density[ii]))
        fileobj.write("\n")
        for line in new_lines[n_end:]:
            fileobj.write(line)

# -------------------------------------------------------------------------------------
# CLASS STATE
# -------------------------------------------------------------------------------------

class State:
    def __init__(self, state_num, atom_num, element, shell_num, l, m):
        self.state_num = state_num
        self.element = element
        self.atom_num = atom_num
        self.shell_num = shell_num
        self.l = l
        self.m = m
        p_dict = {1: "p z", 2: "p x", 3: "p y"}
        d_dict = {1: "d z^2", 2: "d zx", 3: "d zy", 4: "d x^2-y^2", 5: "d xy"}
        if l == 0:
            self.orbital = "s"
        elif l == 1:
            self.orbital = p_dict[self.m]
        elif l == 2:
            self.orbital = d_dict[self.m]

# -------------------------------------------------------------------------------------
# CLASS BAND
# -------------------------------------------------------------------------------------

class Band:
    def __init__(self, band_num, energy, state_nums, weights):
        self.band_num = band_num
        self.energy = energy
        self.state_nums = state_nums
        self.weights = weights

# -------------------------------------------------------------------------------------
# CLASS ATOM PP
# -------------------------------------------------------------------------------------

class AtomPP:
    def __init__(
        self,
        atom_num,
        element,
        states=[],
        bands=[],
        weights=[],
        color=None,
    ):
        self.atom_num = atom_num
        self.element = element
        self.states = states
        self.bands = bands
        self.weights = weights
        self.color = color

# -------------------------------------------------------------------------------------
# READ PROJWFC
# -------------------------------------------------------------------------------------

def read_projwfc(
    filename: str,
    kpoint: int,
    print_summary: bool = False,
):
    """
    Read Quantum Espreso projwfc output.
    """
    with open(filename, "r") as fileobj:
        lines = fileobj.readlines()
    states_list = []
    bands_list = []
    read = False
    kpt = 0
    band_num = 0
    for line in lines:
        if "state #" in line:
            state_num = int(line[12 : 16])
            atom_num = int(line[22 : 26])
            element = line[28 : 30].strip()
            shell_num = int(line[38 : 40])
            l = int(line[44 : 45])
            m = int(line[48 : 50])
            state = State(
                state_num=state_num,
                atom_num=atom_num,
                element=element,
                shell_num=shell_num,
                l=l,
                m=m,
            )
            states_list += [state]
        if " k = " in line:
            kpt += 1
        if kpoint == kpt:
            if "    |psi|^2" in line:
                read = False
                band = Band(
                    band_num=band_num,
                    energy=energy,
                    state_nums=state_nums,
                    weights=weights,
                )
                bands_list += [band]
            if read is True:
                for i in range(5):
                    weight = line[11 + 14 * i : 16 + 14 * i]
                    state_num = line[19 + 14 * i : 23 + 14 * i]
                    try:
                        weights += [float(weight)]
                        state_nums += [int(state_num)]
                    except:
                        pass
            if "==== e(" in line:
                read = True
                band_num = int(line[7 : 11])
                energy = float(line[14 : 26])
                weights = []
                state_nums = []
            if "     e =" in line:
                read = True
                band_num += 1
                energy = float(line[8 : 20])
                weights = []
                state_nums = []
    if print_summary is True:
        print(f"n states = {len(states_list)} - n bands = {len(bands_list)}")
    return states_list, bands_list

# -------------------------------------------------------------------------------------
# GET ATOMS DETAILS
# -------------------------------------------------------------------------------------

def get_atoms_details(
    states_list: list,
    bands_list: list,
    atom_num_list: list,
    delta_e: float = 0.0,
    color_dict: dict = None,
):
    """
    Get the details of atoms from post-processing calculations.
    """
    atoms_pp_list = []
    for atom_num in atom_num_list:
        states = [ss for ss in states_list if ss.atom_num == atom_num]
        element = states[0].element
        color = color_dict[element] if color_dict is not None else None
        atom = AtomPP(
            atom_num=atom_num,
            element=element,
            states=[],
            bands=[],
            weights=[],
            color=color,
        )
        atoms_pp_list += [atom]
        for band in bands_list:
            for state in states:
                if state.state_num in band.state_nums:
                    index = band.state_nums.index(state.state_num)
                    weight = band.weights[index]
                    if weight > delta_e:
                        atom.states += [state]
                        atom.bands += [band]
                        atom.weights += [weight]
    return atoms_pp_list

# -------------------------------------------------------------------------------------
# PRINT ATOMS DETAILS
# -------------------------------------------------------------------------------------

def print_atoms_details(
    atoms_pp_list: list,
    filename: str = "atom_details.out",
):
    """
    Print the details of atoms from post-processing calculations.
    """
    fileobj = open(filename, "w+")
    count_bands = {}
    for atom in atoms_pp_list:
        atom_num = atom.atom_num
        print("\n atom {0} {1}".format(atom.element, str(atom_num)), file=fileobj)
        print(
            "| state num | orbital type | band num | weight |  energy  |", file=fileobj
        )
        for i in range(len(atom.states)):
            state = atom.states[i]
            band = atom.bands[i]
            weight = atom.weights[i]
            print(
                " {0:10d}   {1:9s} {2:13d} {3:8.4f}   {4:+8.3f}".format(
                    state.state_num, state.orbital, band.band_num, weight, band.energy
                ),
                file=fileobj,
            )
            try:
                if state.atom_num not in count_bands[band.band_num]:
                    count_bands[band.band_num] += [state.atom_num]
            except:
                count_bands[band.band_num] = [state.atom_num]
    print("\n SHARED BANDS \n", file=fileobj)
    for num in sorted([n for n in count_bands if len(count_bands[n]) > 1]):
        all_bands = sum([a.bands for a in atoms_pp_list], [])
        band = [b for b in all_bands if b.band_num == num][0]
        string = "band {0:4d} ({1:+8.3f}) shared by atoms: ".format(num, band.energy)
        for atom_num in sorted(count_bands[num]):
            atom = [a for a in atoms_pp_list if a.atom_num == atom_num][0]
            string += " {0:2s}{1:3d}  - ".format(atom.element, atom_num)
        print(string[:-2], file=fileobj)
    print("", file=fileobj)
    fileobj.close()

# -------------------------------------------------------------------------------------
# PLOT ENERGY LEVELS
# -------------------------------------------------------------------------------------

def plot_band_levels(
    atoms_pp_list: list,
    num_min_print: int,
    bands_energies: list,
    e_min: float,
    e_max: float
):
    """
    Plot the band levels.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure()
    if bands_energies is not None:
        for energy in bands_energies:
            plt.plot([0.0, 1.0], [energy] * 2, color="whitesmoke")
    weight_cum = {}
    for atom in atoms_pp_list:
        color = atom.color
        for i in range(len(atom.bands)):
            band = atom.bands[i]
            weight = atom.weights[i]
            try:
                w_min = weight_cum[band]
                weight_cum[band] += weight
            except:
                w_min = 0.0
                weight_cum[band] = weight
            w_max = weight_cum[band]
            plt.plot([w_min, w_max], [band.energy] * 2, color=color)
    for band in weight_cum:
        if weight_cum[band] > num_min_print and e_min < band.energy < e_max:
            x_text = 1.01
            y_text = band.energy - 0.05
            plt.text(x_text, y_text, band.band_num, size="xx-small")
    plt.xlim([0.0, 1.0])
    plt.ylim([e_min, e_max])
    return fig

# -------------------------------------------------------------------------------------
# GET DOS
# -------------------------------------------------------------------------------------

def get_dos(
    filename: str,
    e_fermi: float = None,
):
    """
    Get DOS from output.
    """
    with open(filename, "r") as fileobj:
        lines = fileobj.readlines()
    if e_fermi is None:
        e_fermi = float(lines[0].split()[-2])
    if "dosup(E)" in lines[0] or "ldosup(E)" in lines[0]:
        nspin = 2
    else:
        nspin = 1
    energy = np.zeros(len(lines) - 1)
    if nspin == 2:
        dos = np.zeros((len(lines) - 1, 2))
    else:
        dos = np.zeros(len(lines) - 1)
    for ii, line in enumerate(lines[1:]):
        energy[ii] = float(line.split()[0]) - e_fermi
        if nspin == 2:
            dos[ii, 0] = float(line.split()[1])
            dos[ii, 1] = float(line.split()[2])
        else:
            dos[ii] = float(line.split()[1])
    # Return energy and DOS.
    return energy, dos

# -------------------------------------------------------------------------------------
# GET PDOS VECT
# -------------------------------------------------------------------------------------

def get_pdos_list(
    atoms: Atoms,
    e_fermi: float = None,
    filename: str = "projwfc.pwo",
):
    """
    Get pDOS vector from output.
    """
    if e_fermi is None:
        e_fermi = atoms.calc.eFermi
    states_list = read_projwfc(filename=filename, kpoint=None)[0]
    pdos_list = np.array([None] * len(atoms), dtype=object)
    names_list = []
    for state in states_list:
        atom_num = state.atom_num
        orbital_type = state.orbital[:1]
        name = "pdos.pdos_atm#{0}({1})_wfc#{2}({3})".format(
            atom_num, state.element, state.shell_num, orbital_type
        )
        if name not in names_list:
            names_list += [name]
            energy, pdos = get_dos(filename=name, e_fermi=e_fermi)
            if pdos_list[atom_num - 1] is None:
                pdos_list[atom_num - 1] = {}
            if orbital_type in pdos_list[atom_num - 1]:
                pdos_list[atom_num - 1][orbital_type] += pdos
            else:
                pdos_list[atom_num - 1][orbital_type] = pdos
    # Return energy and pDOS list.
    return energy, pdos_list

# -------------------------------------------------------------------------------------
# PLOT DOS
# -------------------------------------------------------------------------------------

def plot_dos(
    energy: np.ndarray,
    dos: np.ndarray,
    filename: str ="dos.png",
    x_max: float = 10.0,
):
    """
    Plot DOS.
    """
    import matplotlib.pyplot as plt
    # Plot DOS.
    fig = plt.figure()
    plt.plot(dos, energy)
    plt.plot([0.0, x_max], [0.0] * 2, color="red")
    # Set limits and labels.
    plt.xlim([0.0, x_max])
    plt.ylim([-15.0, +10.0])
    plt.xlabel("DOS [a.u.]")
    plt.ylabel("Energy [eV]")
    # Save the plot.
    plt.savefig("dos.png", dpi=300)
    plt.close()

# -------------------------------------------------------------------------------------
# PLOT PDOS LIST
# -------------------------------------------------------------------------------------

def plot_pdos_list(
    atoms: Atoms,
    energy: np.ndarray,
    pdos: np.ndarray,
    filename: str ="pdos.png",
    x_max: float = 10.0,
):
    """
    Plot PDOS list.
    """
    import matplotlib.pyplot as plt
    color_dict = {
        "s": "limegreen",
        "p": "darkorange",
        "d": "royalblue",
        "f": "crimson",
    }
    # Plot PDOS for each atom.
    for ii, atom in enumerate(atoms):
        fig = plt.figure(ii)
        plt.xlim([0.0, x_max])
        plt.ylim([-15.0, +10.0])
        plt.xlabel("pDOS [a.u.]")
        plt.ylabel("Energy [eV]")
        pdos_dict = pdos[ii]
        for orbital_type in pdos_dict:
            pdos = pdos_dict[orbital_type]
            color = color_dict[orbital_type]
            plt.plot(pdos, energy, color=color)
        plt.plot([0.0, x_max], [0.0] * 2, color="red")
        plt.savefig(f"pdos_atm#{ii + 1}({atom.symbol}).png", dpi=300)
        plt.close()

# -------------------------------------------------------------------------------------
# GET FEATURES BANDS
# -------------------------------------------------------------------------------------

def get_features_bands(
    atoms: Atoms,
    energy: np.ndarray,
    pdos_vect: list,
    delta_e: float = 0.1,
    save_pickle: bool = True,
):
    """
    Get the features of bands.
    """
    i_zero = np.argmin(np.abs(energy))
    i_minus = np.argmin(np.abs(energy + delta_e))
    i_plus = np.argmin(np.abs(energy - delta_e))
    features = np.zeros((len(atoms), 8))
    for i, _ in enumerate(atoms):
        pdos_dict = pdos_vect[i]
        for orbital in pdos_dict:
            if len(pdos_dict[orbital].shape) > 1:
                pdos_dict[orbital] = np.sum(pdos_dict[orbital], axis=1)
        pdos_sp = pdos_dict["s"]
        pdos_sp += pdos_dict["p"]
        sp_filling = np.trapz(y=pdos_sp[:i_zero], x=energy[:i_zero])
        sp_density = np.sum(pdos_sp[i_minus:i_plus]) / len(pdos_sp[i_minus:i_plus])
        if "d" in pdos_dict:
            pdos_d = pdos_dict["d"]
            d_filling = np.trapz(y=pdos_d[:i_zero], x=energy[:i_zero])
            d_density = np.sum(pdos_d[i_minus:i_plus]) / len(pdos_d[i_minus:i_plus])
            d_centre = np.trapz(pdos_d * energy, energy) / np.trapz(pdos_d, energy)
            d_mom_2 = np.trapz(
                pdos_d * np.power(energy - d_centre, 2), energy
            ) / np.trapz(pdos_d, energy)
            d_width = np.sqrt(d_mom_2)
            d_mom_3 = np.trapz(
                pdos_d * np.power(energy - d_centre, 3), energy
            ) / np.trapz(pdos_d, energy)
            d_skewness = d_mom_3 / np.power(d_width, 3)
            d_mom_4 = np.trapz(
                pdos_d * np.power(energy - d_centre, 4), energy
            ) / np.trapz(pdos_d, energy)
            d_kurtosis = d_mom_4 / np.power(d_width, 4)
        else:
            d_filling = np.nan
            d_density = np.nan
            d_centre = np.nan
            d_width = np.nan
            d_skewness = np.nan
            d_kurtosis = np.nan
        features[i, 0] = d_filling
        features[i, 1] = d_centre
        features[i, 2] = d_width
        features[i, 3] = d_skewness
        features[i, 4] = d_kurtosis
        features[i, 5] = sp_filling
        features[i, 6] = d_density
        features[i, 7] = sp_density
        if save_pickle is True:
            import pickle
            with open("features_bands.pickle", "wb") as fileobj:
                pickle.dump(features, fileobj)
    return features

# -------------------------------------------------------------------------------------
# WRITE FEATURES OUT
# -------------------------------------------------------------------------------------

def write_features_out(
    atoms: Atoms,
    features_names: list,
    features: np.ndarray,
    filename: str,
):
    """
    Write the features of bands.
    """
    with open(filename, "w+") as fileobj:
        print("Calculated Features", file=fileobj)
        assert len(features_names) == features.shape[1]
        print(f'{"symbol":7s}', end="", file=fileobj)
        for ii in range(features.shape[1]):
            print(f"  {features_names[ii]:11s}", end="", file=fileobj)
        print("", file=fileobj)
        for ii in range(features.shape[0]):
            print(f"{atoms[ii].symbol:7s}", end="", file=fileobj)
            for feature in features[ii, :]:
                print(f"{feature:+13.4e}", end="", file=fileobj)
            print("", file=fileobj)

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
