#!/usr/bin/env python

# -------------------------------------------------------------------------------------
# IMPORTS
# -------------------------------------------------------------------------------------

import os
import matplotlib
matplotlib.use("Agg")
import argparse
from ase.neb import NEB, interpolate, idpp_interpolate
from ase.optimize.lbfgs import LBFGS
from ase.gui.gui import GUI
from ase.io.animation import write_gif
from ase.calculators.espresso import Espresso
from qe_toolkit.io import read_pwi, read_pwo
from qe_toolkit.utils import swap_atoms, write_atoms_pickle
from qe_toolkit.neb import (
    write_neb_inp,
    read_neb_crd,
    read_neb_path,
    reorder_atoms_neb,
    get_atoms_ts_from_neb,
)

# -------------------------------------------------------------------------------------
# PARSER
# -------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--step",
    "-s",
    type=int,
    required=True,
    help="Step of simulation. (0, 1, 2): NEB. (-1): TS search. (-2): SCF.",
)

parser.add_argument(
    "--max_seconds",
    "-ms",
    type=int,
    required=False,
    default=36000,
    help="Maximum number of seconds.",
)

parser.add_argument(
    "--qe_pwi",
    "-pwi",
    type=str,
    required=False,
    default="pw.pwi",
    help="Quantum Espresso input file.",
)

parser.add_argument(
    "--qe_pwo",
    "-pwo",
    type=str,
    required=False,
    default="pw.pwo",
    help="Quantum Espresso output file.",
)

parsed_args = parser.parse_args()

# -------------------------------------------------------------------------------------
# CONTROL
# -------------------------------------------------------------------------------------

reorder_atoms = False
indices_swap = []
intepol_method = "idpp"
steps_idpp = 1e3
fmax_idpp = 1e-2

n_images = 10
filename_neb = "neb.pwi"
filename_path = "pwscf.path"
filename_crd = "pwscf.crd"
filename_ts_search = "atoms_ts.pickle"
first_dir = "./first"
last_dir = "./last"
show_initial = False
show_neb_path = False
write_images = True
repetitions_gif = (1, 1, 1)

# -------------------------------------------------------------------------------------
# STEPS
# -------------------------------------------------------------------------------------

if parsed_args.step == 0:
    neb_data_new = dict(
        nstep_path=200,
        opt_scheme="quick-min",
        CI_scheme="no-CI",
        ds=1,
        k_max=0.40,
        k_min=0.20,
        path_thr=0.10,
        use_masses=True,
        use_freezing=True,
        restart_mode="from_scratch",
    )
    input_data_new = dict(
        ecutwfc=25.0,
        ecutrho=200.0,
        max_seconds=parsed_args.max_seconds,
    )
    gamma = True
    restart_from_crd = False

elif parsed_args.step == 1:
    neb_data_new = dict(
        nstep_path=200,
        opt_scheme="quick-min",
        CI_scheme="auto",
        ds=1,
        k_max=0.60,
        k_min=0.20,
        path_thr=0.10,
        use_masses=True,
        use_freezing=True,
        restart_mode="from_scratch",
    )
    input_data_new = dict(
        max_seconds=parsed_args.max_seconds,
    )
    gamma = False
    restart_from_crd = True

elif parsed_args.step == 2:
    neb_data_new = dict(
        nstep_path=1000,
        opt_scheme="broyden",
        CI_scheme="auto",
        ds=1,
        k_max=0.65,
        k_min=0.30,
        path_thr=0.05,
        use_masses=False,
        use_freezing=False,
        restart_mode="restart",
    )
    input_data_new = dict(
        max_seconds=parsed_args.max_seconds,
    )
    gamma = False
    restart_from_crd = True

elif parsed_args.step == -1:
    neb_data_new = dict()
    input_data_new = dict(
        calculation="scf",
        restart_mode="from_scratch",
        conv_thr=1e-08,
        electron_maxstep=500,
        tprnfor=True,
    )
    gamma = False
    restart_from_crd = True

elif parsed_args.step == -2:
    neb_data_new = dict()
    input_data_new = dict(
        calculation="scf",
        restart_mode="from_scratch",
        electron_maxstep=500,
    )
    gamma = False
    restart_from_crd = True

# -------------------------------------------------------------------------------------
# NEB
# -------------------------------------------------------------------------------------

# Read pwi input file.
atoms = read_pwi(filename=parsed_args.qe_pwi, path_head=first_dir)
input_data = atoms.info["input_data"]
pseudopotentials = atoms.info["pseudopotentials"]
kpts = atoms.info["kpts"]
koffset = atoms.info["koffset"]
# Update input data.
del input_data["calculation"]
del input_data["restart_mode"]
#input_data["scf_must_converge"] = False
#input_data["disk_io"] = "low"
input_data.update(input_data_new)
# Update kpts.
if gamma is True:
    kpts = None
    koffset = None
# Setup espresso calculator.
calc = Espresso(
    input_data=input_data,
    pseudopotentials=pseudopotentials,
    kpts=kpts,
    koffset=koffset,
)
# Read first and last images.
atoms_first = read_pwo(
    filename=parsed_args.qe_pwo,
    filepwi=parsed_args.qe_pwi,
    path_head=first_dir,
)
atoms_last = read_pwo(
    filename=parsed_args.qe_pwo,
    filepwi=parsed_args.qe_pwi,
    path_head=last_dir,
)
# Reorder atoms.
if reorder_atoms is True:
    atoms_first, atoms_last = reorder_atoms_neb(
        atoms_first=atoms_first,
        atoms_last=atoms_last,
    )
# Swap atoms.
if indices_swap:
    atoms_last = swap_atoms(
        atoms=atoms_last,
        indices_swap=indices_swap,
    )
# Show first and last images.
if show_initial is True:
    try: GUI([atoms_first, atoms_last]).run()
    except: pass
# Restart from crd file or interpolate images.
images = [atoms_first]
images += [atoms_first.copy() for i in range(n_images - 2)]
images += [atoms_last]
if restart_from_crd is True:
    images = read_neb_crd(images, filename_crd)
else:
    interpolate(images=images, mic=False, apply_constraint=False)
    if intepol_method == "idpp":
        idpp_interpolate(
            images=images,
            traj=None,
            log=None,
            mic=False,
            steps=steps_idpp,
            fmax=fmax_idpp,
            optimizer=LBFGS,
        )
# Write input files.
if parsed_args.step == -1:
    # TS search calculation.
    index_ts = read_neb_path(images, filename=filename_path, return_index_ts=True)
    images = read_neb_crd(images, filename_crd)
    atoms_ts = get_atoms_ts_from_neb(images, index_ts=index_ts)
    atoms_ts.calc = calc
    os.makedirs('TS_search', exist_ok=True)
    os.chdir('TS_search')
    write_atoms_pickle(
        atoms=atoms_ts,
        filename=filename_ts_search,
    )
    os.chdir('..')
elif parsed_args.step == -2:
    # SCF calculation.
    index_ts = read_neb_path(images, filename=filename_path, return_index_ts=True)
    images = read_neb_crd(images, filename_crd)
    atoms_ts = get_atoms_ts_from_neb(images, index_ts=index_ts)
    os.makedirs('TS_scf', exist_ok=True)
    os.chdir('TS_scf')
    calc.label = 'pw'
    calc.write_input(atoms_ts)
    os.chdir('..')
else:
    # NEB calculation.
    neb_data = dict(
        string_method="neb",
        num_of_images=n_images,
    )
    neb_data.update(neb_data_new)
    write_neb_inp(
        neb_data=neb_data,
        images=images,
        calc=calc,
        filename=filename_neb,
    )

# -------------------------------------------------------------------------------------
# SHOW NEB AND PRINT GIF
# -------------------------------------------------------------------------------------

# Show images.
if show_neb_path is True:
    try: GUI(images).run()
    except: pass
# Write images.
if write_images is True:
    for atoms in images:
        atoms *= repetitions_gif
    write_gif(
        filename="images.gif",
        images=images,
        scale=1000,
        maxwidth=2000,
        radii=0.9,
        rotation="-90x,-30y,+45x",
    )

# -------------------------------------------------------------------------------------
# END
# -------------------------------------------------------------------------------------
