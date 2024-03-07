# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------

import signal
import numpy as np
from time import strftime
from ase.calculators.socketio import SocketIOCalculator
from ase.io import read, Trajectory
from ase.dimer import (
    DimerControl,
    MinModeAtoms,
    MinModeTranslate,
    read_eigenmode,
)
from qe_toolkit.utils import read_atoms_pickle

# -----------------------------------------------------------------------------
# RESTART
# -----------------------------------------------------------------------------

RESTART = False
MAX_SECONDS = 1500
FMAX = 0.05
NSTEPS = 200
MAX_DISPL = 5.0

signal.alarm(MAX_SECONDS)

def get_socket_and_command():
    socket = "espresso_" + strftime("%Y_%m_%d_%H_%M_%S")
    command = f"mpirun pw.x -in PREFIX.pwi --ipi {socket}:UNIX > PREFIX.pwo"
    return socket, command

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------

filename_ts_search = "atoms_ts.pickle"

atoms = read_atoms_pickle(filename=filename_ts_search)
eigenmodes = atoms.info.get("eigenmodes")
vector = atoms.info.get("vector")
bond_indices = atoms.info.get("bond_indices")
calc = atoms.calc

atoms_orig = atoms.copy()

unixsocket, command = get_socket_and_command()

calc.label = "dimer"
calc.command = command

calc.parameters["input_data"].update({
    "conv_thr": 1e-08,
    "electron_maxstep": 500,
})

if RESTART is True:
    atoms = read("dimer.traj")
    eigenmodes = [read_eigenmode("eigenmodes.log")]
    mode_traj = "a"
else:
    mode_traj = "w"

atoms.calc = SocketIOCalculator(calc, unixsocket=unixsocket)

# -----------------------------------------------------------------------------
# DIMER
# -----------------------------------------------------------------------------

mask_dimer = np.array([True] * len(atoms))
mask_dimer[atoms.constraints[0].get_indices()] = False

control_dimer = DimerControl(
    initial_eigenmode_method="displacement",
    displacement_method="vector",
    mask=list(mask_dimer),
    logfile="dimer.log",
    eigenmode_logfile="eigenmodes.log",
    f_rot_min=0.10,
    f_rot_max=1.00,
    trial_angle=np.pi / 4.0,
    trial_trans_step=0.01,
    maximum_translation=0.10,
    dimer_separation=0.01,
    cg_translation=True,
    use_central_forces=True,
    extrapolate_forces=False,
    order=1,
    max_num_rot=1,
)

atoms_minmode = MinModeAtoms(
    atoms=atoms,
    control=control_dimer,
    eigenmodes=eigenmodes,
)

traj_dimer = Trajectory(
    filename="dimer.traj",
    mode=mode_traj,
    atoms=atoms,
    properties=["energy", "forces"],
)

if eigenmodes is None:
    vector /= np.linalg.norm(vector)
    vector *= 0.01
    atoms_minmode.displace(displacement_vector=vector, log=True)

dimer_opt = MinModeTranslate(atoms=atoms_minmode, trajectory=traj_dimer)

# -----------------------------------------------------------------------------
# OBSERVERS
# -----------------------------------------------------------------------------

if MAX_DISPL is not None:

    def check_displacement():
        displ = np.linalg.norm(atoms.positions - atoms_orig.positions)
        print(f"Displacement = {displ:7.4f}")
        if displ > MAX_DISPL:
            dimer_opt.max_steps = 0

    dimer_opt.insert_observer(function=check_displacement, interval=10)

if bond_indices is not None:

    def check_direction():
        dir_vector = (
            +dimer_opt.atoms.eigenmodes[0][bond_indices[1]][:2]
            -dimer_opt.atoms.eigenmodes[0][bond_indices[0]][:2]
        )
        dir_bond = (
            +atoms[bond_indices[1]].position[:2]
            -atoms[bond_indices[0]].position[:2]
        )
        dir_vector /= np.linalg.norm(dir_vector)
        dir_bond /= np.linalg.norm(dir_bond)
        dot_product = np.abs(np.dot(dir_vector, dir_bond))
        print(f"Dot Product  = {dot_product:7.4f}")
        if dot_product < 0.5:
            dimer_opt.max_steps = 0

    dimer_opt.insert_observer(function=check_direction, interval=10)

# -----------------------------------------------------------------------------
# RUN
# -----------------------------------------------------------------------------

count = 0
finished = False
while finished is False and count < 100:
    count += 1
    try:
        dimer_opt.run(fmax=FMAX, steps=NSTEPS)
        if dimer_opt.converged():
            atoms_minmode.summarize()
        finished = True
    except Exception as error:
        print(f"Calculation Failed: {error}")
        unixsocket, command = get_socket_and_command()
        calc.command = command
        atoms.calc = SocketIOCalculator(calc, unixsocket=unixsocket)

if finished is True:
    print("Dimer Calculation Finished")

if not dimer_opt.converged():
    print("Dimer Calculation Not Converged")

# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------
