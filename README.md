# DRL-for-Active-Flow-Control
This repo contains code for the final project of Reinforcement Learning Course. The code executes active flow control of a fluid past a cylinder using Deep Reinforcement Learning. This project is referenced from Rabault, J., Kuchta, M., Jensen, A., Reglade, U., Cerardi, N., 2018. Artificial Neural Networks trained through Deep Reinforcement Learning discover control strategies for active flow control. arXiv preprint arXiv:1808.07664.

# Structure of Repo:
The main code for the project is in the folder named code. It contains following files:
1. iUFL (It is a package necessary for the fenics framework)
2. Mesh (This folder contains the mesh files 'turek_2d.msh')
3. WorkingProbe.py (This contains the calculation of drag, lift (on cylinder) velocity, and pressure at the given probes)
4. WorkingMshGenerator.py (This generates new mesh, although currently one can work without it using already prepared mesh)
5. WorkingMshConvertor.py (This converts .msh file to .h5 file, which is a input type to fenics solver)
6. jet_bcs.py (This is necessary to set-up jet boundary conditions as they are user-defined derived from UserExpression functionality of fenics)
7. WorkingFlowSolver.py (This contains the implementation of IPCS solver scheme using fenics framework) (Inspired from https://fenicsproject.org/pub/tutorial/html/._ftut1009.html)
8. Working_2DCylinder_EnvDef (This creates the environment for RL algorithm, actions, rewards, necesasary plottings and data-saving)
9. Env_instance (This instantiates the environment object)
10. Main_leanring_PPO.ipynb (This sets up the DRL algorithm (Proximal Policy Optimization) and performs training of the agent)
11. Other files are just to gather results
