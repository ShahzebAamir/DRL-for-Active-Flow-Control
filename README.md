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

# Dependencies:
This project requires:
1. Fenics Library (Latest Version can work) (https://fenicsproject.org/download/)
3. Tensorforce Library (Versions 0.4.1)
4. Printind (pip install printind)
5. Numpy, Matplotlib

# Setting up Fenics:
This requires a careful installation as it is not supported in Windows os. Hence, following the steps one can set-up a jupyter notebook with working fenics library.
1. Install Docker Quickstart Terminal (https://docs.bitnami.com/containers/how-to/install-docker-in-windows/)
2. Create a directory in the Users folder (Important to Pull the code in that directory)
3. Go to that directory from your Docker Quickstart Terminal
4. Now install fenics from the docker following this (https://fenics.readthedocs.io/projects/containers/en/latest/jupyter.html)
5. Afterwards, start the container by using: docker start notebook
6. Then execute: docker logs notebook
7. Then copy/paste the ip that is generated for example: http://127.0.0.1:8888/?token=bb5b57183e17f2da23b7766456409498eeda31b732c3d9b1
8. Now replace 127.0.0.1 with the ip that you obtain from running the command (docker-machine ip)
9. Now, you can jupyter notebook with fenics installed

You can set-up fenics in mac and ubuntu as well following the official guide

# Setting up tensorforce from docker
You have to install all the packages from the docker
1. run in docker terminal: docker exec -ti notebook /bin/bash -l
2. type and execute ls
3. run cd local/
4. Now install: pip install tensorforce[tf] (Install version 0.4.2)
5. If you want any other package install using pip install here
  
