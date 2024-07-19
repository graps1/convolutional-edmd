There are two main files in this repository:
- `ks2d.ipynb` contains an implementation of the convolutional EDMD approach for the two-dimensional Kuramoto-Sivashinsky equation. 
- `spring_system.ipynb` contains code for the simply symmetric spring system of four nodes.

Plots created by `ks2d.ipynb` are stored in `plots/`. The data used for EDMD is stored in `ks2d_long.npy` (a numpy array of size * x 16 x 16, where * is the batch size). This data is created when one runs `simulation.py` with the initial condition stored in `ks2d_initial.npy`.