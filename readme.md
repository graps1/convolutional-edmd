There are three main files in this repository:
- `ks2d.ipynb` contains an implementation of the convolutional EDMD approach for the two-dimensional Kuramoto-Sivashinsky equation, using the fast Fourier transform. It also plots the figures in the paper.
- `ks2d_detailed.ipynb` contains a slower implementation that computes the same stuff as `ks2d.ipynb`. It doesn't make use of the FFT, but it is much more instructive and closer to the details given in the paper.
- `spring_system.ipynb` contains code for the symmetric spring system of four nodes.

Plots created by `ks2d.ipynb` are stored in `plots/`. The data used for EDMD is stored in `ks2d_long.npy` (a numpy array of size * x 16 x 16, where * is the batch size). This data is created when one runs `simulation.py` with the initial condition stored in `ks2d_initial.npy`.