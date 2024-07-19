import numpy as np
from shenfun import *
from math import remainder

dt = 0.1

save_period_long = 1
start_time_long = 700
end_time_long = 2500
batch_long = []
f_long = open('ks2d_long.npy', 'wb')

# ---SHENFUN STUFF---
# Size of discretization
N = (16, 16)

K0 = FunctionSpace(N[0], 'F', dtype='D', domain=(0, 3*np.pi))
K1 = FunctionSpace(N[1], 'F', dtype='d', domain=(0, 3*np.pi))
T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})
TV = VectorSpace(T)
gradu = Array(TV)

u = TrialFunction(T)
v = TestFunction(T)

# Create solution and work arrays
d = np.load("ks2d_initial.npy")
U = Array(T, buffer=np.array(d))
U_hat = Function(T)
gradu = Array(TV)
K = np.array(T.local_wavenumbers(True, True, True))
mask = T.get_mask_nyquist()

def LinearRHS(self, u, **params):
    # Assemble diagonal bilinear forms
    return -div(grad(u))-div(grad(div(grad(u))))

def NonlinearRHS(self, U, U_hat, dU, gradu, **params):
    # Assemble nonlinear term
    gradu = TV.backward(1j*K*U_hat, gradu)
    dU = T.forward(0.5*(gradu[0]*gradu[0]+gradu[1]*gradu[1]), dU)
    dU.mask_nyquist(mask)
    if comm.Get_rank() == 0:
        dU[0, 0] = 0
    return -dU

#initialize
X = T.local_mesh(True)
U_hat = U.forward(U_hat)
U_hat.mask_nyquist(mask)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
im = ax.imshow(d)        
plt.ion()

# Integrate using an exponential time integrator
def update(self, u, u_hat, t, tstep, **params):
    print(f"time = {t:.3f}/{end_time_long:.3f}",end="\r")
    u = u_hat.backward(u)
    if t > start_time_long and abs(remainder(t, save_period_long)) < 1e-8:
        batch_long.append(np.array(u)) # simply add coefficients into array
        im.set_data(batch_long[-1])        
        plt.pause(0.01)

if __name__ == '__main__':
    par = {
           'gradu': gradu,
           'count': 0}
    print("starting simulation ...")

    integrator = ETDRK4(T, L=LinearRHS, N=NonlinearRHS, update=update, **par)
    integrator.setup(dt)
    U_hat = integrator.solve(U, U_hat, dt, (0, end_time_long))
    np.save(f_long, np.array(batch_long))
    f_long.close()

    plt.ioff()
    plt.show()