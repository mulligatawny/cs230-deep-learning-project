###############################################################################
# 1D Burgers' Equation Solver using Galerkin's Method and Fourier expansions  #
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

plot = 0

N = 32
nu = 0.4
x = np.linspace(0, 2*np.pi, N+1)[:-1]
u0 = 5*np.sin(x) # initial condition
y0 = np.fft.fft(u0)/N
k = np.arange(-N/2, N/2)
k = np.fft.fftshift(k)

def fun(t, y):
    df = -nu*(k[i]**2)*y[i]
    ad = 0.0
    for m in range(N):
        for n in range(N):
            if k[m]+k[n] == k[i]:
                ad = ad + 1j*k[m]*y[m]*y[n]
    return -ad + df

dt = 0.004
t = 0.0
y = y0
yn = np.zeros_like(x, dtype='complex')
tf = 0.4

times = np.arange(0, tf, dt).reshape(-1,1)

uf = np.zeros((len(x), times.shape[0]))
du = np.zeros_like(uf)

ks = np.zeros_like(x, dtype='complex')

# Runge-Kutta IV time integrator
for it, t in enumerate(times):
    for i in range(N):
        k1 = dt*fun(t, y)
        k2 = dt*fun(t+dt/2, y+k1/2)
        k3 = dt*fun(t+dt/2, y+k2/2)
        k4 = dt*fun(t+dt, y+k3)
        ks[i] = fun(t, y)  # KC
        yn[i] = y[i] + k1/6 + k2/3 + k3/3 + k4/6
    y = yn
    uf[:, it] = np.real(np.fft.ifft(y)*N)  # KC

    du[:, it] = np.real(np.fft.ifft(ks)*N)  # KC

    t = t + dt

ye = np.real(np.fft.ifft(y)*N)

if plot:
    plt.plot(x, u0, 'k-', label='I.C.')
    plt.plot(x, ye, 'o-', color='orangered')#, label='t = {:.1f}'.format(t))
    plt.legend()
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.show()

bog = np.zeros(((N, N, 100)))

uf = uf.reshape((1, uf.shape[0], uf.shape[1]))

uf = uf +bog

du = du.reshape((1, du.shape[0], du.shape[1]))
du = du +bog

plt.plot(x, uf[:, 0, 20])
plt.show()

data = {"t": times, "x": x, "y": y, "uf": uf, "vf": uf, "duf": du, "dvf": du}
#savemat("burgers_smaller.mat", data)
