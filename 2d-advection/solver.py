from mpl_toolkits.mplot3d import Axes3D    ##New Library required for projected 3d plots
import numpy
from matplotlib import pyplot, cm
from scipy.io import savemat

plot = 1

###variable declarations
nx = 81
ny = 81
nt = 100
c = 1
dx = 2 / (nx - 1)
dy = 2 / (ny - 1)
sigma = .2
dt = sigma * dx


# init variables needed for pysindy
times = numpy.arange(0, dt*nt, dt).reshape(-1, 1)
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
uf = numpy.zeros(((len(x), len(y), times.shape[0])))
du = numpy.zeros_like(uf)


u = numpy.ones((ny, nx)) ##create a 1xn vector of 1's
un = numpy.ones((ny, nx)) ##

###Assign initial conditions

##set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
u[int(.5 / dy):int(1 / dy + 1),int(.5 / dx):int(1 / dx + 1)] = 2 

u = numpy.ones((ny, nx))
rhs = numpy.zeros_like(u)
u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2


for n in range(nt): ##loop across number of time steps
    un = u.copy()
    u[1:, 1:] = (un[1:, 1:] - (c * dt / dx * (un[1:, 1:] - un[1:, :-1])) -\
                              (c * dt / dy * (un[1:, 1:] - un[:-1, 1:])))
    rhs[1:, 1:] =  - (c / dx * (un[1:, 1:] - un[1:, :-1])) -\
                              (c / dy * (un[1:, 1:] - un[:-1, 1:]))
    u[0, :] = 1
    u[-1, :] = 1
    u[:, 0] = 1
    u[:, -1] = 1

    uf[:, :, n] = u
    du[:, :, n] = rhs



#data = {"t": times, "x": x, "y": y, "uf": uf, "vf": uf, "duf": du, "dvf": du}
#savemat("advection_small.mat", data)


fig = pyplot.figure(figsize=(11, 7), dpi=100)
ax = fig.gca(projection='3d')
X, Y = numpy.meshgrid(x, y) 
surf2 = ax.plot_surface(X, Y, u[:], cmap=cm.viridis)
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
pyplot.savefig('./adv.png', dpi=600)
pyplot.show()
