import numpy as np
import sympy as sp
import scipy.sparse as sparse
import pytest
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

x, y, t = sp.symbols("x,y,t")


class Wave2D:

    def create_mesh(self, N, sparse=False):
        """Create 2D mesh and store in self.xij and self.yij"""
        self.x = np.linspace(0, 1, N + 1)
        self.dx = 1 / N
        self.xij, self.yij = np.meshgrid(self.x, self.x, indexing="ij", sparse=sparse)
        return self.xij, self.yij

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), "lil")
        D[0] = 0
        D[-1] = 0
        return D

    @property
    def w(self):
        """Return the dispersion coefficient"""
        k_x = self.mx * sp.pi
        k_y = self.my * sp.pi
        w = self.c * sp.sqrt((k_x**2 + k_y**2))
        return w

    def ue(self, mx, my):
        """Return the exact standing wave"""
        return sp.sin(mx * sp.pi * x) * sp.sin(my * sp.pi * y) * sp.cos(self.w * t)

    def initialize(self, N, mx, my):
        r"""Initialize the solution at $U^{n}$ and $U^{n-1}$

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        mx, my : int
            Parameters for the standing wave
        """
        self.mx = mx
        self.my = my
        self.create_mesh(N)

        self.u_exact = sp.lambdify([x, y], self.ue(mx, my).subs(t, 0))
        u0 = self.u_exact(self.xij, self.yij)

        D = self.D2(N) / self.dx**2
        u1 = u0 + 0.5 * (self.c * self.dt) ** 2 * (D @ u0 + u0 @ (D.T))

        return u0, u1

    @property
    def dt(self):
        """Return the time step"""
        return self.cfl * self.dx / self.c

    def l2_error(self, u, t0):
        """Return l2-error norm

        Parameters
        ----------
        u : array
            The solution mesh function
        t0 : number
            The time of the comparison
        """
        u_exact = sp.lambdify([x, y, t], self.ue(self.mx, self.my))
        exact = u_exact(self.xij, self.yij, t0)
        error = u - exact

        return np.sqrt(self.dx**2 * np.sum(error**2))

    def apply_bcs(self):
        self.Unp1[0] = 0
        self.Unp1[-1] = 0
        self.Unp1[:, -1] = 0
        self.Unp1[:, 0] = 0

    def __call__(self, N, Nt, cfl=0.5, c=1.0, mx=3, my=3, store_data=-1):
        """Solve the wave equation

        Parameters
        ----------
        N : int
            The number of uniform intervals in each direction
        Nt : int
            Number of time steps
        cfl : number
            The CFL number
        c : number
            The wave speed
        mx, my : int
            Parameters for the standing wave
        store_data : int
            Store the solution every store_data time step
            Note that if store_data is -1 then you should return the l2-error
            instead of data for plotting. This is used in `convergence_rates`.

        Returns
        -------
        If store_data > 0, then return a dictionary with key, value = timestep, solution
        If store_data == -1, then return the two-tuple (dx, l2-error)
        """
        self.cfl = cfl
        self.c = c

        self.mx = mx
        self.my = my

        Unp1, Un, Unm1 = np.zeros((3, N + 1, N + 1))
        Unm1[:], Un[:] = self.initialize(N, mx, my)
        D = self.D2(N) / self.dx**2

        plotdata = {0: Unm1.copy()}
        l2_err = [self.l2_error(Unm1, 0), self.l2_error(Un, self.dt)]

        for n in range(1, Nt):
            Unp1[:] = 2 * Un - Unm1 + (c * self.dt) ** 2 * (D @ Un + Un @ D.T)
            # Set boundary conditions
            self.Unp1 = Unp1
            self.apply_bcs()
            Unp1 = self.Unp1
            # Swap solutions
            Unm1[:] = Un
            Un[:] = Unp1

            l2_err.append(self.l2_error(Un, self.dt * (n + 1)))

            if n % store_data == 0:
                plotdata[n] = Unm1.copy()  # Unm1 is now swapped to Un

        l2_err.append(self.l2_error(Un, Nt * self.dt))

        if store_data > 0:
            return plotdata
        elif store_data == -1:
            return (self.dx, l2_err)

    def convergence_rates(self, m=4, cfl=0.1, Nt=10, mx=3, my=3):
        """Compute convergence rates for a range of discretizations

        Parameters
        ----------
        m : int
            The number of discretizations to use
        cfl : number
            The CFL number
        Nt : int
            The number of time steps to take
        mx, my : int
            Parameters for the standing wave

        Returns
        -------
        3-tuple of arrays. The arrays represent:
            0: the orders
            1: the l2-errors
            2: the mesh sizes
        """
        E = []
        h = []
        N0 = 8
        for _ in range(m):
            dx, err = self(N0, Nt, cfl=cfl, mx=mx, my=my, store_data=-1)
            E.append(err[-1])
            h.append(dx)
            N0 *= 2
            Nt *= 2
        r = [np.log(E[i - 1] / E[i]) / np.log(h[i - 1] / h[i]) for i in range(1, m, 1)]
        return r, np.array(E), np.array(h)


class Wave2D_Neumann(Wave2D):

    def D2(self, N):
        """Return second order differentiation matrix"""
        D = sparse.diags([1, -2, 1], [-1, 0, 1], (N + 1, N + 1), "lil")
        D[0, 1] = 2
        D[-1, -2] = 2
        return D

    def ue(self, mx, my):
        return sp.cos(mx * sp.pi * x) * sp.cos(my * sp.pi * y) * sp.cos(self.w * t)

    def apply_bcs(self):
        ## Implemented in the D2 function
        pass


def test_convergence_wave2d():
    sol = Wave2D()
    r, E, h = sol.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 1e-2


def test_convergence_wave2d_neumann():
    solN = Wave2D_Neumann()
    r, E, h = solN.convergence_rates(mx=2, my=3)
    assert abs(r[-1] - 2) < 0.05

@pytest.mark.parametrize("m", np.linspace(1, 15, 15))
def test_exact_wave2d(m):
    cfl = 1/np.sqrt(2)
    N = 10
    Nt = 10

    sol = Wave2D()
    dx, errs = sol(N, Nt, cfl=cfl, mx=m, my=m)
    assert np.allclose(errs, np.zeros_like(errs), atol=1e-12)

    solN = Wave2D_Neumann()
    dxN, errsN = solN(N, Nt, cfl=cfl, mx=m, my=m)
    assert np.allclose(errsN, np.zeros_like(errsN), atol=1e-12)


def make_gif():
    N = 300
    Nt = 200
    cfl = 1/np.sqrt(2)
    m = 4

    sol = Wave2D_Neumann()
    data = sol(N, Nt, cfl=cfl, mx=m, my=m, store_data=2)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    frames = []
    for _, val in data.items():
        frame = ax.plot_surface(sol.xij, sol.yij, val, vmin=-0.5*data[0].max(),
                               vmax=data[0].max(), cmap=cm.magma,
                               linewidth=0, antialiased=False)
        frames.append([frame])

    ani = animation.ArtistAnimation(fig, frames, interval=400, blit=True,
                                    repeat_delay=1000)
    ani.save('report/neumannwave.gif', writer='pillow', fps=10)

if __name__ == "__main__":
    make_gif()