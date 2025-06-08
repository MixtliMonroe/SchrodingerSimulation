import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def normalize(psi: np.ndarray, dx: float) -> np.ndarray:
    '''
    Normalises a given wavefunction
    inputs:
        - psi: wavefunction to be normalized
        - dx: x step size
    outputs:
        - normalized wavefunction
    '''
    norm = sum(dx*np.abs(psi)**2)
    return psi / np.sqrt(norm)

def wavepacket(x: float, dx: float, x0: float = 0, sigma: float = 1, p: float = 0) -> complex:
    '''
    Returns the wavefunction of a 1D wavepacket as a function of x
    inputs:
        - x: position parameter
        - x0: centre position of wavepacket
        - sigma: standard deviation of wavepacket
        - p: momentum of wavepacket
    outputs:
        - wavefunction at position x
    '''
    return normalize(np.exp(-((x-x0)/sigma)**2 + p*x*1j), dx)

def position_space_to_momentum_space(psi: np.ndarray, dx: float, hbar: float = 1) -> tuple:
    '''
    Converts a wavefunction from position space to momentum space
    inputs:
        - psi: wavefunction in position space
        - dx: x step size
        - hbar: reduced planck constant
    outputs:
        - momentum parameteer
        - momentum wavefunction
    '''
    phi = np.fft.fft(psi)/np.sqrt(hbar)
    p = np.fft.fftfreq(psi.size, d=dx)/hbar
    dp = p[1]-p[0]

    phi = normalize(phi, dp)
    p = np.concatenate((p[int((len(p)+1)/2):], p[:int((len(p)+1)/2)])) # numpy arranges this array in a weird order, this line is to fix it
    phi = np.concatenate((phi[int((len(phi)+1)/2):], phi[:int((len(phi)+1)/2)])) # numpy arranges this array in a weird order, this line is to fix it
    return 2*np.pi*p, phi

def d2psi_dx2(psi: np.ndarray, dx: float) -> np.ndarray:
    '''
    Calculates the second derivative of psi w.r. to x using finite differences
    *** Assumes psi has periodic boundaries ***
    inputs:
        - psi: wavefunction in position space
        - dx: x step size
    outputs:
        - second derivative of psi w.r. to x
    '''
    d2psi_dx2 = np.array([psi[(i+1)%len(psi)] - 2*psi[i] + psi[i-1] for i in range(len(psi))])
    return d2psi_dx2 / dx

def dpsi_dt(psi: np.ndarray, dx: float, V: np.ndarray, m: float = 1, hbar: float = 1) -> np.ndarray:
    '''
    Uses Schrodinger's equation to calculate the time derivative of psi in position space
    inputs:
        - psi: wavefunction in position space
        - dx: x step size
        - V: potential function
        - m: particle mass
        - hbar: reduced planck constant
    outputs:
        - time derivative of psi
    '''
    dpsi_dt = -1j/hbar * (-hbar**2/(2*m) * d2psi_dx2(psi, dx) + V*psi)
    return dpsi_dt

def rk4(psi: np.ndarray, dt: float, *kwargs) -> np.ndarray:
    '''
    Calculates psi for the next timestep using the 4th-order Runge-Kutta method
    inputs:
        - psi: wavefunction in position space
        - dt: timestep size
        - *kwargs: other arguments for dpsi_dt function
    outputs:
        - psi at the next timestep
    '''
    k1 = dpsi_dt(psi, *kwargs)
    k2 = dpsi_dt(psi+dt/2 * k1, *kwargs)
    k3 = dpsi_dt(psi+dt/2 * k2, *kwargs)
    k4 = dpsi_dt(psi+dt * k3, *kwargs)
    return psi + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

def simulate(psi: np.ndarray, dx: float, T: float, V: np.ndarray, dt: float = 1e-2, m: float = 1, hbar: float = 1) -> list:
    '''
    Simulates the evolution of the wavefunction over time to time T
    inputs:
        - psi: initial wavefunction in position space
        - dx: x step size
        - T: total simulation time
        - V: potential function
        - dt: timestep size
        - m: particle mass
        - hbar: reduced planck constant
    outputs:
        - list of wavefunctions at each timestep
    '''
    simulation_steps = [psi]
    t = 0

    while t < T:
        psi = rk4(psi, dt, dx, V, m, hbar)
        simulation_steps += [psi]
        t += dt
    
    return simulation_steps

def plot_probability_density(ax: plt.Axes, wavefunction: np.ndarray, s: np.ndarray, V: np.ndarray = None, plot_square_wavefunction: bool = True, real_imag: bool = False) -> tuple:
    '''
    Returns the plot of the probability density of the wavefunction
    inputs:
        - ax: axes to plot on
        - wavefunction: wavefunction parametrised in s
        - s: independent variable (could be x or p)
        - V: potential function (optional)
        - plot_square_wavefunction: if true, plot the square of the magnitude of the wavefunction, otherwise, plot the magnitude of the wavefunction
        - real_imag: whether to plot the real and imaginary parts of the wavefunction
    outputs:
        - tuple of probability density plots as matplotlib objects
    '''
    if plot_square_wavefunction: # Probability density plot
        probability_density = np.abs(wavefunction)**2
        prob, *_ = ax.plot(s, probability_density, color="k", ls="solid", label=r"$|\Psi|^2$")
    else:
        probability_density = np.abs(wavefunction)
        prob, *_ = ax.plot(s, probability_density, color="k", ls="solid", label=r"$|\Psi|$")
    
    if V is not None: # Potential plot
        potential, *_ = ax.plot(s, V, color="green", ls="dotted", label=r"$V$")
        ax.fill_between(s, 0, V, color="green", alpha=0.5)
    else:
        potential = None

    if real_imag: # Real and imaginary plots
        real, *_ = ax.plot(s, np.real(wavefunction), color="r", ls="dashed", lw=1, label=r"$\Re(\Psi)$")
        imag, *_ = ax.plot(s, np.imag(wavefunction), color="b", ls="dashed", lw=1, label=r"$\Im(\Psi)$")
    else:
        real, imag = None, None

    return prob, real, imag, potential

def plot_complex(ax: plt.Axes, wavefunction: np.ndarray):
    '''
    Returns the plot of the wavefunction in the complex plane
    inputs:
        - ax: axes to plot on
        - wavefunction: wavefunction
    outputs:
        - complex plot as a matplotlib object
    '''
    complex_plt, *_ = ax.plot(np.real(wavefunction), np.imag(wavefunction), color="k", ls="solid", label=r"$\Psi$")
    ax.grid()
    return complex_plt

def animate_wavefunction(simulation_frames: list, x: np.ndarray, dt: float = 1e-2, V: np.ndarray = None, hbar: float = 1):
    '''
    Constructs an animation of the wavefunction evolving over time which includes a probability density plot, a complex plot, a momentum probability plot, and a momentum complex plot.
    inputs:
        - simulation_frames: list of wavefunctions at each timestep
        - dt: timestep size
        - x: independent variable (could be x or p)
        - V: potential function (optional)
    outputs:
        - animation as a matplotlib object
    '''
    fig = plt.figure()
    ax = fig.subplots(ncols=2, nrows=2)
    fig.tight_layout(h_pad=1.5)


    # Plot initial wavefunction:

    # probability density plot
    ax[0][0].set_title('Spatial probability plot')
    prob, real, imag, potential = plot_probability_density(ax[0][0], simulation_frames[0], x, V, plot_square_wavefunction = False, real_imag = True)
    ax[0][0].set_xlim(-2,2)
    ax[0][0].set_ylim(-1.5,2.5)
    ax[0][0].set_box_aspect(1)
    ax[0][0].legend(loc="upper left", prop={'size': 6})
    # complex plot
    ax[0][1].set_title('Spatial complex plot')
    complex_plt = plot_complex(ax[0][1], simulation_frames[0])
    ax[0][1].set_xlim(-1.4,1.4)
    ax[0][1].set_ylim(-1.4,1.4)
    ax[0][1].grid()
    ax[0][1].set_box_aspect(1)
    # momentum probability plot
    ax[1][0].set_title('Momentum probability plot')
    p, phi = position_space_to_momentum_space(simulation_frames[0], x[1]-x[0], hbar)
    p_prob, p_real, p_imag, _ = plot_probability_density(ax[1][0], phi, p, plot_square_wavefunction = True, real_imag = False)
    ax[1][0].set_xlim(-40,40)
    ax[1][0].set_ylim(-0.2,2)
    ax[1][0].set_box_aspect(1)
    ax[1][0].legend(loc="upper left", prop={'size': 6})
    #momentum complex plot
    ax[1][1].set_title('Momentum complex plot')
    p_complex_plt = plot_complex(ax[1][1], phi)
    ax[1][1].set_xlim(-1.4,1.4)
    ax[1][1].set_ylim(-1.4,1.4)
    ax[1][1].grid()
    ax[1][1].set_box_aspect(1)

    def animate(frame):
        print(f"Frame: {frame}/{len(simulation_frames)-1}        \r")
        prob.set_data((x, np.abs(simulation_frames[frame]))) # Update probability density plot with |psi| **NOT |psi|^2**
        real.set_data((x, np.real(simulation_frames[frame]))) # Update real part of probability density plot
        imag.set_data((x, np.imag(simulation_frames[frame]))) # Update imaginary part of probability density plot

        complex_plt.set_data((np.real(simulation_frames[frame]), np.imag(simulation_frames[frame]))) # Update complex plot

        p, phi = position_space_to_momentum_space(simulation_frames[frame], x[1]-x[0], hbar) # Update momentum space
        p_prob.set_data((p, np.abs(phi)**2)) # Update momentum probability plot

        p_complex_plt.set_data((np.real(phi), np.imag(phi))) # Update momentum complex plot

        return prob, real, imag, p_prob, p_real, p_imag, p_complex_plt

    anim = FuncAnimation(fig, animate, frames=int(len(simulation_frames)), interval=dt*1000)

    return anim


if __name__ == "__main__":
    # Example: Simulating a wavepacket in a 1D infinite square well
    X = np.linspace(-20, 20, 5000)
    dx = X[1] - X[0]
    psi = wavepacket(X, dx, x0=0, sigma=0.5, p=20)

    V = np.zeros(len(X))
    for i,x in enumerate(X):
        if x < -1.9 or x > 1.9:
            V[i] = 30

    simulation_steps = simulate(psi, dx, 120, V)

    animation = animate_wavefunction(simulation_steps, X, V=V)
    plt.show()
