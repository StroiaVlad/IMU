import matplotlib.pyplot as mpl
import numpy as np
from conversion import ECEF2LLA

def plotPosition(T, state):
    fig1, (ax1, ax2, ax3) = mpl.subplots(1, 3, figsize = (20, 5))

    ax1.plot(T, state[0, :], linewidth = 2.0)
    ax1.set_title("X Position over time", fontsize = 25)
    ax1.set_ylabel("X(m), Y(m), Z(m)", fontsize = 25)
    ax1.tick_params(axis='x', labelsize = 15)
    ax1.tick_params(axis='y', labelsize = 15)

    ax2.plot(T, state[1, :], linewidth = 2.0)
    ax2.set_title("Y Position over time", fontsize = 25)
    ax2.set_xlabel("Time(s)", fontsize = 25)
    ax2.tick_params(axis='x', labelsize = 15)
    ax2.tick_params(axis='y', labelsize = 15)

    ax3.plot(T, state[2, :], linewidth = 2.0)
    ax3.set_title("Z Position over time", fontsize = 25)
    ax3.tick_params(axis='x', labelsize = 15)
    ax3.tick_params(axis='y', labelsize = 15)

    fig1.savefig("position.png")


def plotVelocity(T, state):
    fig2, ((ax1, ax2), (ax3, ax4)) = mpl.subplots(2, 2, figsize=(15, 10))

    ax1.plot(T, state[3, :], linewidth=2.0)
    ax1.set_title(r'$V_{x}$ over time', fontsize=25)
    ax1.set_ylabel(r'$V_{x}(m/s), V_{y}(m/s), V_{z}(m/s)$', fontsize=25)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax2.plot(T, state[4, :], linewidth=2.0)
    ax2.set_title(r'$V_{y}$ over time', fontsize=25)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    ax3.plot(T, state[5, :], linewidth=2.0)
    ax3.set_title(r'$V_{z}$ over time', fontsize=25)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)

    velocity_norm = np.zeros(T.shape)
    for index in range(0, T.shape[0]):
        velocity_norm[index] = np.linalg.norm(state[3:6, index])
    ax4.plot(T, velocity_norm, linewidth=2.0)
    ax4.set_title(r'$||V||_{2}$ over time', fontsize=25)
    ax4.tick_params(axis='x', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)

    fig2.savefig("velocity.png")

def plotAttitude(T, state):
    fig3, (ax1, ax2, ax3) = mpl.subplots(1, 3, figsize=(20, 5))

    ax1.plot(T, state[6, :], linewidth=2.0)
    ax1.set_title(r'$\theta$ (roll) over time', fontsize=25)
    ax1.set_ylabel(r'$\theta^\circ, \phi^\circ, \psi^\circ$', fontsize=25)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax2.plot(T, state[7, :], linewidth=2.0)
    ax2.set_title(r'$\phi$ (pitch) over time', fontsize=25)
    ax2.set_xlabel("Time(s)", fontsize=25)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    ax3.plot(T, state[8, :], linewidth=2.0)
    ax3.set_title(r'$\psi$ (yaw) over time', fontsize=25)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)

    fig3.savefig("attitude.png")

def plotProfile(T, state):
    LLA = np.ones([3, state.shape[1]])
    for index in range(0, int(LLA.shape[1])):
        [phi, lamda, h] = ECEF2LLA(state[0:3, index])
        LLA[0, index] = phi
        LLA[1, index] = lamda
        LLA[2, index] = (h / 1000 + 6356.75231425) * 3.28084  # to convert to km, subtract the radius of the earth and then convert to ft

    #print(LLA)
    fig4, ((ax1, ax2), (ax3, ax4)) = mpl.subplots(2, 2, figsize=(15, 10))

    ax1.plot(T, LLA[0, :], linewidth=2.0)
    ax1.set_title("Lat over time", fontsize=20)
    ax1.set_ylabel(r'$Lat^\circ, Long^\circ$', fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax2.plot(T, LLA[1, :], linewidth=2.0)
    ax2.set_title("Long over time", fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    ax3.plot(T, LLA[2, :], linewidth=2.0)
    ax3.set_title("Altitude Profile", fontsize=20)
    ax3.set_xlabel("Time(s)", fontsize=20)
    ax3.set_ylabel(r'h(ft), $Lat^\circ$', fontsize=20)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)

    ax4.scatter(LLA[1, :], LLA[0, :])
    ax4.set_title("Flight Profile Overview", fontsize=20)
    ax4.set_xlabel(r'$Long^\circ$', fontsize=20)
    ax4.tick_params(axis='x', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)

    fig4.savefig("profile.png")

    mpl.show()


