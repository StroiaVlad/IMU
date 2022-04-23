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
        velocity_norm[index] = np.linalg.norm(state[3:6, index]) * 1.94384 # from m/s to kts
    ax4.plot(T, velocity_norm, linewidth=2.0)
    ax4.set_title(r'$||V||_{2}$ over time', fontsize=25)
    ax4.set_ylabel(r'$||V||_{2}$ (kts)', fontsize=25)
    ax4.tick_params(axis='x', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)

    fig2.savefig("velocity.png")

def plotAttitude(T, state):
    fig3, (ax1, ax2, ax3) = mpl.subplots(1, 3, figsize=(20, 5))

    ax1.plot(T, state[6, :] * 180/np.pi, linewidth=2.0) # heading in degrees
    #ax1.set_ylim([-0.04, 0.04])
    ax1.set_title(r'$\psi$ (heading) over time', fontsize=25)
    ax1.set_ylabel(r'$\psi^\circ, \theta^\circ, \phi^\circ$', fontsize=25)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax2.plot(T, state[7, :] * 180/np.pi, linewidth=2.0) # attitude in degrees
    #ax2.set_ylim([-0.04, 0.04])
    ax2.set_title(r'$\theta$ (attitude) over time', fontsize=25)
    ax2.set_xlabel("Time(s)", fontsize=25)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    ax3.plot(T, state[8, :] * 180/np.pi, linewidth=2.0) # roll in degrees
    #ax3.set_ylim([-0.04, 0.04])
    ax3.set_title(r'$\phi$ (roll) over time', fontsize=25)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)

    fig3.savefig("attitude.png")

def plotProfile(T, state):
    LLA = np.ones([3, state.shape[1]])
    for index in range(0, int(LLA.shape[1])):
        [phi, lamda, h] = ECEF2LLA(state[0:3, index])
        LLA[0, index] = phi
        LLA[1, index] = lamda
        LLA[2, index] = h

    #print(LLA)
    fig4, ((ax1, ax2), (ax3, ax4)) = mpl.subplots(2, 2, figsize=(15, 10))

    ax1.plot(T, LLA[0, :] * 180/np.pi, linewidth=2.0) # lat in degrees
    ax1.set_title("Lat over time", fontsize=20)
    ax1.set_ylabel(r'$Lat^\circ, Long^\circ$', fontsize=20)
    ax1.tick_params(axis='x', labelsize=15)
    ax1.tick_params(axis='y', labelsize=15)

    ax2.plot(T, LLA[1, :] * 180/np.pi, linewidth=2.0) # long in degrees
    ax2.set_title("Long over time", fontsize=20)
    ax2.tick_params(axis='x', labelsize=15)
    ax2.tick_params(axis='y', labelsize=15)

    ax3.plot(T, LLA[2, :] * 3.28084, linewidth=2.0) # altitude in feet
    ax3.set_title("Altitude Profile", fontsize=20)
    ax3.set_xlabel("Time(s)", fontsize=20)
    ax3.set_ylabel(r'h(ft), $Lat^\circ$', fontsize=20)
    ax3.tick_params(axis='x', labelsize=15)
    ax3.tick_params(axis='y', labelsize=15)

    ax4.scatter(LLA[1, :] * 180/np.pi, LLA[0, :] * 180/np.pi) # both lat and long converted in deg
    #ax4.set_xlim([-0.04, 0.04])
    ax4.set_title("Flight Profile Overview", fontsize=20)
    ax4.set_xlabel(r'$Long^\circ$', fontsize=20)
    ax4.tick_params(axis='x', labelsize=15)
    ax4.tick_params(axis='y', labelsize=15)

    fig4.savefig("profile.png")

    mpl.show()


