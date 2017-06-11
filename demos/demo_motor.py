#!/usr/bin/env python
"""
Using navboxplus to perfectly control a motor sensed with only a cheap encoder.
Model-augmented state is: [position, velocity, drag/inertia, b/inertia, disturbance].

"""
from __future__ import division
import numpy as np; npl = np.linalg
import matplotlib.pyplot as plt
from navboxplus import NavBoxPlus

# Motor dynamics
def motor(x, u, wf, dt):
    xdot = np.array([x[1],
                     x[4] + x[3]*u - x[2]*x[1],
                     0, 0, 0])  # parameters "don't change" (we assume)
    xnext = x + xdot*dt + wf
    if xnext[2] < 0.1: xnext[2] = 0.1  # prevent parameter drift into nonphysical
    if xnext[3] < 0.1: xnext[3] = 0.1
    return xnext

# Encoder model (only noise in the form of discretization)
res = 512/360  # ticks/deg
z_per_t = 50  # samples/s
def encoder(x, u, wh):
    return np.floor(res*x[0])

# True noise characteristics
wf0_true = np.array([0, 0, 0, 0, 0])
Cf_true = np.diag([0, 0, 1E-3, 1E-6, 0])

# Our guesses at the dynamics and sensor noise characteristics
wf0 = np.zeros(5)
Cf = np.diag([1E-7, 1E-4, 1E-3, 1E-6, 1E-4])  # disturbance is not really constant
wh0 = 0
Ch = 1  # just because the encoder discretization acts like noise

# Simulation time domain (also chooses predict frequency)
T = 80  # s
dt = 0.02  # s
t = np.arange(0, T, dt)  # s
i_per_z = int(1/(z_per_t*dt))  # iters/sample

# Desired trajectory
# r = [180, 0] * np.ones((len(t), 2))  # setpoint
rv = 0.5
r = 15*np.vstack((np.sin(rv*t), rv*np.cos(rv*t))).T  # sinusoid

# Unknown external disturbance (tracked as a state)
dist = 3*np.cos(2*rv*(t+2)) + 3  # sinusoid
dist[:len(t)//2] = 0  # but starts as no disturbance

# Controller with feedback and feedforward based on estimated model
gains = np.array([1, 1])
def controller(r, rnext, x, Cx, dt):
    ur = (1/x[3]) * ((rnext[1] - r[1])/dt + x[2]*r[1] - x[4])
    return gains.dot(r - x[:2]) + ur

# State, estimate, covariance, and measurement timeseries
x = np.zeros((len(t), 5))
xh = np.zeros((len(t), 5))
Cx = np.zeros((len(t), 5, 5))
z = np.zeros((len(t), 1))

# Initial conditions
x[0] = [15, 0, 5, 2, dist[0]]
xh[0] = [-15, 10, 1, 1, 0]
Cx[0] = 10*np.eye(5)

# Configure navboxplus
nav = NavBoxPlus(x0=np.copy(xh[0]),
                 Cx0=np.copy(Cx[0]),
                 g=controller,
                 f=motor,
                 hDict={'encoder': lambda x, u, wh: res*x[0] + wh},
                 n_r=2,
                 n_wf=5,
                 n_whDict={'encoder': 1}) # ^^^ linear encoder model captures actual behavior

# Simulation
for i, ti in enumerate(t[1:]):

    # Chose control and predict next state
    try:
        u = nav.predict(r[i], r[i+1], wf0, Cf, dt)
    except npl.linalg.LinAlgError:
        print("Cholesky failed in predict!")
        break

    # Advance true state using control
    wf = np.random.multivariate_normal(wf0_true, Cf_true)
    x[i+1] = motor(x[i], u, wf, dt)
    x[i+1, 4] = dist[i+1]  # update disturbance

    # When new measurement comes in...
    if i % i_per_z == 0:

        # Get new measurement from real world
        z[i+1] = encoder(x[i+1], 0, 0)

        # Update state estimate
        try:
            nav.correct('encoder', z[i+1], wh0, Ch)
        except npl.linalg.LinAlgError:
            print("Cholesky failed in correct!")
            break

    # ...otherwise hold last measurement (for plotting only)
    else:
        z[i+1] = np.copy(z[i])

    # Record new estimate
    xh[i+1], Cx[i+1] = nav.get_state_and_cov()

# Just checkin...
if not nav.is_pdef(nav.Cx):
    print("WOAH your state estimate covariance is not posdef, how'd that happen?")

#### Plots

fig1 = plt.figure()
fig1.suptitle("Estimation and Tracking via Online UKF-Learned Model", fontsize=22)
ax1 = fig1.add_subplot(5, 1, 1)
ax1.plot(t[:i], x[:i, 0], label="true", color='g', lw=3)
ax1.plot(t[:i], xh[:i, 0], label="estimate", color='k', ls=':', lw=3)
ax1.plot(t[:i], r[:i, 0], label="desired", color='r', ls='--')
ax1.set_xlim([0, ti])
ax1.set_ylabel("position\ndeg", fontsize=12)
ax1.legend(loc='upper right')
ax1.grid(True)

ax1 = fig1.add_subplot(5, 1, 2)
ax1.plot(t[:i], x[:i, 1], label="true", color='g', lw=3)
ax1.plot(t[:i], xh[:i, 1], label="estimate", color='k', ls=':', lw=3)
ax1.plot(t[:i], r[:i, 1], label="desired", color='r', ls='--')
ax1.set_xlim([0, ti])
ax1.set_ylabel("velocity\ndeg/s", fontsize=12)
ax1.grid(True)

ax1 = fig1.add_subplot(5, 1, 3)
ax1.plot(t[:i], x[:i, 2], label="true", color='g', lw=3)
ax1.plot(t[:i], xh[:i, 2], label="estimate", color='k', ls=':', lw=3)
ax1.set_xlim([0, ti])
ax1.set_ylabel("drag/inertia\n(deg/s^2)/(deg/s)", fontsize=12)
ax1.grid(True)

ax1 = fig1.add_subplot(5, 1, 4)
ax1.plot(t[:i], x[:i, 3], label="true", color='g', lw=3)
ax1.plot(t[:i], xh[:i, 3], label="estimate", color='k', ls=':', lw=3)
ax1.set_xlim([0, ti])
ax1.set_ylabel("b/inertia\n(deg/s^2)/PWM", fontsize=12)
ax1.grid(True)

ax1 = fig1.add_subplot(5, 1, 5)
ax1.plot(t[:i], x[:i, 4], label="true", color='g', lw=3)
ax1.plot(t[:i], xh[:i, 4], label="estimate", color='k', ls=':', lw=3)
ax1.set_xlim([0, ti])
ax1.set_ylabel("disturbance\ndeg/s^2", fontsize=12)
ax1.set_xlabel("time\ns", fontsize=12)
ax1.grid(True)

fig2 = plt.figure()
fig2.suptitle("Encoder Measurements", fontsize=22)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.plot(t[:i], z[:i], color='b', lw=2)
ax2.set_xlim([0, ti])
ax2.set_ylabel("ticks", fontsize=16)
ax2.set_xlabel("time\ns", fontsize=16)
ax2.grid(True)

plt.show()
