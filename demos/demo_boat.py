#!/usr/bin/env python
"""
Using navboxplus to control a marine ship.
Compare this to concurrent-learning approaches.

---------------
Augmented State
---------------
x_world_position   m                 |
y_world_position   m                 |
yaw_from_x         rad               | Rigid-Body States
x_body_velocity    m/s               |
y_body_velocity    m/s               |
yaw_rate           rad/s             |

m-wm_xu            kg                |
m-wm_yv            kg                | Inertial Parameters
m*xg-wm_yr         kg*m              |
Iz-wm_nr           kg*m^2            |

d_xuu              N/(m/s)^2         |
d_yvv              N/(m/s)^2         | Drag Parameters
d_nrr              (N*m)/(rad/s)^2   |

d_yrr              N/(rad/s)^2       |
d_yrv              N/(m*rad/s^2)     |
d_yvr              N/(m*rad/s^2)     | Cross-Flow Parameters
d_nvv              (N*m)/(m/s)**2    |
d_nrv              (N*m)/(m*rad/s^2) |
d_nvr              (N*m)/(m*rad/s^2) |

-------------
Effort Inputs
-------------
x_body_force       N
y_body_force       N
z_torque           N*m

------------
Sensor Suite
------------
Perfect measurement of the non-parameter states,
a typical assumption of adaptive controllers!

"""
from __future__ import division
import numpy as np; npl = np.linalg
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from navboxplus import NavBoxPlus

######################################################################################### BUILD A BOAT

class ShipParams:
    """
    Class to generate a resonable set of ship parameters from just
    total mass, z-moment of inertia, and stern-bow position of COM.
    Also provides an array called vec of the minimal set of unique
    parameters for which the corresponding dynamics are linear in.

    """
    def __init__(self, m, Iz, xg):
        # Boat total mass, rotational inertia and center of gravity
        self.m = np.float64(m)  # kg
        self.Iz = np.float64(Iz)  # kg*m^2
        self.xg = np.float64(xg)  # m

        # Fluid inertial effects
        self.wm_xu = -0.025*m  # kg
        self.wm_yv = -0.25*m  # kg
        self.wm_yr = -0.25*m*xg  # kg*m
        self.wm_nr = -0.25*Iz  # kg*m^2

        # Drag
        self.d_xuu = 0.25 * self.wm_xu  # N/(m/s)^2
        self.d_yvv = 0.25 * self.wm_yv  # N/(m/s)^2
        self.d_nrr = 0.25 * (self.wm_nr + self.wm_yr)  # (N*m)/(rad/s)^2

        # Cross-flow
        self.d_yrr = 0.25 * self.wm_yr  # N/(rad/s)^2
        self.d_yrv = 0.25 * self.wm_yr  # N/(m*rad/s^2)
        self.d_yvr = 0.25 * self.wm_yv  # N/(m*rad/s^2)
        self.d_nvv = 0.25 * self.d_yvv  # (N*m)/(m/s)^2
        self.d_nrv = 0.25 * self.d_yrv  # (N*m)/(m*rad/s^2)
        self.d_nvr = 0.25 * (self.wm_nr + self.wm_yv)  # (N*m)/(m*rad/s^2)

        # Minimal parameter vector
        self.vec = np.array([self.m-self.wm_xu,
                             self.m-self.wm_yv,
                             self.m*self.xg-self.wm_yr,
                             self.Iz-self.wm_nr,
                             self.d_xuu,
                             self.d_yvv,
                             self.d_nrr,
                             self.d_yrr,
                             self.d_yrv,
                             self.d_yvr,
                             self.d_nvv,
                             self.d_nrv,
                             self.d_nvr])

# Our true parameters; in the real world these aren't known
params_true = ShipParams(m=1000, Iz=1500, xg=0.1)

# Our best guess at the true parameters
params = ShipParams(m=500, Iz=800, xg=0)

# Function that unwraps an angle in radians
unwrap = lambda ang: np.mod(ang+np.pi, 2*np.pi) - np.pi

def xplus(x, v):
    """
    Adds two 3DOF states with an arbitrary number of parameters.

    """
    xv = np.add(x, v)
    xv[2] = unwrap(xv[2])
    return xv

def xminus(x2, x1):
    """
    Subtracts two 3DOF states with an arbitrary number of parameters.

    """
    e = np.subtract(x2, x1)
    e[2] = unwrap(e[2])
    return e

def dyn_mats(x):
    """
    Returns the inertial matrix, centripetal matrix, and drag matrix
    which influence the ship dynamics.

    """
    # Inertial matrix
    M = np.array([[x[6],    0,    0],
                  [   0, x[7], x[8]],
                  [   0, x[8], x[9]]])

    # Centripetal-coriolis matrix
    C = np.array([[                    0,          0, -x[8]*x[5] - x[7]*x[4]],
                  [                    0,          0,              x[6]*x[3]],
                  [x[8]*x[5] + x[7]*x[4], -x[6]*x[3],                      0]])

    # Drag matrix
    D = np.array([[-x[10]*abs(x[3]),                                    0,                                    0],
                  [               0, -(x[11]*abs(x[4]) + x[14]*abs(x[5])), -(x[15]*abs(x[4]) + x[13]*abs(x[5]))],
                  [               0, -(x[16]*abs(x[4]) + x[17]*abs(x[5])), -(x[18]*abs(x[4]) + x[12]*abs(x[5]))]])

    return M, C, D

def dynamics(x, u, wf, dt):
    """
    Discrete-time marine ship dynamics.
    Remember x is fully augmented with the minimal parameters.

    """
    # Inertial, centripetal, and drag matrices
    M, C, D = dyn_mats(x)

    # Rotation matrix (orientation, converts body to world)
    sy = np.sin(x[2]); cy = np.cos(x[2])
    R = np.array([[cy, -sy, 0],
                  [sy,  cy, 0],
                  [ 0,   0, 1]])

    # M*vdot + C*v + D*v = u  and  pdot = R*v
    xdot = np.concatenate((R.dot(x[3:6]), npl.inv(M).dot(u - (C + D).dot(x[3:6])), np.zeros(13)))
    return xplus(x + xdot*dt, wf)

######################################################################################### CONTROL DESIGN

# Controller configuration
gains_p = 10*np.array([1000, 1000, 3000])  # [N/m, N/m, (N*m)/rad]
gains_d = 10*np.array([1000, 1000, 3000])  # [N/(m/s), N/(m/s), (N*m)/(rad/s)]
feedback = None; feedforward = None  # for externally recording these quantities

def controller(r, rnext, x, Cx, dt):
    """
    Controller with feedback and feedforward based on estimated model.

    """
    # For externally recording these quantities
    global feedback, feedforward

    # World to body frame rotation matrix
    sy = np.sin(x[2]); cy = np.cos(x[2])
    Rinv = np.array([[ cy, sy, 0],
                     [-sy, cy, 0],
                     [  0,  0, 1]])

    # PD feedback
    error = xminus(r, x[:6])
    feedback = gains_p*(Rinv.dot(error[:3])) + gains_d*(error[3:6])

    # Solve dynamic for feedforward
    M, C, D = dyn_mats(np.append(r, x[6:]))
    rdot = xminus(rnext, r) / dt
    feedforward = M.dot(rdot[3:]) + (C + D).dot(r[3:])

    # Classic
    return feedback + feedforward

def sensor(x, u, wh):
    """
    Algebraic sensor model.
    Perfect measurement of the non-parameter states,
    a typical assumption of adaptive controllers!

    """
    return x[:6] + wh

######################################################################################### GUIDANCE

def circle(t):
    """
    Trajectory generator for random circular paths.

    """
    global p_tgen, t_tgen
    if p_tgen is None or t_tgen > 2*new_tgen:
        t_tgen = 0
        p_tgen = [20, 40]*np.random.sample(2)
    a, p = p_tgen
    w = 2*np.pi/p
    s = np.sin(w*t); c = np.cos(w*t)
    h = unwrap(np.pi/2 + np.arctan2(s, c))
    t_tgen += dt
    return np.array([a*c, a*s, h, w*a, 0, w])

def lemni(t):
    """
    Trajectory generator for random figure-8 paths.

    """
    global p_tgen, t_tgen
    if p_tgen is None or t_tgen > 2*new_tgen:
        t_tgen = 0
        p_tgen = [20, 20]*np.random.sample(2)
    a, p = p_tgen
    w = 2*np.pi/p
    sq2 = np.sqrt(2)
    ri = np.zeros(n_r)
    s = np.sin(w*t); c = np.cos(w*t)
    ri[:2] = a*sq2*c/(s**2+1) - (a+1), a*sq2*c*s/(s**2+1)
    v = [(a*sq2*w*s*(s**2 - 3))/(s**2 + 1)**2, (a*sq2*w*(3*np.cos(2*w*t) - 1))/(2*(s**2 + 1)**2)]
    ri[2] = np.arctan2(v[1], v[0])
    sr = np.sin(ri[2]); cr = np.cos(ri[2])
    ri[3:] = cr*v[0] + sr*v[1], -sr*v[0] + cr*v[1], -(3*w*c)/(c**2 - 2)
    t_tgen += dt
    return ri

# Simulation time domain (also chooses predict frequency)
T = 30  # s
dt = 0.01  # s
t = np.arange(0, T, dt)  # s

# Choose trajectory generator
tgen = lemni
new_tgen = T/3  # s
p_tgen = [10, 20]  # [amplitude, period]
t_tgen = 0

######################################################################################### SIM AND NAV SETUP

# Some common dimensionalities
n_r = 6
n_p = 13
n_x = n_r + n_p
n_u = 3
n_z = 6

# True noise characteristics
wf0_true = np.zeros(n_x)
Cf_true = np.zeros((n_x, n_x))
wh0_true = np.zeros(n_z)
Ch_true = np.zeros((n_z, n_z))

# Our guesses at the noise characteristics
# We cannot express any perfect confidence
wf0 = np.zeros(n_x)
Cf = np.diag(np.append(1E-10*np.ones(n_r), 1E-4*np.ones(n_p)))
wh0 = np.zeros(n_z)
Ch = 1E-10*np.eye(n_z)

# State, estimate, covariance, reference, and effort timeseries
x_true = np.zeros((len(t), n_x))
x = np.zeros((len(t), n_x))
Cx = np.zeros((len(t), n_x, n_x))
r = np.zeros((len(t), n_r))
u = np.zeros((len(t), n_u))
uff = np.zeros((len(t), n_u))

# Initial conditions
x_true[0] = np.append(tgen(0), params_true.vec)
x[0] = np.append(tgen(0), params.vec)
Cx[0] = np.diag(np.append(1E-10*np.ones(n_r), 50*np.ones(n_p)))

# Configure navboxplus
nav = NavBoxPlus(x0=np.copy(x[0]),
                 Cx0=np.copy(Cx[0]),
                 g=controller,
                 f=dynamics,
                 hDict={'sensor': sensor},
                 n_r=n_r,
                 n_wf=n_x,
                 n_whDict={'sensor': n_r},
                 plimits=None)

######################################################################################### SIMULATION

# Simulate
for i, ti in enumerate(t[1:]):

    # Generate desired reference state
    r[i] = tgen(ti)
    r[i+1] = tgen(ti+dt)

    # Chose control and predict next state
    try:
        u[i+1] = nav.predict(r[i], r[i+1], wf0, Cf, dt)
        uff[i+1] = feedforward
    except npl.linalg.LinAlgError:
        print("Cholesky failed in predict!")
        break

    # Advance true state using control
    wf_true = np.random.multivariate_normal(wf0_true, Cf_true)
    x_true[i+1] = dynamics(x_true[i], u[i+1], wf_true, dt)

    # Get new measurement from real world
    wh_true = np.random.multivariate_normal(wh0_true, Ch_true)
    z = sensor(x_true[i+1], u[i+1], wh_true)

    # Update state estimate
    try:
        nav.correct('sensor', z, wh0, Ch)
    except npl.linalg.LinAlgError:
        print("Cholesky failed in correct!")
        break

    # Record new estimate
    x[i+1], Cx[i+1] = nav.get_state_and_cov()
end = i
print("Final parameter RMS error: {}".format(np.sqrt(npl.norm(x_true[-1, n_r:] - x[-1, n_r:]))))

# Just checkin...
if not nav.is_pdef(nav.Cx):
    print("WOAH your state estimate covariance is not posdef, how'd that happen?\n")

######################################################################################### PLOTS

# Figure for individual results
fig1 = plt.figure()
fig1.suptitle('State Evolution', fontsize=20)
fig1rows = 2
fig1cols = 4

# Plot x position
ax = fig1.add_subplot(fig1rows, fig1cols, 1)
ax.set_title('X Position (m)', fontsize=16)
ax.plot(t[:end], x[:end, 0], 'k',
        t[:end], r[:end, 0], 'g--')
ax.set_xlim([0, t[end]])
ax.grid(True)

# Plot y position
ax = fig1.add_subplot(fig1rows, fig1cols, 2)
ax.set_title('Y Position (m)', fontsize=16)
ax.plot(t[:end], x[:end, 1], 'k',
        t[:end], r[:end, 1], 'g--')
ax.set_xlim([0, t[end]])
ax.grid(True)

# Plot yaw position
ax = fig1.add_subplot(fig1rows, fig1cols, 3)
ax.set_title('Heading (deg)', fontsize=16)
ax.plot(t[:end], np.rad2deg(x[:end, 2]), 'k',
        t[:end], np.rad2deg(r[:end, 2]), 'g--')
ax.set_xlim([0, t[end]])
ax.grid(True)

# Plot control efforts
ax = fig1.add_subplot(fig1rows, fig1cols, 4)
ax.set_title('Wrench (N, N, N*m)', fontsize=16)
ax.plot(t[:end], u[:end, 0], 'b', label="Fx")
ax.plot(t[:end], u[:end, 1], 'g', label="Fy")
ax.plot(t[:end], u[:end, 2], 'r', label="Mz")
ax.legend(loc='upper right')
ax.plot(t[:end], uff[:end, 0], 'b--',
        t[:end], uff[:end, 1], 'g--',
        t[:end], uff[:end, 2], 'r--')
ax.set_xlim([0, t[end]])
ax.grid(True)

# Plot x velocity
ax = fig1.add_subplot(fig1rows, fig1cols, 5)
ax.set_title('Surge (m/s)', fontsize=16)
ax.plot(t[:end], x[:end, 3], 'k',
        t[:end], r[:end, 3], 'g--')
ax.set_xlim([0, t[end]])
ax.set_xlabel('Time (s)')
ax.grid(True)

# Plot y velocity
ax = fig1.add_subplot(fig1rows, fig1cols, 6)
ax.set_title('Sway (m/s)', fontsize=16)
ax.plot(t[:end], x[:end, 4], 'k',
        t[:end], r[:end, 4], 'g--')
ax.set_xlim([0, t[end]])
ax.set_xlabel('Time (s)')
ax.grid(True)

# Plot yaw velocity
ax = fig1.add_subplot(fig1rows, fig1cols, 7)
ax.set_title('Yaw Rate (deg/s)', fontsize=16)
ax.plot(t[:end], np.rad2deg(x[:end, 5]), 'k',
        t[:end], np.rad2deg(r[:end, 5]), 'g--')
ax.set_xlim([0, t[end]])
ax.set_xlabel('Time (s)')
ax.grid(True)

# Plot parameter estimates
ax = fig1.add_subplot(fig1rows, fig1cols, 8)
ax.set_title('Parameter Estimates', fontsize=16)
colors=plt.cm.rainbow(np.linspace(0, 1, n_p))
for i in xrange(n_p):
    ax.plot(t[:end], x[:end, n_r+i], c=colors[i], ls='-')
    ax.plot(t[:end], x_true[:end, n_r+i], c=colors[i], ls='--')
ax.set_xlim([0, t[end]])
ax.set_xlabel('Time (s)')
ax.grid(True)

# Create plot for in-depth look at parameters
pnames = ["m-wm_xu", "m-wm_yv", "m*xg-wm_yr", "Iz-wm_nr",
          "d_xuu", "d_yvv", "d_nrr", "d_yrr", "d_yrv",
          "d_yvr", "d_nvv", "d_nrv", "d_nvr"]
fig2 = plt.figure()
fig2.suptitle('Parameter Estimation', fontsize=20)
ax = fig2.add_subplot(2, 1, 1)
ax.set_ylabel('Estimates', fontsize=16)
for i in xrange(n_p):
    ax.plot(t[:end], x[:end, n_r+i], c=colors[i], ls='-')
    ax.plot(t[:end], x_true[:end, n_r+i], c=colors[i], ls='--')
ax.set_xlim([0, t[end]])
ax.grid(True)
ax = fig2.add_subplot(2, 1, 2)
ax.set_ylabel('Errors', fontsize=16)
perror = x_true[:end, n_r:] - x[:end, n_r:]
for i in xrange(n_p):
    ax.plot(t[:end], perror[:end, i], c=colors[i], label=pnames[i])
ax.set_xlim([0, t[end]])
ax.set_xlabel('Time (s)', fontsize=16)
ax.legend(loc='upper right')
ax.grid(True)

# Plot physical trajectory
fig3 = plt.figure()
fig3.suptitle('Pose Space', fontsize=20)
ax = fig3.add_subplot(1, 1, 1)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.plot(x_true[:end, 0], x_true[:end, 1], 'k', r[:, 0], r[:, 1], 'g--')
ax.scatter(x_true[0, 0], x_true[0, 1], color='r', s=50)
skip = int(0.75/dt); qscale = 0.25
ax.quiver(x[::skip, 0], x[::skip, 1],
          qscale*np.cos(x[::skip, 2]), qscale*np.sin(x[::skip, 2]), width=0.003, color='k')
ax.quiver(r[::skip, 0], r[::skip, 1],
          qscale*np.cos(r[::skip, 2]), qscale*np.sin(r[::skip, 2]), width=0.003, color='g')
ax.set_aspect('equal', 'datalim')
ax.grid(True)

# Points and lines for representing positions and headings
pthick = 80
lthick = 3
llen = 1
p = ax.scatter(x[0, 0], x[0, 1], color='k', s=pthick)
h = ax.plot([x[0, 0], x[0, 0] + llen*np.cos(x[0, 2])],
            [x[0, 1], x[0, 1] + llen*np.sin(x[0, 2])], color='k', linewidth=lthick)
pref = ax.scatter(r[0, 0], r[0, 1], color='b', s=pthick)
href = ax.plot([r[0, 0], r[0, 0] + llen*np.cos(r[0, 2])],
               [r[0, 1], r[0, 1] + llen*np.sin(r[0, 2])], color='b', linewidth=lthick)

# Function for updating the animation frame
def update_ani(arg, ii=[0]):
    i = ii[0]  # don't ask...
    if np.isclose(t[i], np.around(t[i], 1)):
        fig3.suptitle('Evolution (Time: {})'.format(t[i]), fontsize=24)
    p.set_offsets((x[i, 0], x[i, 1]))
    h[0].set_data([x[i, 0], x[i, 0] + llen*np.cos(x[i, 2])],
                  [x[i, 1], x[i, 1] + llen*np.sin(x[i, 2])])
    pref.set_offsets((r[i, 0], r[i, 1]))
    href[0].set_data([r[i, 0], r[i, 0] + llen*np.cos(r[i, 2])],
                     [r[i, 1], r[i, 1] + llen*np.sin(r[i, 2])])
    ii[0] += int(1 / (dt * framerate))
    if ii[0] >= end:
        print("Resetting animation!")
        ii[0] = 0
    return [p, h, pref, href]

# Run animation
framerate = min((20, 1/dt))
ani = animation.FuncAnimation(fig3, func=update_ani, interval=dt*1000)

plt.show()