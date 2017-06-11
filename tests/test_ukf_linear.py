#!/usr/bin/env python
"""
Unit test to verify that the UKF in NavBoxPlus matches
the output of a normal KF for a linear problem.

"""
from __future__ import division
import numpy as np; npl = np.linalg
from pprint import pprint
from navboxplus import NavBoxPlus

# Verification tolerance
rtol = 1E-5
atol = 1E-6

for test_iter in xrange(1000):
    # print test_iter

    # State and additive process noise dimensionalities
    n_x = 30
    n_wf = n_x

    # Measurement and additive sensor noise dimensionalities
    n_z = 8
    n_wh = n_z

    # Controllable-state and control dimensionalities (irrelevant to this test)
    n_r = n_x
    n_u = n_x

    # Timestep and random, linear, discrete process model
    dt = 1
    A = np.round(np.random.sample((n_x, n_x)) - 0.5, 1)
    # print A, '\n'
    def f(x, u, wf, dt):
        return A.dot(x) + wf

    # Random, linear sensor model
    H = np.round(20*np.random.sample((n_z, n_x)) - 0.5, 0)
    # print H, '\n'
    def h(x, u, wh):
        return H.dot(x) + wh

    # For generating random covariance matrices
    cs = 5
    Cx_seed = cs*(np.random.sample((n_x, n_x))-0.5)
    Cf_seed = cs*(np.random.sample((n_x, n_x))-0.5)
    Ch_seed = cs*(np.random.sample((n_z, n_z))-0.5)

    # Random initial state and covariance
    x = 30*(np.random.sample(n_x)-0.5)
    Cx = Cx_seed.dot(Cx_seed.T) + 2*cs*np.eye(n_x)

    # Random mean and covariance for process noise
    wf0 = 3*(np.random.sample(n_wf)-0.5)
    Cf = Cf_seed.dot(Cf_seed.T) + 2*cs*np.eye(n_x)

    # Random mean and covariance for sensor noise
    wh0 = 3*(np.random.sample(n_wh)-0.5)
    Ch = Ch_seed.dot(Ch_seed.T) + 2*cs*np.eye(n_z)

    # Create NavBoxPlus object (g is just a dummy function)
    nav = NavBoxPlus(x0=x,
                     Cx0=Cx,
                     g=lambda r, rnext, x, Cx, dt: r-x,
                     f=f,
                     hDict={'sen': h},
                     n_r=n_r,
                     n_wf=n_wf,
                     n_whDict={'sen': n_wh})

    # # Display all attributes to verify initialization
    # pprint (vars(nav))

    # Predict step of KF and UKF
    x2 = A.dot(x) + wf0
    Cx2 = A.dot(Cx).dot(A.T) + Cf
    nav.predict(x, x, wf0, Cf, dt)

    # print x2
    # print nav.x, '\n'
    # print Cx2
    # print nav.Cx, '\n---'

    # Are they the same?
    if not np.allclose(x2, nav.x, rtol=rtol, atol=atol):
        print x2
        print nav.x
        print "\nLargest deviation: {}\n".format(np.max(np.abs(x2 - nav.x)))
        raise ValueError
    if not np.allclose(Cx2, nav.Cx, rtol=rtol, atol=atol):
        print Cx2
        print nav.Cx
        print "\nLargest deviation: {}\n".format(np.max(np.abs(Cx2 - nav.Cx)))
        raise ValueError

    # Get a random measurement and update estimates accordingly with KF and UKF
    z = 10*(np.random.sample(n_z)-0.5)
    e = z - (H.dot(x2) + wh0)
    S = H.dot(Cx2).dot(H.T) + Ch
    K = Cx2.dot(H.T).dot(npl.inv(S))
    x3 = x2 + K.dot(e)
    Cx3 = Cx2 - K.dot(H).dot(Cx2)
    nav.correct('sen', z, wh0, Ch)

    # print x3
    # print nav.x, '\n'
    # print Cx3
    # print nav.Cx, '\n---'

    # Are they the same?
    if not np.allclose(x3, nav.x, rtol=rtol, atol=atol):
        print x3
        print nav.x
        print "\nLargest deviation: {}\n".format(np.max(np.abs(x3 - nav.x)))
        raise ValueError
    if not np.allclose(Cx3, nav.Cx, rtol=rtol, atol=atol):
        print Cx3
        print nav.Cx
        print "\nLargest deviation: {}\n".format(np.max(np.abs(Cx3 - nav.Cx)))
        raise ValueError

print "SUCCESS: NavBoxPlus passed linear KF verification test.\n"
