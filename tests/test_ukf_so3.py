#!/usr/bin/env python
"""
Unit test to verify that the UKF in NavBoxPlus has a
reasonable output when operating on a state with a quaternion.

"""
from __future__ import division
import numpy as np; npl = np.linalg
from time import time
from navboxplus import NavBoxPlus
from navboxplus.boxops import quaternion_boxplus, quaternion_boxminus, normalize_l2

# For easy reading of results, suppress scientific notation
np.set_printoptions(suppress=True)

# State is [quaternion_orientation_body_to_world, angular_velocity_in_body]
n_x = 7
n_m = 6

# Process uncertainty resides only in our angular acceleration model
n_wf = 3

# Sensor is ordinary gyroscope
n_z = 3
n_wh = 3

# Typical torque control, but that isn't relevant to this test
n_r = 7
n_u = 3

# Boxplus combines quaternion boxplus and ordinary addition for angular velocity
def xplus(x, v):
    return np.append(quaternion_boxplus(x[:4], v[:3]), x[4:]+v[3:])

# Boxminus combines quaternion boxminus and ordinary subtraction for angular velocity
def xminus(x2, x1):
    return np.append(quaternion_boxminus(x2[:4], x1[:4]), x2[4:]-x1[4:])

# Zero angular acceleration process model, except noise enters through angular acceleration
dt = 0.01
def f(x, u, wf, dt):
    return np.append(quaternion_boxplus(x[:4], x[4:]*dt), x[4:]+wf)

# Additive noise gyroscope with tiny bias
def h(x, u, wh):
    return x[4:] + wh + 0.001

# Unity-gain feedback torque controller, irrelevant to this test
def g(r, rnext, x, Cx, dt):
    return quaternion_boxminus(r[:4], x[:4]) + (r[4:] - x[4:])

# Some initial state that is obvious what will happen after a predict and correct
q = normalize_l2([1, 3, 2, -9])
x = np.append(q, [1, 0, 0])
Cx = np.eye(n_m)

# Simple process noise characteristics
wf0 = np.zeros(n_wf)
Cf = np.eye(n_wf)

# Simple sensor noise characteristics
wh0 = np.zeros(n_wh)
Ch = np.eye(n_wh)

# Create NavBoxPlus object
nav = NavBoxPlus(x0=x,
                 Cx0=Cx,
                 g=g,
                 f=f,
                 hDict={'gyro': h},
                 n_r=n_r,
                 n_wf=n_wf,
                 n_whDict={'gyro': n_wh},
                 xplus=xplus,
                 xminus=xminus)

# Run predict step
start_predict = time()
nav.predict(x, x, wf0, Cf, dt)
predict_duration = time() - start_predict

# Display prediction results
print "Original state: {}".format(np.round(x, 3))
print "Predicted next state: {}".format(np.round(nav.x, 3)), '\n'
print "Original covariance:\n", np.round(Cx, 3)
print "Covariance after prediction:\n", np.round(nav.Cx, 3), '\n'
print "Predict took {} ms.".format(np.round(1000*predict_duration, 3))
print "\n----\n"
assert nav.is_pdef(nav.Cx)

# Correct after receiving some gyro measurement
z = [-2, 10, 0]
start_correct = time()
nav.correct('gyro', z, wh0, Ch)
correct_duration = time() - start_correct

# Display correction results
print "Gyro measurement: {}".format(z)
print "Corrected state: {}".format(np.round(nav.x, 3)), '\n'
print "Covariance after correction:\n", np.round(nav.Cx, 3), '\n'
print "Correct took {} ms.".format(np.round(1000*correct_duration, 3)), '\n'
assert nav.is_pdef(nav.Cx)

print "Look good?\n"
