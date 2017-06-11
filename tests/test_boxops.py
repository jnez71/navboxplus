#!/usr/bin/env python
"""
Unit test for the internal boxops library.
This test validates with orientation_library, available at:
https://github.com/jnez71/orientation_library
which is mostly based on the infamous transformations.py.

"""
from __future__ import division
import numpy as np; npl = np.linalg
from orientation_library import oritools as ori
from navboxplus import boxops

# Test angle operations
for a in np.arange(-100, 100, 1):
    v = 2*np.pi*np.random.sample() - np.pi
    apv = boxops.angle_boxplus(a, v)
    assert apv >= -np.pi and apv <= np.pi
    assert np.isclose(np.cos(apv), np.cos(a+v))
    assert np.isclose(boxops.angle_boxminus(apv, a), v)

# Test quaternion operations
flip_quat = lambda q: np.append(q[3], q[:3])  # oritools uses [w, x, y, z] form
for i in xrange(100):
    q = ori.trns.random_quaternion()
    v = 10*(np.random.sample(3) - 0.5)
    v = v / npl.norm(v) * boxops.angle_unwrap(npl.norm(v))
    qpv = boxops.quaternion_boxplus(q, v)
    ori_qpv = ori.plus(flip_quat(q), ori.quaternion_from_rotvec(v))
    qpvmq = boxops.quaternion_boxminus(qpv, q)
    assert np.allclose(flip_quat(qpv), ori_qpv)
    assert np.isclose(npl.norm(qpv), 1)
    assert np.allclose(boxops.quaternion_boxplus(q, qpvmq), qpv)
    assert np.allclose(qpvmq, v)
    assert np.allclose(qpvmq, ori.error(flip_quat(q), flip_quat(qpv)))
    assert np.allclose(boxops.euler_from_quaternion(q), ori.trns.euler_from_quaternion(flip_quat(q)))

print("SUCCESS: boxops.py passed all tests.\n")
