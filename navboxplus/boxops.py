"""
Convenience library to help you construct your boxplus and boxminus state space operations.
Currently only angle (SO2) and quaternion (SO3) boxoperations are provided here, which you
can combine with ordinary add and subtract when constructing boxoperations for your full state.

Quaternions are represented with the current ROS standard, [x, y, z, w].

Plan to eventually include:
    Rigid-transformations (SE3)
    Invertible matrices (GLn)
    Positive-definite matrices (SPDn)

(SPDn is technically not a Lie group, but it can always be decomposed into a
product of SOn and GLn members over which we can boxoperate and reconstruct).

"""
from __future__ import division
import numpy as np; npl = np.linalg

################################################# ANGLES

def angle_boxplus(a, v):
    """
    Returns the unwrapped angle obtained by adding v to a in radians.

    """
    return angle_unwrap(a + v)

####

def angle_boxminus(a2, a1):
    """
    Returns the unwrapped angle obtained by subtracting a1 from a2.

    """
    return angle_unwrap(a2 - a1)

####

def angle_unwrap(a):
    """
    Shifts an angle to a value from -pi to pi radians.

    """
    return np.mod(a+np.pi, 2*np.pi) - np.pi

################################################# QUATERNIONS

def quaternion_boxplus(q, v):
    """
    Returns the unit quaternion obtained by perturbing q by v.

    """
    return normalize_l2(quaternion_multiply(quaternion_from_rotvec(v), q))

####

def quaternion_boxminus(q2, q1):
    """
    Returns the tangent vector perturbing q1 to q2.
    The magnitude of this tangent vector is always between -pi and pi radians.

    """
    return rotvec_from_quaternion(quaternion_multiply(q2, quaternion_inverse(q1)))

####

def normalize_l2(v):
    """
    Returns v unitized by its L2-norm.

    """
    return np.divide(v, npl.norm(v))

####

def quaternion_multiply(ql, qr):
    """
    Multiplies two quaternions as ql*qr.

    """
    return np.array([ ql[0]*qr[3] + ql[1]*qr[2] - ql[2]*qr[1] + ql[3]*qr[0],
                     -ql[0]*qr[2] + ql[1]*qr[3] + ql[2]*qr[0] + ql[3]*qr[1],
                      ql[0]*qr[1] - ql[1]*qr[0] + ql[2]*qr[3] + ql[3]*qr[2],
                     -ql[0]*qr[0] - ql[1]*qr[1] - ql[2]*qr[2] + ql[3]*qr[3]], dtype=np.float64)

####

def quaternion_inverse(q):
    """
    Inverts a quaternion q.

    """
    qi = np.array(q, dtype=np.float64, copy=True)
    np.negative(qi[:3], out=qi[:3])
    return qi / np.dot(qi, qi)

####

def rotvec_from_quaternion(q):
    """
    Returns a rotation vector corresponding to a quaternion q.

    """
    anorm = npl.norm(q[:3])
    if np.isclose(anorm, 0): return np.zeros(3)
    if q[3] < 0: q = np.negative(q)
    qnorm = npl.norm(q)
    return 2*np.arccos(q[3]/qnorm)*(q[:3]/anorm)

####

def quaternion_from_rotvec(v):
    """
    Returns the quaternion corresponding to a rotation vector v.

    """
    mag = npl.norm(v)
    if np.isclose(mag, 0):
        return np.array([0, 0, 0, 1], dtype=np.float64)
    angle_by_2 = np.mod(mag, 2*np.pi) / 2
    axis = np.divide(v, mag)
    return np.append(np.sin(angle_by_2)*axis, np.cos(angle_by_2))

################################################# MORE TOOLS FOR CONVENIENCE

def euler_from_quaternion(quaternion, axes='sxyz'):
    """
    Return Euler angles from quaternion for specified axis sequence.

    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

####

def quaternion_matrix(quaternion):
    """
    Return homogeneous rotation matrix from quaternion.

    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if np.isclose(nq, 0):
        return np.identity(4)
    q *= np.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array(((1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
                        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
                        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
                        (                0.0,                 0.0,                 0.0, 1.0)))

####

def euler_from_matrix(matrix, axes='sxyz'):
    """
    Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes
    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = np.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if not np.isclose(sy, 0):
            ax = np.arctan2( M[i, j],  M[i, k])
            ay = np.arctan2( sy,       M[i, i])
            az = np.arctan2( M[j, i], -M[k, i])
        else:
            ax = np.arctan2(-M[j, k],  M[j, j])
            ay = np.arctan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = np.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if not np.isclose(cy, 0):
            ax = np.arctan2( M[k, j],  M[k, k])
            ay = np.arctan2(-M[k, i],  cy)
            az = np.arctan2( M[j, i],  M[i, i])
        else:
            ax = np.arctan2(-M[j, k],  M[j, j])
            ay = np.arctan2(-M[k, i],  cy)
            az = 0.0
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

# Map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
               'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
               'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
               'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
               'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
               'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
               'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
               'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())
_NEXT_AXIS = [1, 2, 0, 1]
