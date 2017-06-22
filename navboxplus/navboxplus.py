"""
Main class for creating a NavBoxPlus object, which provides a UKF for state and parameter
estimation as well as a general framework to feed those estimates into an adaptive controller.

"""
from __future__ import division
import numpy as np; npl = np.linalg
import scipy.linalg as spl

class NavBoxPlus(object):
    """
    Use the predict and correct functions as measurements come in.
    Note that the predict function will also return the control you should apply.
    Call get_state_and_cov to, you know, get the current state estimate and covariance.

    """
    def __init__(self, x0, Cx0, g, f, hDict, n_r, n_wf, n_whDict,
                 xplus=np.add, xminus=np.subtract, zminusDict=None,
                 alpha=1E-3, beta=2, kappa=1E-6):
        """
                x0: Best guess at the initial full state, (n_x >= n_m)
               Cx0: Initial state estimate covariance matrix, (n_m by n_m)
                 g: Control generating function, u = g(r, rnext, x, Cx, dt)
                 f: Discrete-time state dynamic, xnext = f(xlast, u, wf, dt)
             hDict: Dictionary of sensor model functions, {'sensor_i': zi = hi(x, u, whi)}
               n_r: Number of physical (non-parameter) state values
              n_wf: Number of values in process noise array
          n_whDict: Dictionary like hDict but containing the lengths of the sensor noise arrays {'sensor_i': n_whi}
             xplus: Perturbs a state by a tangent n_m-vector v on the state manifold, x2 = xplus(x1, v)
            xminus: Returns the tangent n_m-vector v that perturbs one state to another, v = xminus(x2, x1)
        zminusDict: Dictionary of measurement boxminus functions (like xminus is for the state), None == np.subtract
             alpha: |
              beta: | Unscented transform parameters (Wikipedia's recommendations)
             kappa: |

        Dimensionalities of all other variables are deduced from these inputs.

        """
        # Initialize state and draw the distinction between physical states and parameters
        self.x = np.atleast_1d(x0).astype(np.float64)
        self.n_x = len(self.x)
        self.n_r = int(n_r)
        self.n_p = self.n_x - self.n_r
        assert self.n_p >= 0

        # Initialize state covariance matrix and infer state-space manifold dimensionality
        self.Cx = np.atleast_2d(Cx0).astype(np.float64)
        self.n_m = len(self.Cx)
        assert self.n_x >= self.n_m

        # Verify that state covariance is positive definite
        self.is_pdef = lambda M: np.all(npl.eigvals(M) > 0)
        assert self.is_pdef(self.Cx)

        # Set controller function and infer control dimensionality
        self.g = g
        _len1d = lambda a: len(np.atleast_1d(a))
        self.n_u = _len1d(self.g(np.random.sample(self.n_r), np.random.sample(self.n_r),
                                 np.random.sample(self.n_x), self.Cx, np.random.sample()))
        self.u = np.zeros(self.n_u)

        # Set and verify process model
        self.f = f
        self.n_wf = int(n_wf)
        assert _len1d(self.f(np.random.sample(self.n_x),
                             np.random.sample(self.n_u),
                             np.random.sample(self.n_wf),
                             np.random.sample())) == self.n_x

        # Store sensor dictionaries and key list
        self.hDict = hDict
        self.hKeys = self.hDict.keys()
        self.n_whDict = {}
        self.n_zDict = {}
        self.zminusDict = {}
        for key in self.hKeys:
            self.n_whDict[key] = int(n_whDict[key])
            self.n_zDict[key] = _len1d(self.hDict[key](np.random.sample(self.n_x),
                                                       np.random.sample(self.n_u),
                                                       np.random.sample(self.n_whDict[key])))
            try: self.zminusDict[key] = zminusDict[key]
            except: self.zminusDict[key] = np.subtract

        # Set and verify boxplus
        self.xplus = xplus
        assert _len1d(self.xplus(np.random.sample(self.n_x), np.random.sample(self.n_m))) == self.n_x

        # Set and verify boxminus
        self.xminus = xminus
        assert _len1d(self.xminus(np.random.sample(self.n_x), np.random.sample(self.n_x))) == self.n_m

        # Process and sensor models wrapped for augmented-state sigma points
        self.sf = lambda s, dt: self.f(s[:self.n_x], self.u, s[self.n_x:], dt)
        self.sh = lambda s, h: h(s[:self.n_x], self.u, s[self.n_x:])

        # Boxoperations wrapped for augmented-state sigma points
        self.sxplus = lambda s, sv: np.append(self.xplus(s[:self.n_x], sv[:self.n_m]),
                                              s[self.n_x:] + sv[self.n_m:])
        self.sxminus = lambda s2, s1: np.append(self.xminus(s2[:self.n_x], s1[:self.n_x]),
                                                s2[self.n_x:] - s1[self.n_x:])

        # Core unscented transform parameters
        self.alpha = np.float64(alpha)
        self.beta = np.float64(beta)
        self.kappa = np.float64(kappa)

        # Memoize derived unscented transform parameters
        self.utpDict = {}
        alpha2 = self.alpha**2
        bp1ma2 = self.beta + 1 - alpha2
        a2m1 = alpha2 - 1
        a2k = alpha2*self.kappa
        for key in ['_f'] + self.hKeys:
            if key == '_f': L = self.n_wf + self.n_m
            else: L = self.n_whDict[key] + self.n_m
            l = L*a2m1 + a2k
            Lpl = L + l
            ldLpl = l/Lpl
            r2Lpl = 1/(2*Lpl)
            lba = ldLpl + bp1ma2
            self.utpDict[key] = [Lpl, ldLpl, r2Lpl, lba]

        # Super sensor dictionary contains all sensor metadata
        self.hDict_full = {}
        for key in self.hKeys:
            self.hDict_full[key] = (self.hDict[key],
                                    self.zminusDict[key],
                                    self.n_zDict[key],
                                    self.n_whDict[key],
                                    self.utpDict[key])


    def predict(self, r, rnext, wf0, Cf, dt):
        """
            r: Desired reference state at time t
        rnext: Desired reference state at time t+dt
          wf0: Mean of the process noise
           Cf: Covariance of the process noise
           dt: Timestep to predict forward

        Progresses the state estimate one dt into the future.
        Returns the control effort u.

        """
        # Compute control action
        self.u = self.g(r, rnext, self.x, self.Cx, dt)

        # Store relevant parameters
        utp = self.utpDict['_f']

        # Compute sigma points, propagate through process, and store tangent-space representation
        M = spl.cholesky(utp[0]*spl.block_diag(self.Cx, Cf))
        s0 = np.append(self.x, wf0)
        fS = [self.sf(s0, dt)]
        fT_sum = np.zeros(self.n_m)
        for Vi in np.vstack((M, -M)):
            fS.append(self.sf(self.sxplus(s0, Vi), dt))
            fT_sum += self.xminus(fS[-1], fS[0])

        # Determine the mean of the propagated sigma points from the tangent vectors
        self.x = self.xplus(fS[0], utp[2]*fT_sum)

        # Determine the covariance from the tangent-space deviations from the mean
        fv0 = self.xminus(fS[0], self.x)
        fPi_sum = np.zeros((self.n_m, self.n_m))
        for fSi in fS[1:]:
            fv = self.xminus(fSi, self.x)
            fPi_sum += np.outer(fv, fv)
        self.Cx = utp[3]*np.outer(fv0, fv0) + utp[2]*fPi_sum

        # Over and out
        return np.copy(self.u)


    def correct(self, sensor, z, wh0, Ch):
        """
        sensor: String of the sensor key name
             z: Value of the measurement
           wh0: Mean of that sensor's noise
            Ch: Covariance of that sensor's noise

        Updates the state estimate with the given sensor measurement.

        """
        # Store relevant functions and parameters
        h, zminus, n_z, _, utp = self.hDict_full[sensor]

        # Compute sigma points and emulate their measurement error
        M = spl.cholesky(utp[0]*spl.block_diag(self.Cx, Ch))
        V = np.vstack((M, -M))
        s0 = np.append(self.x, wh0)
        hE = [zminus(z, self.sh(s0, h))]
        for i, Vi in enumerate(V):
            hE.append(zminus(z, self.sh(self.sxplus(s0, Vi), h)))
        hE = np.array(hE)

        # Determine the mean of the sigma measurement errors
        e0 = utp[1]*hE[0] + utp[2]*np.sum(hE[1:], axis=0)

        # Determine the covariance and cross-covariance
        hV = hE - e0
        hPi_sum = np.zeros((n_z, n_z))
        hPci_sum = np.zeros((self.n_m, n_z))
        for Vqi, hVi in zip(V[:, :self.n_m], hV[1:]):
            hPi_sum += np.outer(hVi, hVi)
            hPci_sum += np.outer(Vqi, hVi)
        Pzz = utp[3]*np.outer(hV[0], hV[0]) + utp[2]*hPi_sum
        Pxz = utp[2]*hPci_sum

        # Kalman update the state estimate and covariance
        K = Pxz.dot(npl.inv(Pzz))
        self.x = self.xplus(self.x, -K.dot(e0))
        self.Cx = self.Cx - K.dot(Pzz).dot(K.T)


    def get_state_and_cov(self):
        """
        Returns the current state estimate and its covariance matrix.

        """
        return np.copy(self.x), np.copy(self.Cx)
