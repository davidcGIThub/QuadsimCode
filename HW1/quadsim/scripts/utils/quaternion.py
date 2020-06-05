"""Hamilton quaternion class definition module
Quaternion class definition and convenience functions for working on
the S3 manifold as well as for converting to and from different
attitude representations
"""
# system imports
import numpy as np

# local imports
import utils.math_tools as mt


class Quaternion:
    """Hamilton quaternion class
    args: q_0ijk is a numpy.array, list, or tuple of [q0, qx, qy, qz]

    This class uses 1D numpy arrays for vectors, not 2D arrays to
    explicitly denote column vectors.

    Defines box_plus, box_minus, and multiplication of quaternions
    also defines lie algebra -> group -> algebra functions
    """
    def __init__(self, q_0xyz=[1., 0, 0, 0]):
        assert len(q_0xyz) == 4
        self.arr = np.array([*q_0xyz])
        # enforce positive q0
        if self.arr[0] < 0.0:
            self.arr *= -1
        # enforce unit quaternion
        self.normalize()

    def __str__(self):
        return f'[{self.arr[0]}, {self.arr[1]}, {self.arr[2]}, {self.arr[3]}]'

    def __repr__(self):
        s = f'{self.arr[0]} + {self.arr[1]}i + {self.arr[2]}j + ' \
            + f'{self.arr[3]}k'
        return s.replace('+ -', '- ')

    def __len__(self):
        return 4

    def __matmul__(self, other):
        """Perform quaternion multiplication with @"""
        return self.otimes(other)

    def __matmul__(self, other):
        """Perform quaternion multiplication with @"""
        return self.otimes(other)

    @property
    def q0(self):
        """Get scalar portion of quaternion"""
        return self.arr[0]

    @property
    def qx(self):
        """Get x element of qbar"""
        return self.arr[1]

    @property
    def qy(self):
        """Get y element of qbar"""
        return self.arr[2]

    @property
    def qz(self):
        """Get z element of qbar"""
        return self.arr[3]

    @property
    def qbar(self):
        """Get vector portion of quaternion"""
        return self.arr[1:]

    @property
    def elements(self):
        """Get elements of quaternion as a 4d vector"""
        return self.arr

    @property
    def euler(self):
        """Get euler angle representation as a 3d vector
        returns [roll, pitch, yaw]
        """
        q0,qx,qy,qz = self.arr
        roll = np.arctan2(2.0*(q0*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
        pitch = np.arcsin (2.0*(q0*qy - qz*qx))
        yaw = np.arctan2(2.0*(q0*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
        return np.array([roll, pitch, yaw])

    @property
    def R(self):
        """Get passive rotation matrix
        e.g. R_i2b = q_i2b.R
        """
        qbar_skew = mt.skew(self.qbar)
        return np.eye(3) + 2*(-self.q0*qbar_skew + qbar_skew @ qbar_skew)

    @property
    def axis_angle(self):
        """Get axis-angle representation of attitude"""
        scale = 2*np.arccos(self.q0) / np.sqrt(1 - self.q0**2)
        return scale * self.qbar

    @property
    def otimes_mat(self):
        return np.array([
            [self.arr[0], -self.arr[1], -self.arr[2], -self.arr[3]],
            [self.arr[1],  self.arr[0], -self.arr[3],  self.arr[2]],
            [self.arr[2],  self.arr[3],  self.arr[0], -self.arr[1]],
            [self.arr[3], -self.arr[2],  self.arr[1],  self.arr[0]] ])

    def conjugate(self):
        """Get quaternion's conjugate"""
        q_conj = self.arr.copy()
        q_conj[1:] *= -1
        return Quaternion(q_conj)

    def inverse(self):
        """Get quaternion's inverse"""
        return self.conjugate()

    def normalize(self):
        """Makes current quaternion unit length"""
        self.arr /= mt.norm2(self.arr)

    def inv(self):
        """Invert current quaternion"""
        self.arr[1:] *= -1

    def copy(self):
        return Quaternion(self.arr)

    def otimes(self, other):
        """Quaternion multiplication function with positive q0 enforcement"""
        out_arr = self.otimes_mat @ other.arr
        return Quaternion(out_arr)

    def rota(self, vec3):
        """Active rotation of 3d vector (same as q.R.T @ vec3)
        e.g. vec_i = q_i2b.rota(vec_b)
        """
        qbar_skew = mt.skew(self.qbar)
        temp = 2 * qbar_skew @ vec3
        return vec3 + self.q0 * temp + qbar_skew @ temp

    def rotp(self, vec3):
        """Passive rotation of 3d vector (same as q.R @ vec3)
        e.g. vec_b = q_i2b.rotp(vec_i)
        """
        qbar_skew = mt.skew(self.qbar)
        temp = 2 * qbar_skew @ vec3
        return vec3 - self.q0 * temp + qbar_skew @ temp

    def box_minus(self, q2):
        """Returns Log(q2.inverse.otimes(self)) with 3d vector output"""
        q2_inv = q2.inverse()
        q_diff = q2_inv.otimes(self)
        return Log(q_diff)

    def box_plus(self, vec3):
        """Returns self.otimes(Exp(vec3)) with Quaternion type output"""
        return self.otimes(Exp(vec3))

def identity():
    """Return an identity quaternion: [1, 0, 0, 0]"""
    return Quaternion([1.,0,0,0])

def random():
    """Create a random quaternion"""
    rand_arr = np.random.uniform(low=-1, high=1, size=4)
    rand_arr /= mt.norm2(rand_arr)
    return Quaternion(rand_arr)

def Log(quat):
    """Map from Lie Group -> Lie Algebra -> R3
    Log = vee(log(quat))
    """
    assert isinstance(quat, Quaternion)
    qbar_norm = mt.norm2(quat.qbar)
    if qbar_norm < 1e-8:
        vec3 = np.zeros(3)
    else:
        vec3 = 2*np.arctan2(qbar_norm, quat.q0)*quat.qbar/qbar_norm
    return vec3

def Exp(vec3):
    """Map from R3 -> Lie Algebra -> Lie Group (quaternion)
    Exp = exp(hat(vec3))
    """
    assert vec3.size == 3
    angle = mt.norm2(vec3)
    if angle > 1e-6:
        angle_half = angle / 2
        axis = vec3 / angle
        q_arr = np.array([np.cos(angle_half), *(np.sin(angle_half)*axis)])
    else:
        q_arr = np.array([1., *(vec3/2)])
        q_arr /= mt.norm2(q_arr)
    return Quaternion(q_arr)

def from_two_unit_vectors(v1, v2):
    """Create Quaternion from 2 unit vectors"""
    assert v1.size == 3
    assert v2.size == 3
    u = v1.copy()
    v = v2.copy()

    d = u @ v
    if d < 1.0:
        invs = (2 * (1+d)) ** -0.5
        qbar = mt.skew(u) @ v * invs
        q0 = 0.5 / invs
        q_arr = np.array([q0, *qbar])
    return Quaternion(q_arr)

def from_R(R):
    """Create Quaternion from 3x3 rotation matrix"""
    tr = np.trace(R)

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.
        q0 = 0.25 * S
        qx = (R[1, 2] - R[2, 1]) / S
        qy = (R[2, 0] - R[0, 2]) / S
        qz = (R[0, 1] - R[1, 0]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.
        q0 = (R[1, 2] - R[2, 1]) / S
        qx = 0.25 * S
        qy = (R[1, 0] + R[0, 1]) / S
        qz = (R[2, 0] + R[0, 2]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.
        q0 = (R[2, 0] - R[0, 2]) / S
        qx = (R[1, 0] + R[0, 1]) / S
        qy = 0.25 * S
        qz = (R[2, 1] + R[1, 2]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.
        q0 = (R[0, 1] - R[1, 0]) / S
        qx = (R[2, 0] + R[0, 2]) / S
        qy = (R[2, 1] + R[1, 2]) / S
        qz = 0.25 * S
    return Quaternion([q0, qx, qy, qz])

def from_axis_angle(vec3):
    """Create Quaternion from axis-angle representation"""
    return Exp(vec3)

def from_euler(euler):
    """Create Quaternion from Euler angles [roll, pitch, yaw]"""
    cp,ct,cs = np.cos(euler / 2)
    sp,st,ss = np.sin(euler / 2)

    q0 = cp*ct*cs + sp*st*ss
    qx = sp*ct*cs - cp*st*ss
    qy = cp*st*cs + sp*ct*ss
    qz = cp*ct*ss - sp*st*cs

    q_arr = np.array([q0, qx, qy, qz])
    return Quaternion(q_arr)
