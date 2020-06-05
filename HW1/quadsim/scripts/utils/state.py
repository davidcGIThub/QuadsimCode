# system imports
import numpy as np

# local imports
import utils.quaternion as quat


class ErrorState:
    """Error state class definition and convenience functions for
    working with 12-dof states. ErrorState is a vector space with
    operators defined accordingly.

    pos: position of body expressed in inertial frame
    att: attitude of body (q_i2b)
    vel: velocity of body expressed in body frame
    ang_vel: angular velocity of body expressed in body frame
    """
    def __init__(self, pos=np.zeros(3), att=np.zeros(3),
                 vel=np.zeros(3), ang_vel=np.zeros(3)):
        self.arr = np.array([*pos, *att, *vel, *ang_vel])

    def __str__(self):
        pos = str(self.pos).replace(' ', ', ')
        att = str(self.att).replace(' ', ', ')
        vel = str(self.vel).replace(' ', ', ')
        ang_vel = str(self.ang_vel).replace(' ', ', ')
        return f'pos: {pos}\natt: {att}\nvel: {vel}\nang_vel: {ang_vel}'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return 12

    def __add__(self, other):
        pos = self.pos + other.pos
        att = self.att + other.att
        vel = self.vel + other.vel
        ang_vel = self.ang_vel + other.ang_vel
        return ErrorState(pos, att, vel, ang_vel)

    def __iadd__(self, other):
        self.pos += other.pos
        self.att += other.att
        self.vel += other.vel
        self.ang_vel += other.ang_vel
        return self

    def __sub__(self, other):
        pos = self.pos - other.pos
        att = self.att - other.att
        vel = self.vel - other.vel
        ang_vel = self.ang_vel - other.ang_vel
        return ErrorState(pos, att, vel, ang_vel)

    def __isub__(self, other):
        self.pos -= other.pos
        self.att -= other.att
        self.vel -= other.vel
        self.ang_vel -= other.ang_vel
        return self

    def __mul__(self, scalar):
        """Perform scalar multiplication (*)"""
        pos = self.pos*scalar
        att = self.att*scalar
        vel = self.vel*scalar
        ang_vel = self.ang_vel*scalar
        return ErrorState(pos, att, vel, ang_vel)

    def __rmul__(self, scalar):
        """Perform scalar multiplication (*)"""
        return self.__mul__(scalar)

    def __imul__(self, scalar):
        """Perform scalar multiplication (*=)"""
        self.arr *= scalar
        return self

    # inertial frame position convenience properties
    @property
    def pos(self):
        """Get body position in inertial frame"""
        return self.arr[:3]
    @pos.setter
    def pos(self, new_pos: np.ndarray):
        self.arr[:3] = new_pos

    # inertial to body attitude convenience properties
    @property
    def att(self):
        """Get the attitude of the ErrorState as a 3D vector"""
        return self.arr[3:6]
    @att.setter
    def att(self, new_att: np.ndarray):
        self.arr[3:6] = new_att

    # body frame velocity convenience properties
    @property
    def vel(self):
        """Get body velocity in the body frame"""
        return self.arr[6:9]
    @vel.setter
    def vel(self, new_vel: np.ndarray):
        self.arr[6:9] = new_vel

    # body frame angular velocity convenience properties
    @property
    def ang_vel(self):
        """Get velocity"""
        return self.arr[9:]
    @ang_vel.setter
    def ang_vel(self, new_ang_vel: np.ndarray):
        self.arr[9:] = new_ang_vel

    def copy(self):
        """return copy of State"""
        return ErrorState(self.pos, self.att, self.vel, self.ang_vel)


class State:
    """State class definition and convenience functions for working
    with 12-dof states represented with 13 parameters where attitude
    is represented with a unit quaternion.

    pos: position of body expressed in inertial frame
    att: attitude of body represented with a unit quaternion
    vel: velocity of body expressed in body frame
    ang_vel: angular velocity of body expressed in body frame
    """
    def __init__(self, pos=np.zeros(3), att=np.array([1.,0,0,0]),
                 vel=np.zeros(3), ang_vel=np.zeros(3)):
        self.pos = pos
        self.att = att
        self.vel = vel
        self.ang_vel = ang_vel

    def __str__(self):
        pos = str(self.pos).replace(' ', ', ')
        att = str(self.att)
        vel = str(self.vel).replace(' ', ', ')
        ang_vel = str(self.ang_vel).replace(' ', ', ')
        return f'pos: {pos}\natt: {att}\nvel: {vel}\nang_vel: {ang_vel}'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return 13

    def box_plus(self, error_state: ErrorState):
        """Add vector portions of state & error state and use Exp/[+]
        to concatinate attitude portion. Returns State type.

        e.g. new_state = state [+] error_state
        """
        pos = self.pos + error_state.pos
        att = self.att.box_plus(error_state.att)
        vel = self.vel + error_state.vel
        ang_vel = self.ang_vel + error_state.ang_vel
        return State(pos, att.elements, vel, ang_vel)

    def ibox_plus(self, error_state: ErrorState):
        """Perform box_plus in-place. See box_plus function for details.

        e.g. state [+=] error_state
        """
        self.pos += error_state.pos
        self.att = self.att.box_plus(error_state.att).elements
        self.vel += error_state.vel
        self.ang_vel += error_state.ang_vel

    def box_minus(self, other):
        """Subtract vector portions of 2 states and use Log/[-] to get
        difference in attitude portion.
        Returns ErrorState type.

        e.g. error_state = self - other
        """
        pos = self.pos - other.pos
        att = self.att.box_minus(other.att)
        vel = self.vel - other.vel
        ang_vel = self.ang_vel - other.ang_vel
        return ErrorState(pos, att, vel, ang_vel)

    # inertial frame position convenience properties
    @property
    def pos(self):
        return self._pos
    @pos.setter
    def pos(self, new_pos: np.ndarray):
        self._pos = new_pos.copy()
    @property
    def px(self):
        """Get x position"""
        return self._pos[0]
    @property
    def py(self):
        """Get y position"""
        return self._pos[1]
    @property
    def pz(self):
        """Get z position"""
        return self._pos[2]

    # inertial to body attitude convenience properties
    @property
    def att(self):
        """Get the attitude of the body as a Quaternion"""
        return self._att
    @att.setter
    def att(self, new_att: np.ndarray):
        self._att = quat.Quaternion(new_att)

    # body frame velocity convenience properties
    @property
    def vel(self):
        """Get body velocity in the body frame"""
        return self._vel
    @vel.setter
    def vel(self, new_vel: np.ndarray):
        """Set """
        self._vel = new_vel.copy()
    @property
    def vx(self):
        return self._vel[0]
    @property
    def vy(self):
        return self._vel[1]
    @property
    def vz(self):
        return self._vel[2]

    # body frame angular velocity convenience properties
    @property
    def ang_vel(self):
        """Get angular velocity"""
        return self._ang_vel
    @ang_vel.setter
    def ang_vel(self, new_ang_vel: np.ndarray):
        self._ang_vel = new_ang_vel.copy()
    @property
    def wx(self):
        """Get x angular velocity"""
        return self._ang_vel[0]
    @property
    def wy(self):
        """Get y angular velocity"""
        return self._ang_vel[1]
    @property
    def wz(self):
        """Get z angular velocity"""
        return self._ang_vel[2]

    def copy(self):
        """return copy of State"""
        return State(self.pos, self.att.elements, self.vel, self.ang_vel)
