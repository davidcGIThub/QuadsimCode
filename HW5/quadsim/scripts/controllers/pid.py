
class ScalarPID:
    """A PID controller. kp, ki, and kd are the gains, sigma is used
    to take dirty derivatives, Ts is the approximate time step between
    each run, and flag is used to specify which term to differentiate:

    if flag:
        u_d = kd*error_dot
    else:
        u_d = -kd*y_dot
    """
    def __init__(self, kp=1.0, kd=0.0, ki=0.0, sigma=0.05, Ts=0.002, flag=True):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.Ts = Ts
        denom = 2*sigma + Ts
        self.a1 = (2*sigma - Ts) / denom
        self.a2 = 2 / denom
        self.error_prev = 0.0
        self.value_prev = 0.0
        self.integrator = 0.0
        self.differentiator = 0.0
        self.flag = flag

    def run(self, ref, y):
        error = ref - y
        u = self.kp*error

        if not self.ki == 0.0:
            self.integrate(error)
            u += self.ki*self.integrator

        if not self.kd == 0.0:
            if self.flag:
                self.differentiate(error)
            else:
                self.differentiate(y)
            u += self.kd*self.differentiator

        self.error_prev = error

        return u

    def integrate(self, error):
        self.integrator += self.Ts*0.5*(error + self.error_prev)

    def differentiate(self, value):
        self.differentiator *= self.a1
        self.differentiator += self.a2*(value - self.value_prev)
        self.value_prev = value

class VectorPID:
    def __init__(self):
        pass
