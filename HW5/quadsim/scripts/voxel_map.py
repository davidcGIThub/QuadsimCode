import numpy as np

class VoxelMap():
    def __init__(self, voxels_per_side = 25, width_on_a_side = 100, initial_prob = 0.5,
         probDetection = 0.85, probFalseAlarm = 0.3, probStatic = 0.9 , probTransient = 0.2):
        try:
            if voxels_per_side % 2 == 0:
                raise EvenInputVoxelsError
            else:
                self.voxels_per_side = voxels_per_side
        except EvenInputVoxelsError:
            self.voxels_per_side = 25
        self.N = voxels_per_side 
        self.W = width_on_a_side 
        self.radius = width_on_a_side/2
        self.initial_prob = initial_prob
        self.voxel_width = self.W/self.N
        self.scale = self.N/self.W
        self.offset = (self.N//2)
        self.offset_vec = self.offset*np.array([1,1,1])
        self.occ_meas = np.zeros([self.N,self.N,self.N])
        self.free_meas = np.zeros([self.N,self.N,self.N])
        self.prob_map = self.initial_prob*np.ones([self.N,self.N,self.N])
        self.residual = np.array([0.0,0.0,0.0])
        self.pD = probDetection
        self.pFA = probFalseAlarm
        self.ps = probStatic
        self.pt = probTransient

    def set_params(self, pD = None, pFA = None, ps = None, pt = None):
        if pD is not None:
            self.pD = pD
        if pFA is not None:
            self.pFA = pFA
        if ps is not None:
            self.ps = ps
        if pt is not None:
            self.pt = pt

    def reset_map(self, initial_prob = None):
        if initial_prob is not None:
            self.initial_prob = initial_prob
        self.prob_map = self.initial_prob*np.ones([self.N,self.N,self.N])

    def update_map(self, measurements = None):
        self.prob_map = self.ps*self.prob_map+self.pt*(1-self.prob_map)
        if measurements is not None:
            measurements = (self.scale)*measurements
            abs_vals = abs(measurements)
            x_maj = (abs_vals[:,0]>=abs_vals[:,1])&(abs_vals[:,0]>=abs_vals[:,2])
            y_maj = (abs_vals[:,1]>=abs_vals[:,2])&(~x_maj)
            z_maj = (~x_maj)&(~y_maj)
            x_proj = measurements[x_maj]
            y_proj = measurements[y_maj]
            z_proj = measurements[z_maj]
            for it in range(self.offset):
                x_in = x_proj[abs(x_proj[:,0])>it]
                ax = abs(x_in[:,0])
                x_in = it*np.array([np.sign(x_in[:,0]),x_in[:,1]/ax,x_in[:,2]/ax]).T
                x_indices = np.rint(x_in).astype(int) + self.offset_vec
                self.free_meas[x_indices[:,0],x_indices[:,1],x_indices[:,2]] += 1
                y_in = y_proj[abs(y_proj[:,1])>it]
                ay = abs(y_in[:,1])
                y_in = it*np.array([y_in[:,0]/ay,np.sign(y_in[:,1]),y_in[:,2]/ay]).T
                y_indices = np.rint(y_in).astype(int) + self.offset_vec
                self.free_meas[y_indices[:,0],y_indices[:,1],y_indices[:,2]] += 1
                z_in = z_proj[abs(z_proj[:,2])>it]
                az = abs(z_in[:,2])
                z_in = it*np.array([z_in[:,0]/az,z_in[:,1]/az,np.sign(z_in[:,2])]).T
                z_indices = np.rint(z_in).astype(int) + self.offset_vec
                self.free_meas[z_indices[:,0],z_indices[:,1],z_indices[:,2]] += 1
            while np.any(self.free_meas > 0):
                (coords_x,coords_y,coords_z) = np.nonzero(self.free_meas)
                self.prob_map[coords_x,coords_y,coords_z] = ((1-self.pD)*self.prob_map[coords_x,coords_y,coords_z])/(
                                                            (1-self.pD)*self.prob_map[coords_x,coords_y,coords_z] + 
                                                            (1-self.pFA)*(1-self.prob_map[coords_x,coords_y,coords_z]))
                self.free_meas[coords_x,coords_y,coords_z] -= 1
            measurements = measurements[(abs(measurements[:,0])<self.radius)&
                                        (abs(measurements[:,1])<self.radius)&
                                        (abs(measurements[:,2])<self.radius)]
            occ_indices = np.rint(measurements).astype(int) + self.offset_vec
            self.occ_meas[occ_indices[:,0],occ_indices[:,1],occ_indices[:,2]] += 1
            while np.any(self.occ_meas > 0):
                (coords_x,coords_y,coords_z) = np.nonzero(self.occ_meas)
                self.prob_map[coords_x,coords_y,coords_z] = (self.pD*self.prob_map[coords_x,coords_y,coords_z])/(
                                                             self.pD*self.prob_map[coords_x,coords_y,coords_z] + 
                                                             (self.pFA)*(1-self.prob_map[coords_x,coords_y,coords_z]))
                self.occ_meas[coords_x,coords_y,coords_z] -= 1

    def shift_map(self, shift):
        shift = (self.scale)*shift + self.residual
        int_shift = np.rint(shift).astype(int)
        self.residual = shift - int_shift
        if int_shift[0] > 0:
            self.prob_map = np.append(self.initial_prob*np.ones((int_shift[0],self.N,self.N)),
                                      self.prob_map[:-int_shift[0],:,:], 
                                      axis=0)
        elif int_shift[0] < 0:
            self.prob_map = np.append(self.prob_map[-int_shift[0]:,:,:],
                                      self.initial_prob*np.ones((-int_shift[0],self.N,self.N)), 
                                      axis=0)
        if int_shift[1] > 0:
            self.prob_map = np.append(self.initial_prob*np.ones((self.N,int_shift[1],self.N)),
                                      self.prob_map[:,:-int_shift[1],:], 
                                      axis=1)
        elif int_shift[1] < 0:
            self.prob_map = np.append(self.prob_map[:,-int_shift[1]:,:],
                                      self.initial_prob*np.ones((self.N,-int_shift[1],self.N)), 
                                      axis=1)
        if int_shift[2] > 0:
            self.prob_map = np.append(self.initial_prob*np.ones((self.N,self.N,int_shift[2])),
                                      self.prob_map[:,:,:-int_shift[2]], 
                                      axis=2)
        elif int_shift[2] < 0:
            self.prob_map = np.append(self.prob_map[:,:,-int_shift[2]:],
                                      self.initial_prob*np.ones((self.N,self.N,-int_shift[2])), 
                                      axis=2)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        (coords_x,coords_y,coords_z) = np.nonzero(self.prob_map)
        vals = self.prob_map[coords_x,coords_y,coords_z].ravel()
        out_string = ""
        for i in range(len(vals)):
            out_string += "at ({0},{1},{2}) val={3}\n".format(str(coords_x[i]), 
                                                              str(coords_y[i]), 
                                                              str(coords_z[i]), 
                                                              str(vals[i]))
        return out_string

class EvenInputVoxelsError(Exception):
    """Exception to be raised when voxels per side input is even"""
    
    def __str__(self):
        return f'The number of voxels per side must be odd: Using default value'
