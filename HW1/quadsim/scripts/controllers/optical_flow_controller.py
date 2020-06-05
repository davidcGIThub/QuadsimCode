# system imports
import numpy as np

class OpticalFlowController:
    """Optical flow controller class that produces velocity commands from feature pairs."""
    def __init__(self, follow_option, wall_option, k_terrain, Dv_terrain, k_wall, Dv_wall, k_canyon):
        self.velocity_command = np.zeros((3,1))
        self.init = True
        self.img_shape = np.zeros((0,0)) # x pixels, y pixels
        self.ROI_bounds = np.array([[0,0,0,0], # mid-left (xmin, xmax, ymin, ymax)
                                    [0,0,0,0], # mid-right
                                    [0,0,0,0], # far-left
                                    [0,0,0,0], # far-right
                                    [0,0,0,0], # bottom
                                    [0,0,0,0]]) # top
        self.feat1 = np.empty((2,1))  # numpy array of features from previous time step
        self.feat2 = np.empty((2,1))  # numpy array of features from current time step
        self.feat1_cal = np.empty((3,1))  # numpy array of calibrated features from previous time step
        self.feat2_cal = np.empty((3,1))  # numpy array of calibrated features from current time step
        self.flow = np.empty((3,1))  # optical flow vector
        self.ind_ml = np.array([])  # mid-left logical indices
        self.ind_mr = np.array([])  # mid-right logical indices
        self.ind_fl = np.array([])  # far-left logical indices
        self.ind_fr = np.array([])  # far-right logical indices
        self.ind_b = np.array([])  # bottom logical indices
        self.ind_t = np.array([])  # top logical indices
        self.follow_option = follow_option  # optical flow command option: 0=collision_avoidance 1=terrain_following 2=wall_following 3=canyon_following
        self.wall_option = wall_option  # flag to identify whether to follow wall on right or left: 0=left 1=right
        self.k_terrain = k_terrain
        self.Dv_terrain = Dv_terrain
        self.k_wall = k_wall
        self.Dv_wall = Dv_wall
        self.k_canyon = k_canyon

    def process_optical_flow(self, img_data, Kc):
        if self.init:
            self.extract_img_data(img_data)
            self.init = False
        self.Ts = img_data.dt.to_sec()
        # self.segment_features(img_data.pairs, img_data.feature_count)
        self.segment_features(img_data.features_prev, img_data.features_next)
        self.calibrate_features(Kc)
        self.flow = (self.feat2_cal - self.feat1_cal)/self.Ts  # calculate flow
        self.calc_time_to_collision()
        if self.follow_option == 0:
            v1 = self.collision_avoidance()
            v2 = self.terrain_follow()
            v = v1 + v2
        elif self.follow_option == 1:
            v = self.terrain_follow()
        elif self.follow_option == 2:
            v1 = self.wall_follow()
            v2 = self.terrain_follow()
            v = v1 + v2
        elif self.follow_option == 3:
            v1 = self.canyon_follow()
            v2 = self.terrain_follow()
            v = v1 + v2
        else:
            v = np.array((0.0, 0.0, 0.0))

        self.velocity_command = v

    def extract_img_data(self, img_data):
        self.img_shape = np.array((img_data.img_width, img_data.img_height))
        self.ROI_bounds = np.array([[int((1/4.)*self.img_shape.item(0)), int((1/2.)*self.img_shape.item(0)), int((1/4.)*self.img_shape.item(1)), int((3/4.)*self.img_shape.item(1))], # mid-left (xmin, xmax, ymin, ymax)
                                    [int((1/2.)*self.img_shape.item(0)), int((3/4.)*self.img_shape.item(0)), int((1/4.)*self.img_shape.item(1)), int((3/4.)*self.img_shape.item(1))], # mid-right
                                    [0, int((1/4.)*self.img_shape.item(0)), int((1/4.)*self.img_shape.item(1)), int((3/4.)*self.img_shape.item(1))], # far-left
                                    [int((3/4.)*self.img_shape.item(0)), self.img_shape.item(0), int((1/4.)*self.img_shape.item(1)), int((3/4.)*self.img_shape.item(1))], # far-right
                                    [int((1/4.)*self.img_shape.item(0)), int((3/4.)*self.img_shape.item(0)), int((3/4.)*self.img_shape.item(1)), self.img_shape.item(1)], # bottom
                                    [int((1/4.)*self.img_shape.item(0)), int((3/4.)*self.img_shape.item(0)), 0, int((1/4.)*self.img_shape.item(1))]]) # top

    # def segment_features(self, pairs, count):
    def segment_features(self, features_prev, features_next):
        feat1 = np.array(features_prev).reshape(-1,2).T
        feat2 = np.array(features_next).reshape(-1,2).T
        self.ind_ml = np.logical_and(feat2[0] >= self.ROI_bounds[0,0], np.logical_and(feat2[0] <= self.ROI_bounds[0,1], np.logical_and(feat2[1] >= self.ROI_bounds[0,2], feat2[1] <= self.ROI_bounds[0,3])))
        self.ind_mr = np.logical_and(feat2[0] >= self.ROI_bounds[1,0], np.logical_and(feat2[0] <= self.ROI_bounds[1,1], np.logical_and(feat2[1] >= self.ROI_bounds[1,2], feat2[1] <= self.ROI_bounds[1,3])))
        self.ind_fl = np.logical_and(feat2[0] >= self.ROI_bounds[2,0], np.logical_and(feat2[0] <= self.ROI_bounds[2,1], np.logical_and(feat2[1] >= self.ROI_bounds[2,2], feat2[1] <= self.ROI_bounds[2,3])))
        self.ind_fr = np.logical_and(feat2[0] >= self.ROI_bounds[3,0], np.logical_and(feat2[0] <= self.ROI_bounds[3,1], np.logical_and(feat2[1] >= self.ROI_bounds[3,2], feat2[1] <= self.ROI_bounds[3,3])))
        self.ind_b = np.logical_and(feat2[0] >= self.ROI_bounds[4,0], np.logical_and(feat2[0] <= self.ROI_bounds[4,1], np.logical_and(feat2[1] >= self.ROI_bounds[4,2], feat2[1] <= self.ROI_bounds[4,3])))
        self.ind_t = np.logical_and(feat2[0] >= self.ROI_bounds[5,0], np.logical_and(feat2[0] <= self.ROI_bounds[5,1], np.logical_and(feat2[1] >= self.ROI_bounds[5,2], feat2[1] <= self.ROI_bounds[5,3])))
        self.feat1 = feat1
        self.feat2 = feat2

    def calibrate_features(self, Kc):
        feat1c = np.vstack((self.feat1, np.ones((1,len(self.feat1[0])))))
        feat2c = np.vstack((self.feat2, np.ones((1,len(self.feat2[0])))))
        feat1_cal = np.linalg.inv(Kc) @ feat1c
        feat2_cal = np.linalg.inv(Kc) @ feat2c
        self.feat1_cal = feat1_cal
        self.feat2_cal = feat2_cal

    def calc_time_to_collision(self):
        self.tau = np.divide(self.feat2_cal[0,:], self.flow[0,:])
        self.tau[np.isnan(self.tau)] = np.inf
        self.tau[self.tau <= 0.0] = np.inf

    def collision_avoidance(self): # TODO: NOT FUNCTIONAL YET
        # # avoid impending collisions
        tau_thresh = 3.0
        vs = 1.0
        vc = 0.0
        tau_min_r = np.min(self.tau[self.ind_mr])
        tau_min_l = np.min(self.tau[self.ind_ml])
        if tau_min_r < tau_min_l and tau_min_r < tau_thresh:
            vc = -vs
        if tau_min_l < tau_min_r and tau_min_l < tau_thresh:
            vc = vs
        return np.array((0,vc,0))

    def terrain_follow(self):
        # # follow terrain floor
        k = self.k_terrain
        Dv_des = self.Dv_terrain
        e = self.feat2_cal[1, self.ind_b]
        e[e == np.inf] = 0.0
        e_dot = self.flow[1, self.ind_b]
        e_dot[e_dot == 0.0] = np.inf
        Dv_ave = np.sum(e/(e_dot*np.sqrt(np.power(e,2)+1.0)))
        vc = -k*(Dv_des - Dv_ave)
        if np.abs(vc) == np.inf:
            vc = 0
        return np.array((0,0,vc))
        
    def wall_follow(self):
        k = self.k_wall
        Dv_des = self.Dv_wall
        
        # # follow wall on left
        if self.wall_option == 0:
            e = self.feat2_cal[0, self.ind_fl]
            e[e == np.inf] = 0.0
            e_dot = self.flow[0, self.ind_fl]
            e_dot[e_dot == np.inf] = 0.0
            Dv_ave = np.sum(e/(e_dot*np.sqrt(np.power(e,2)+1.0)))
            vc = k*(Dv_des - Dv_ave)
            if np.abs(vc) == np.inf:
                vc = 0
        
        # # follow wall on right
        if self.wall_option == 1:
            e = self.feat2_cal[0, self.ind_fr]
            e[e == np.inf] = 0.0
            e_dot = self.flow[0, self.ind_fr]
            e_dot[e_dot == 0.0] = np.inf
            Dv_ave = np.sum(e/(e_dot*np.sqrt(np.power(e,2)+1.0)))
            vc = -k*(Dv_des - Dv_ave)
            if np.abs(vc) == np.inf:
                vc = 0

        return np.array((0,vc,0))
        
    def canyon_follow(self):
        # # follow canyon features
        k = self.k_canyon
        e_dot_r = self.flow[0, self.ind_fr]
        e_dot_l = self.flow[0, self.ind_fl]
        e_dot_r[np.abs(e_dot_r) == np.inf] = 0.0
        e_dot_l[np.abs(e_dot_l) == np.inf] = 0.0
        e_dot_l_ave = np.sum(e_dot_l)/len(e_dot_l)
        e_dot_r_ave = np.sum(e_dot_r)/len(e_dot_r)
        vc = k*((np.linalg.norm(e_dot_l_ave) - np.linalg.norm(e_dot_r_ave))/(np.linalg.norm(e_dot_l_ave) + np.linalg.norm(e_dot_r_ave)))
        if np.abs(vc) == np.inf:
            vc = 0
        return np.array((0,vc,0))