# system imports
import numpy as np
import cv2 as cv


class FeatureTracker:
    """Image processing class that finds trackable features from images."""
    def __init__(self, max_corners=500, quality=0.1, min_dist=10.0):
        print('[FeatureTracker] OpenCV version:', cv.__version__)
        print('[FeatureTracker] CV DIR:', cv.__file__)
        self.window_title = 'Good Features to Track'
        self.track_opts = [max_corners, quality, min_dist]
        self.features = None
        self.features_prev = None
        self.img = np.empty(0)
        self.gray_img = np.empty(0)
        self.gray_img_prev = np.empty(0)
        self.features_paired = np.empty(0)
        self.init = True

    def save_trackable_features(self, img, show=False):
        self.img = img
        self.gray_img = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        self.features = cv.goodFeaturesToTrack(self.gray_img, *self.track_opts)
        nextPts=None
        if not self.init:
            p1, stat, err = cv.calcOpticalFlowPyrLK(self.gray_img_prev, self.gray_img, self.features_prev, nextPts)
            self.features_paired = np.hstack((self.features_prev, p1))
            self.features_paired = self.features_paired[np.logical_and(stat.flatten() == 1, err.flatten() <= 50.0), :,:]
        else:
            self.init = False

        if show and isinstance(self.features, np.ndarray):
            ## overlay circles on output image features
            # for feature in self.features:
            #     cv.circle(self.img, tuple(feature[0]),2,(0,0,255),-1)

            ## overlay arrows on output image showing optical flow
            for i in range(len(self.features_paired)):
                cv.arrowedLine(self.img, tuple(self.features_paired[i,0,:]), tuple(self.features_paired[i,1,:]), (0, 0, 255))
            cv.imshow(self.window_title, self.img)
            cv.waitKey(1)
        
        # age data
        self.gray_img_prev = self.gray_img
        self.features_prev = self.features
