# system imports
import numpy as np
import utils.math_tools as mt
import cv2 as cv


def calculate_calibration_matrix(
    horizontal_fov_deg,
    image_size
):
    """
    Calculate the calibration matrix, assuming no skew.
    horizontal_fov_deg should be in degrees,
    image_size is [width, height]
    """
    f = 0.5 * image_size[0] \
        / np.tan(np.radians(horizontal_fov_deg) / 2.)
    Kc = np.array([
        [f, 0., image_size[0] / 2.],
        [0., f, image_size[1] / 2.],
        [0., 0., 1.]
    ])
    return Kc


def triangulate_points(prev_pts, next_pts, translation, rotation, method, K_inv):
    """
    Triangulate a set of unnormalized image feature pairs

    prev_pts and next_pts: (2xn) matrices, where n is the number of points

    rotation: quaternion from prev to next

    translation: translation from prev to next, expressed in next
    We assume the translation is known to scale, so the
    returned points will also be to scale

    K_inv: inverse of camera matrix

    method: 0 = naive
            1 = Sampson approx
            2 = full analytical solution

    methods 1 and 2 call method 0 after point updates

    Returns a set of points in the next frame as a 3xn array
    """
    if prev_pts.shape[1] != next_pts.shape[1]:
        print("[Point Triangulation] Feature lists not the same size!")
        return None

    n = prev_pts.shape[1]

    # get normalized image coordinates
    x_a = K_inv @ np.block([[prev_pts],[np.ones(n)]])
    x_b = K_inv @ np.block([[next_pts],[np.ones(n)]])

    if method == 0:
        pass
    elif method == 1:
        x_a, x_b = sampson_update(x_a, x_b, translation, rotation)
    elif method == 2:
        x_a, x_b = analytical_update(x_a, x_b, translation, rotation)
    else:
        print("[Point Triangulation] Method not recognized!")
        return None

    return naive_triangulate_points(x_a, x_b, translation, rotation)


def naive_triangulate_points(x_a, x_b, translation, rotation):
    """
    Assumes points satisfy the epipolar constraint. Uses the eigenvector
    corresponding the to smallest eigenvalue of A.T @ A as depth solution
    """
    n = x_a.shape[1]

    R = rotation.R
    # R = rotation

    # create A matrix
    x_a_rot = R @ x_a
    A = np.zeros((3*n, n+1))
    for i, x in enumerate(x_b.T):
        x_b_cross = mt.skew(x)
        A[3*i:3*(i+1), i] = x_b_cross @ x_a_rot[:,i]
        A[3*i:3*(i+1), -1] = x_b_cross @ translation

    # get Lambda by taking the eigenvector corresponding the smallest
    # singular value of A
    u, s, vh = np.linalg.svd(A)

    # normalize so that gamma = 1
    Lambda = vh[-1,:]/vh[-1,-1]

    # multiply to get 3d points in previous frame
    p_a = x_a * Lambda[:-1]

    p_a = p_a[:, np.bitwise_and(Lambda[:-1] < 20.0, Lambda[:-1] > 1.0)]

    # get points in new body frame
    p_b = R @ p_a + translation.reshape(3,1)

    return p_b


def sampson_update(x_a, x_b, translation, rotation):
    raise NotImplementedError


def analytical_update(x_a, x_b, translation, rotation):
    """
    Computes the analytic solution to the point match correction problem
    (see Section 12.5 of Multiple View Geometry book)
    """
    n = x_a.shape[1]
    x_a = x_a[:2,:].reshape(1,n,2)
    x_b = x_b[:2,:].reshape(1,n,2)

    x_a_cor, x_b_cor = cv.correctMatches(mt.skew(translation) @ rotation.R,
                            x_a, x_b)

    x_a_cor = x_a_cor[0,:,:].T
    x_b_cor = x_b_cor[0,:,:].T
    x_a_cor = np.block([[x_a_cor],[np.ones(n)]])
    x_b_cor = np.block([[x_b_cor],[np.ones(n)]])

    return x_a_cor, x_b_cor
