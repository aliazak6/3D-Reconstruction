"""
Homework 5
Submission Functions
"""

import numpy as np

"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
       Function: eight_point
        Description: Computes the fundamental matrix from corresponding points using the 8-point algorithm.
"""
def eight_point(pts1, pts2, M):
    from helper import refineF
    #Transform points to homogenous coordinates
    h_pts1 = np.hstack((pts1, np.ones((pts1.shape[0],1)))).astype(np.uint16)
    h_pts2 = np.hstack((pts2, np.ones((pts2.shape[0],1)))).astype(np.uint16)
    
    # Scale the points
    T = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    h_pts1 = (T @ h_pts1.T).T # 3x3 * 3xN = 3xN -> N x 3
    h_pts2 = (T @ h_pts2.T).T 

    # Construct A matrix
    N = h_pts1.shape[0]
    A = np.zeros((N,9))
    for i in range(N):
        A[i] = [h_pts1[i][0]*h_pts2[i][0], h_pts1[i][0]*h_pts2[i][1], h_pts1[i][0],\
                h_pts1[i][1]*h_pts2[i][0], h_pts2[i][1]*h_pts1[i][1], h_pts2[i][1],\
                h_pts2[i][0],h_pts2[i][1], 1]
    # Compute F
    u, s, vh = np.linalg.svd(A)
    F = vh[np.argmin(s)].reshape(3,3)

    # Rank 2 constraint on F
    u, s, vh = np.linalg.svd(F)
    s[-1] = 0
    F = np.dot(u*s, vh)

    # Refine F
    F = refineF(F, h_pts1[:,:-1], h_pts2[:,:-1])
    
    # Unscale F
    F = T.T @ F @ T
    
    return F


def closest_point_on_line(line, point, im):
    """
    Find the closest point on a line to a given point
    """
    x0, y0 = point[0], point[1]
    im_x = np.array(range(640))
    a, b, c = line[0], line[1], line[2]
    # find all y values for the line
    epi_y = (-a/b)*im_x - c/b
    # find the closest line point on the given point
    points = list(zip(im_x,epi_y))
    #Calculate euclidean distance between the point and all points on the line
    point_2 = points[np.linalg.norm(points - point[:2], axis=1).argmin()]
    return point_2
    


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):

    # window oluşturmak lazım
    window_size = 10
    # Calculate epipolar lines
    h_pts1 = np.hstack((pts1, np.ones((pts1.shape[0],1)))).astype(np.uint16)
    epipolar_lines = (F @ h_pts1.T).T # Nx3

    # Find the closest point on the epipolar line
    pts2 = np.zeros((pts1.shape[0],2))
    im_x = np.array(range(640))
    points_average = []
    for i in range(pts1.shape[0]):
        # Find the closest point on the epipolar line
        line = epipolar_lines[i]
        x0, y0 = pts1[i][0], pts1[i][1]
        # Calculate the window coordinates
        xmin, ymin = max(x0 - window_size,0), max(y0 - window_size,0)
        xmax, ymax = min(xmin + 2*window_size + 1,im1.shape[1]),min(ymin + 2*window_size + 1,im1.shape[0])
        a, b, c = line[0], line[1], line[2]
        # find all y values for the line
        epi_y = (-a/b)*im_x - c/b
        # find the closest line point on the given point
        points = np.array(list(zip(im_x,epi_y))).astype(np.uint16)

        # At this point in the algorithm, we have points variable with points in epipolar line
        # We have x0 and y0 -> original points in im1
        # We need to average x0,y0 in im1 with a window. Do same in im2 with points. Minimum difference should be our pts2[i]

        im1_window = im1[ymin:ymax, xmin:xmax]
        points_distance = []

        for point in points:
            # Create a window around the point
            x2min, y2min = max(point[0] - window_size,0), max(point[1] - window_size,0)
            x2max, y2max = min(x2min + 2*window_size + 1,im2.shape[1]),min(y2min + 2*window_size + 1,im2.shape[0])
            if x2max == im2.shape[1]:
                x2min = x2max -(2*window_size + 1)
            if y2max == im2.shape[0]:
                y2min = y2max -(2*window_size + 1)
            im2_window = im2[y2min:y2max, x2min:x2max]
            dist = np.sqrt(np.sum((im1_window - im2_window)**2))
            points_distance.append(dist)
        pts2[i] = points[np.argmin(points_distance)]
    return pts2

if __name__ == '__main__':
    import os
    import cv2
    from helper import displayEpipolarF
    from helper import epipolarMatchGUI
    # Test your code here
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, '../data/some_corresp.npz')
    data = np.load(data_path)
    pts1 = data.get("pts1") 
    pts2 = data.get("pts2")
    M = np.max(pts1.shape)
    F = eight_point(pts1, pts2, M)
    
    im1_path = os.path.join(dirname, '../data/im1.png')
    im2_path = os.path.join(dirname, '../data/im2.png')
    im1 = cv2.imread(im1_path)
    im2 = cv2.imread(im2_path)
    #displayEpipolarF(im1,im2, F)
    epipolarMatchGUI(im1, im2, F)

"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    pass


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation
    pass


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
