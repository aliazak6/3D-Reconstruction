"""
Homework 5
Submission Functions
"""

import numpy as np
from scipy.signal import convolve2d
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

  


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):

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



"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    #F_norm = np.linalg.inv(K2.T) @ F @ np.linalg.inv(K1)
    E = K1.T @ F @ K2
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    pts3d = []
    # Triangulate the points
    for point1 , point2 in zip(pts1, pts2):
        x0, y0 = point1
        x1, y1 = point2
        A = np.vstack([  \
        y0*P1[2] - P1[1],\
        P1[0] - x0*P1[2],\
        y1*P2[2] - P2[1],\
        P2[0] - x1*P2[2]])
        U, S, V = np.linalg.svd(A)
        X = V[-1,:]
        X = X[:-1]/X[-1]
        pts3d.append(X)
    pts3d = np.array(pts3d)
    return pts3d

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
    # Computer optial centers
    c1 = -np.linalg.inv(K1@R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2@R2) @ (K2 @ t2)
    # Compute the new rotation matrix
    r1 = ((c1-c2) / np.linalg.norm(c1-c2))
    r2 = (np.cross(R1[2,:].T,r1))
    r3 = (np.cross(r2,r1))
    R1p = np.array([r1,r2,r3])
    r2 = (np.cross(R2[2,:].T,r1))
    r3 = (np.cross(r2,r1))
    R2p = R1p
    # Compute new instrinsic parameters
    K1p = K2
    K2p = K2
    # Compute new translation vectors
    t1p = -R1p@c1
    t2p = -R2p@c2
    # Compute new rectification matrices
    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p

"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    dispM = np.zeros_like(im1)
    w = (win_size - 1) / 2
    dist = np.zeros((im1.shape[0],im1.shape[1],max_disp+1))
    dist[:,:,:] = np.arange(0, max_disp+1)
    # distance calculated with wanted d value can be accessed with dist[:,:,d] 
    kernel = np.ones((win_size,win_size))
    conv_im1 = convolve2d(im1, kernel, mode='same')
    for d in range(max_disp+1):
        im2_d = np.roll(im2,d, axis=1)
        conv_im2_d = convolve2d(im2_d, kernel, mode='same')
        dist[:,:,d] = (conv_im1 - conv_im2_d)**2
    dispM = np.argmin(dist, axis=2)
    return dispM



"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1 = -np.linalg.inv(K1@R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2@R2) @ (K2 @ t2)
    b = np.linalg.norm(c1 - c2)
    f = K1[1,1]
    # using 1e-9 to avoid division by zero
    depthM = (b *f) / (dispM + 1e-9)
    return depthM


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
