import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import os 
import cv2
# 1. Load the two temple images and the points from data/some_corresp.npz
dirname = os.path.dirname(__file__)
data = np.load(os.path.join(dirname,'../data/some_corresp.npz'))
pts1_some = data.get("pts1") 
pts2_some = data.get("pts2")
im1 = cv2.imread(os.path.join(dirname, '../data/im1.png'))
im2 = cv2.imread(os.path.join(dirname, '../data/im2.png'))
# 2. Run eight_point to compute F
M = np.max(pts1_some.shape)
F = sub.eight_point(pts1_some, pts2_some, M)
# 3. Load points in image 1 from data/temple_coords.npz
pts1_temple = np.load(os.path.join(dirname,'../data/temple_coords.npz')).get("pts1")
# 4. Run epipolar_correspondences to get points in image 2
pts2_mine = sub.epipolar_correspondences(im1, im2, F, pts1_temple)
# 5. Compute the camera projection matrix P1
K1 = np.load(os.path.join(dirname,'../data/intrinsics.npz')).get("K1")
K2 = np.load(os.path.join(dirname,'../data/intrinsics.npz')).get("K2")
extrinsic1 = np.hstack((np.eye(3), np.zeros((3,1))))
P1 = K1 @ extrinsic1
# 6. Use camera2 to get 4 camera projection matrices P2
E = sub.essential_matrix(F, K1, K2)
P2 = hlp.camera2(E)
# 7. Run triangulate using the projection matrices
# 8. Figure out the correct P2
best_P2 = 0
num_occurences = 0
for i in range(4):
    pts3d = sub.triangulate(P1, pts1_temple, P2[:,:,i], pts2_mine)
    # Multiplying the projection matrix with the 3D points to get the image points
    pts3d_h = np.hstack([pts3d, np.ones([pts3d.shape[0], 1])])
    reprojection1 = (P1@(pts3d_h.T)).T
    reprojection2 = (P2[:,:,i]@(pts3d_h.T)).T
    # Counting the number of points that are in front of the camera, meaning z > 0
    # Reason for multipliyng is that both images dont have 0 reprojections
    occurence = np.sum((reprojection1[:,2]>0)*(reprojection2[:,2]>0))
    if occurence > num_occurences:
        num_occurences = occurence
        best_P2 = i
P2 = P2[:,:,best_P2]
pts3d = sub.triangulate(P1, pts1_temple, P2, pts2_mine)
# 9. Scatter plot the correct 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2])
# Set the axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# Show the plot
plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz
# P = K[R|t]
# extrinsic = [R|t] = inv(K) @ P
extrinsic2 = np.linalg.inv(K2) @ P2
np.savez(os.path.join(dirname,'../data/extrinsics.npz'), R1=extrinsic1[:3,:3], R2=extrinsic2[:3,:3], t1=extrinsic1[:3,3], t2=extrinsic2[:3,3])

 # Project the 3D points back to image 1 to compute the reprojection error
proj1 = (K1 @ P1 @ np.hstack((pts3d, np.ones((pts3d.shape[0], 1)))).T).T
proj1 = proj1[:, :2] / proj1[:, 2:]
error = np.mean(np.linalg.norm(pts1_some - proj1, axis=1))
print('Reprojection error: {}'.format(error))
