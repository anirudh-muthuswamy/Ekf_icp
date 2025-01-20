import numpy as np
import matplotlib.pyplot as plt

def ComputeOptimalRigidRegistration(X, Y, corr):
    #Calculate Centroids
    X_corr = np.array([X[i] for i, _ in corr])
    Y_corr = np.array([Y[j] for _, j in corr])

    centroid_x = np.mean(X_corr, axis=0)
    centroid_y = np.mean(Y_corr, axis=0)

    #Center the point Clouds
    X_centered = X_corr - centroid_x
    Y_centered = Y_corr - centroid_y

    #calculate cross variance matrix W 
    W =  X_centered.T @ Y_centered

    #Compute SVD of W
    U, _, Vt = np.linalg.svd(W)

    #Compute optimal rotation R
    R_pred = Vt.T @ U.T

    #Compute optimal translation t
    t_pred = centroid_y - R_pred @ centroid_x

    return t_pred, R_pred

def EstimateCorrespondences(X, Y, t_0, R_0, d_max):

    t = t_0
    R = R_0

    for i in range(num_iter):
        #initialize correspondences
        corr = []

        for i in range(num_pts):

            x_i = X[i]
            transformed_x_i = R @ x_i + t
            
            #euclidean distance 
            distances = np.linalg.norm((Y - transformed_x_i), axis = 1)

            #find closest point in Y
            j = np.argmin(distances)
            min_dist = distances[j] 

            if min_dist < d_max:
                corr.append((i, j))

        t_new, R_new = ComputeOptimalRigidRegistration(X, Y, corr)

        t = t_new
        R = R_new

    return t, R

if __name__ == "__main__":

    X = np.loadtxt('pclX.txt')
    Y = np.loadtxt('pclY.txt')

    num_pts = X.shape[0]

    #initial guess for optimal registration
    t_0 = 0.0
    R_0 = np.eye(3)

    #maximum admissible distance
    d_max =0.25
    num_iter = 30

    t, R = EstimateCorrespondences(X, Y, t_0, R_0, d_max)

    X_transformed = (R @ X.T).T + t

    #rmse between Y and X_transformed
    rmse = np.sqrt(np.mean((Y - X_transformed)**2))

    print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
    print("t: \n", t, "\nR: \n", R)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot point cloud Y
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], color='r', label='Target (Y)')

    # Plot transformed point cloud X
    ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], color='b', label='Transformed (X)')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()

    plt.show()