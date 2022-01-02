import numpy as np

def myHoughTransform(image, rhoRes, thetaRes):
    # YOUR CODE HERE
    #Get image dimensions
    # y for rows and x for columns 
    Ny = image.shape[0]
    Nx = image.shape[1]

    # diagonal distance of the image 
    diag_dist = np.ceil(np.round(np.sqrt(Nx**2 + Ny**2)))
    # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(-90, 90, thetaRes))
    #Range of radius
    rhos = np.arange(-diag_dist, diag_dist+1, rhoRes)
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    # finad all edge(nonzero) pixel indexes
    y_idx, x_idx = np.nonzero(image)
    # Cycle through edge points
    for i in range(len(x_idx)):
        x = x_idx[i]
        y = y_idx[i]
        for j in range(len(thetas)):
            rho = int(x*np.cos(thetas[j]) + y*np.sin(thetas[j]))
            H[rho + int(diag_dist), j] += 1
    
    return H, rhos, thetas
