import numpy as np
import cv2
from numpy.core.numeric import indices  # For cv2.dilate function

def myHoughLines(H, nLines):
    # YOUR CODE HERE
    # loop through number of peaks to identify
    rhos = []
    thetas = []
    H_copy = np.copy(H)
    neighbor_size = 3
    for n in range(nLines):
        # find argmax in flattened array
        idx = np.argmax(H_copy)
        # remap to shape of H
        # https://stackoverflow.com/questions/48135736/what-is-an-intuitive-explanation-of-np-unravel-index
        H_copy_idx = np.unravel_index(idx, H_copy.shape)
        # Surpress indicies in neighborhood
        # first separate x, y indexes from argmax(H)
        idx_y, idx_x = H_copy_idx
        rhos.append(idx_y)
        thetas.append(idx_x)
        # if idx_x is too close to the edges choose appropriate values
        min_x = 0 if idx_x - (neighbor_size/2) < 0 else idx_x - (neighbor_size/2)
        max_x = H.shape[1] if idx_x + (neighbor_size/2) + 1 > H.shape[1] else idx_x + (neighbor_size/2) + 1
        # if idx_y is too close to the edges choose appropriate values
        min_y = 0 if idx_y - (neighbor_size/2) < 0 else idx_y - (neighbor_size/2)
        max_y = H.shape[0] if idx_y + (neighbor_size/2) + 1 > H.shape[0] else idx_y + (neighbor_size/2) + 1
        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                # remove neighbor in H_copy
                H_copy[y, x] = 0

    return rhos, thetas