import numpy as np
import cv2
from numpy.core.numeric import indices  # For cv2.dilate function

def find_local_maxima(H, neighbor_size):
    # size Neighborhood size (3=>3*3)
    H_copy = np.copy(H)
    ssize = int((neighbor_size-1)/2)
    peaks = np.zeros(H_copy.shape)
    h, w = H_copy.shape
    for y in range(ssize, h-ssize):
        for x in range(ssize, w-ssize):
            val = H_copy[y, x]
            if val > 0:
                neighborhood = np.copy(H_copy[y-ssize:y+ssize+1, x-ssize:x+ssize+1])
                neighborhood[ssize, ssize] = 0
                if val > np.max(neighborhood):
                    peaks[y, x] = val
    return peaks

def myHoughLines(H, nLines):
    # YOUR CODE HERE
    # loop through number of peaks to identify
    rhos = []
    thetas = []
    neighbor_size = 5
    peaks = find_local_maxima(H, neighbor_size)
    indices = peaks.ravel().argsort()[-nLines:]
    indices = (np.unravel_index(i, peaks.shape) for i in indices)
    # peaks_idx = [(peaks[i], i) for i in indices]
    for i in indices:
        rhos.append(i[0])
        thetas.append(i[1])
    #############################################
    # H_copy = np.copy(H)
    # for n in range(nLines):
    #     # find argmax in flattened array
    #     idx = np.argmax(H_copy)
    #     # remap to shape of H
    #     # https://stackoverflow.com/questions/48135736/what-is-an-intuitive-explanation-of-np-unravel-index
    #     H_copy_idx = np.unravel_index(idx, H_copy.shape)
    #     # Surpress indicies in neighborhood
    #     # first separate x, y indexes from argmax(H)
    #     idx_y, idx_x = H_copy_idx
    #     rhos.append(idx_y)
    #     thetas.append(idx_x)
    #     # if idx_x is too close to the edges choose appropriate values
    #     min_x = 0 if idx_x - (neighbor_size/2) < 0 else idx_x - (neighbor_size/2)
    #     max_x = H.shape[1] if idx_x + (neighbor_size/2) + 1 > H.shape[1] else idx_x + (neighbor_size/2) + 1
    #     # if idx_y is too close to the edges choose appropriate values
    #     min_y = 0 if idx_y - (neighbor_size/2) < 0 else idx_y - (neighbor_size/2)
    #     max_y = H.shape[0] if idx_y + (neighbor_size/2) + 1 > H.shape[0] else idx_y + (neighbor_size/2) + 1
    #     for x in range(int(min_x), int(max_x)):
    #         for y in range(int(min_y), int(max_y)):
    #             # remove neighbor in H_copy
    #             H_copy[y, x] = 0

    return rhos, thetas