import numpy as np
import copy

def myImageFilter(img0, h):
    # https://github.com/alisaaalehi/convolution_as_multiplication/blob/master/Convolution_as_multiplication.ipynb
    img0_row, img0_col = img0.shape
    h_row, h_col = h.shape
    img1_row = img0_row - h_row + 1
    img1_col = img0_col - h_col + 1
    # zero pad the filter
    h_zero_padded = np.pad((img1_row - h_row, 0),
                            (0, img1_col - h_col),
                            'constant', constant_values=0)
    img1 = np.zeros((img1_row, img1_col))
    for i in range(img1_row):
        for j in range(img1_col):
            img1[i, j] = (img0[i:i+h_row, j:j+h_col]*h).sum()
    return img1
