import numpy as np
import matplotlib.pyplot as plt
import math

def houghLine(image):
    '''
    1. Initialize accumulator array(votes) "H"  to all zeros (width: diagonal dist, height: theta)
    2. For each edge point(x, y) in the image
        For theta = 0 to 180
            rho = x * cos(theta) + y * sin(theta)
            H(theta, rho) = H(theta, rho) + 1
    3. Find the value(s) of (theta, rho) where H(theta, rho) is a local maximum
    4. The detected line in the image is given by
        rho = x * cos(theta) + y * sin(theta)
    '''
    # row, col = image.shape
    # d = int(math.sqrt(row**2 + col**2))
    # H = np.zeros((d+1, 180+1))
    # for i in range(row):
    #     for j in range(col):
    #         for theta in range(0, 181):
    #             theta = np.deg2rad(theta)
    #             rho = j * np.cos(theta) + i * np.sin(theta)
    #             H[int(rho),int(np.rad2deg(theta))] += 1
    #Get image dimensions
    # y for rows and x for columns 
    Ny = image.shape[0]
    Nx = image.shape[1]

    #Max diatance is diagonal one 
    Maxdist = int(np.round(np.sqrt(Nx**2 + Ny ** 2)))
     # Theta in range from -90 to 90 degrees
    thetas = np.deg2rad(np.arange(0, 180))
    #Range of radius
    rs = np.linspace(-Maxdist, Maxdist, 2*Maxdist)
    H = np.zeros((2 * Maxdist, len(thetas)))
    for y in range(Ny):
     for x in range(Nx):
         # Check if it is an edge pixel
         #  NB: y -> rows , x -> columns
          if image[y,x] > 0:
              # Map edge pixel to hough space
              for k in range(len(thetas)):
                   # Calculate space parameter
                    r = x*np.cos(thetas[k]) + y * np.sin(thetas[k])
                    # Update the accumulator
                    # N.B: r has value -max to max
                    # map r to its idx 0 : 2*max
                    H[int(r) + Maxdist,k] += 1
    return H

def main():
    image = np.zeros((50,50))
    image[10, 10] = 1
    accumulator = houghLine(image)
    plt.figure('Original Image')
    plt.imshow(image)
    plt.set_cmap('gray')
    plt.figure('Hough Space')
    plt.imshow(accumulator)
    plt.set_cmap('gray')
    plt.show()
if __name__ == '__main__':
    main()