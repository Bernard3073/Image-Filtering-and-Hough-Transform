import numpy as np
from scipy import signal    # For signal.gaussian function
import math
from myImageFilter import myImageFilter
import cv2
import matplotlib.pyplot as plt


def grad_angle(imgx, imgy):
    # img_grad = np.zeros_like(imgx)
    grad_mag, grad_dir = cv2.cartToPolar(imgx, imgy)
    # grad_mag = np.sqrt(imgx**2 + imgy**2)
    # grad_mag *= 255/grad_mag.max()
    # grad_dir = np.arctan2(imgy, imgx)
    # plt.imshow(grad_mag, cmap='gray')
    # plt.title('Gradient Magnitude')
    # plt.show()
    return grad_mag, grad_dir

def non_max_suppression(grad_mag, grad_dir):
    img_row, img_col = grad_mag.shape
    output = np.zeros(grad_mag.shape)
    for row in range(1, img_row-1):
        for col in range(1, img_col-1):
            dir = grad_dir[row, col]
            if(0 <= dir < np.pi/8) or (15*np.pi/8 <= dir <= 2*np.pi):
                before_pixel = grad_mag[row, col-1]
                after_pixel = grad_mag[row, col+1]
            elif (np.pi / 8 <= dir < 3 * np.pi / 8) or (9 * np.pi / 8 <= dir < 11 * np.pi / 8):
                before_pixel = grad_mag[row + 1, col - 1]
                after_pixel = grad_mag[row - 1, col + 1]

            elif (3 * np.pi / 8 <= dir < 5 * np.pi / 8) or (11 * np.pi / 8 <= dir < 13 * np.pi / 8):
                before_pixel = grad_mag[row - 1, col]
                after_pixel = grad_mag[row + 1, col]

            else:
                before_pixel = grad_mag[row - 1, col - 1]
                after_pixel = grad_mag[row + 1, col + 1]

            if grad_mag[row, col] >= before_pixel and grad_mag[row, col] >= after_pixel:
                output[row, col] = grad_mag[row, col]
    # plt.imshow(output, cmap='gray')
    # plt.title("Non Max Suppression")
    # plt.show()
    return output

def threshold(image, low, high, weak):
 
    output = np.zeros(image.shape)
 
    strong = 255
 
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))
 
    output[strong_row, strong_col] = strong
    output[weak_row, weak_col] = weak

    # plt.imshow(output, cmap='gray')
    # plt.title("threshold")
    # plt.show()
 
    return output

def hysteresis(image, weak):
    image_row, image_col = image.shape

    top_to_bottom = image.copy()

    for row in range(1, image_row):
        for col in range(1, image_col):
            if top_to_bottom[row, col] == weak:
                if top_to_bottom[row, col + 1] == 255 or top_to_bottom[row, col - 1] == 255 or top_to_bottom[row - 1, col] == 255 or top_to_bottom[
                    row + 1, col] == 255 or top_to_bottom[
                    row - 1, col - 1] == 255 or top_to_bottom[row + 1, col - 1] == 255 or top_to_bottom[row - 1, col + 1] == 255 or top_to_bottom[
                    row + 1, col + 1] == 255:
                    top_to_bottom[row, col] = 255
                else:
                    top_to_bottom[row, col] = 0

    bottom_to_top = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(image_col - 1, 0, -1):
            if bottom_to_top[row, col] == weak:
                if bottom_to_top[row, col + 1] == 255 or bottom_to_top[row, col - 1] == 255 or bottom_to_top[row - 1, col] == 255 or bottom_to_top[
                    row + 1, col] == 255 or bottom_to_top[
                    row - 1, col - 1] == 255 or bottom_to_top[row + 1, col - 1] == 255 or bottom_to_top[row - 1, col + 1] == 255 or bottom_to_top[
                    row + 1, col + 1] == 255:
                    bottom_to_top[row, col] = 255
                else:
                    bottom_to_top[row, col] = 0

    right_to_left = image.copy()

    for row in range(1, image_row):
        for col in range(image_col - 1, 0, -1):
            if right_to_left[row, col] == weak:
                if right_to_left[row, col + 1] == 255 or right_to_left[row, col - 1] == 255 or right_to_left[row - 1, col] == 255 or right_to_left[
                    row + 1, col] == 255 or right_to_left[
                    row - 1, col - 1] == 255 or right_to_left[row + 1, col - 1] == 255 or right_to_left[row - 1, col + 1] == 255 or right_to_left[
                    row + 1, col + 1] == 255:
                    right_to_left[row, col] = 255
                else:
                    right_to_left[row, col] = 0

    left_to_right = image.copy()

    for row in range(image_row - 1, 0, -1):
        for col in range(1, image_col):
            if left_to_right[row, col] == weak:
                if left_to_right[row, col + 1] == 255 or left_to_right[row, col - 1] == 255 or left_to_right[row - 1, col] == 255 or left_to_right[
                    row + 1, col] == 255 or left_to_right[
                    row - 1, col - 1] == 255 or left_to_right[row + 1, col - 1] == 255 or left_to_right[row - 1, col + 1] == 255 or left_to_right[
                    row + 1, col + 1] == 255:
                    left_to_right[row, col] = 255
                else:
                    left_to_right[row, col] = 0

    final_image = top_to_bottom + bottom_to_top + right_to_left + left_to_right

    final_image[final_image > 255] = 255

    return final_image


def myEdgeFilter(img0, sigma):
    hsize = 2 * math.ceil(3 * sigma) + 1
    h_1d = signal.gaussian(hsize, sigma)
    h_2d = np.outer(h_1d, h_1d)
    img0 = myImageFilter(img0, h_2d)
    # Apply sobel in x,y dir
    imgx = cv2.Sobel(img0, cv2.CV_64F, 1, 0, ksize=hsize)
    imgy = cv2.Sobel(img0, cv2.CV_64F, 0, 1, ksize=hsize)
    # cv2.imshow('imgx', imgx)
    # cv2.imshow('imgy', imgy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    img_grad, img_dir = grad_angle(imgx, imgy)
    img = non_max_suppression(img_grad, img_dir)
    # img = threshold(img, 5, 20, 50)
    # img = hysteresis(img, 50)
    # plt.imshow(img, cmap='gray')
    # plt.title("Canny Edge Detector")
    # plt.show()
    return img
