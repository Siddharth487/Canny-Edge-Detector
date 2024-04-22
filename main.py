import os
import cv2
import numpy as np

# Loading the image
load_image = os.path.join('C:\\Users\\Siddharth\\Desktop\\2B\\Classes\\MTE 203\\203 project 2\\stewie.jpg')
image = cv2.imread(load_image)
big_image = cv2.resize(image, (int(image.shape[1]*3), int(image.shape[0]*3)))
grayscale_image = cv2.cvtColor(big_image, cv2.COLOR_BGR2GRAY)

def conv2D(image, kernel):
    # Getting the dimensions of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculating the convolution dimensions
    conv_height = image_height - kernel_height + 1
    conv_width = image_width - kernel_width + 1

    result = np.zeros((conv_height, conv_width)) # Initializing the result (convolved) image

    # Iterating over the image
    for i in range(conv_height):
        for j in range(conv_width):
            region = image[i:i + kernel_height, j:j + kernel_width] # Extracting the region of interest from the image
            result[i, j] = np.sum(region * kernel) # discrete multiplication
    return result

def gauss_blur(image, sigma):
    # Creating a Gaussian kernel
    kernel_size = 3
    kernel = np.fromfunction(
        lambda x, y:  np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)),
        (kernel_size, kernel_size)
    )

    blurred_image = conv2D(image, kernel)   # Convolving the image with the Gaussian kernel
    blurred_image = blurred_image / np.sum(blurred_image)  # Normalizing the kernel
    return blurred_image

def sobel_filter(image):
    # Defining a 5x5 Sobel kernel for x and y directions
    kernel_x = np.array([[-1, -2, 0, 2, 1], [-4, -8, 0, 8, 4], [-6, -12, 0, 12, 6], [-4, -8, 0, 8, 4], [-1, -2, 0, 2, 1]])
    kernel_y = np.array([[-1, -4, -6, -4, -1],  [-2, -8, -12, -8, -2], [0, 0, 0, 0, 0], [2, 8, 12, 8, 2],[1, 4, 6, 4, 1]])

    print(kernel_x)
    print(kernel_y)
    grad_x = conv2D(image, kernel_x) # Convolving the image with Sobel kernels
    grad_y = conv2D(image, kernel_y)
    grad_magnitude = np.hypot(np.abs(grad_x), np.abs(grad_y))  # Taking the magnitude of the horizontal and vertical kernels
    grad_direction = np.arctan2(grad_y, grad_x) * (180 / np.pi)
    return grad_x, grad_y, grad_magnitude, grad_direction

def nonmax_suppression(gradient_magnitude, gradient_direction, high_ratio, low_ratio):
    suppressed = np.zeros_like(gradient_magnitude)
    thresh = np.zeros_like(gradient_magnitude)
    high_threshold = gradient_magnitude.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    for i in range(1, gradient_magnitude.shape[0] - 1):
        for j in range(1, gradient_magnitude.shape[1] - 1):
            direction = gradient_direction[i, j]

            # Determining direction of neighbouring indices using the rounded gradient direction (angle)
            if (0 <= direction < 22.5) or (157.5 <= direction <= 180):
                next = [(i, j+1), (i, j-1)]
            elif 22.5 <= direction < 67.5:
                next = [(i-1, j+1), (i+1, j-1)]
            elif 67.5 <= direction < 112.5:
                next = [(i-1, j), (i+1, j)]
            elif 112.5 <= direction < 157.5:
                next = [(i-1, j-1), (i+1, j+1)]

            if gradient_magnitude[i,j] >= gradient_magnitude[next[0]] and gradient_magnitude[i,j] >= gradient_magnitude[next[1]]:
                suppressed[i,j] = gradient_magnitude[i,j]
            # Comparing the gradient magnitude with neighbouring indicies
            mag = gradient_magnitude[i, j]

            # Apply thresholding

            if mag >= low_threshold:  # if less than low threshold, not an edge
                if mag >= high_threshold:
                    thresh[i, j] = 255  # Strong edge
                elif mag >= low_threshold:
                    thresh[i, j] = 50  # Weak edge
    return suppressed, thresh

def hysteresis(thresh, strong_edge, weak_edge):
    edges = np.zeros_like(thresh)
    weak_i, weak_j = np.where((thresh >= weak_edge) & (thresh < strong_edge))

    # Connect weak edges to strong edges
    for i, j in zip(weak_i, weak_j):
        if any(thresh[x, y] >= strong_edge for x in range(i-1, i+2) for y in range(j-1, j+2)):
            edges[i, j] = 255
    return edges

# Main code
blurred_image = gauss_blur(grayscale_image, 1.0)
grad_x, grad_y, grad_magnitude, grad_direction = sobel_filter(blurred_image)

grad_magnitude_normalized = grad_magnitude/grad_magnitude.max()*255
grad_magnitude_norm = cv2.convertScaleAbs(grad_magnitude_normalized)

suppressed, thresh = nonmax_suppression(grad_magnitude_norm, grad_direction, 0.15, 0.09)
final_canny = hysteresis(thresh, 255, 50)

cv2.imshow('Frame View:', grayscale_image)
# cv2.imshow('Frame View:', grad_magnitude_norm)
cv2.waitKey(0)
# cv2.imshow('Frame View:', sup_and_threshold)
# cv2.waitKey(0)
# cv2.imshow('Frame View:', final_canny.astype(np.uint8))
# cv2.waitKey(0)
cv2.imshow('Frame View:', final_canny.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
