import cv2
import numpy as np


def show_image(img, title='title'):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def x_derivative(img):
    derivative_finder_x = np.array([[0.25, -0.25]], dtype=np.float32)
    ix = cv2.filter2D(src=img, ddepth=-1, kernel=derivative_finder_x)
    return ix

def y_derivative(img):
    derivative_finder_y = np.array([[0.25, -0.25]], dtype=np.float32).T
    iy = cv2.filter2D(src=img, ddepth=-1, kernel=derivative_finder_y)
    return iy


def find_image_derivatives(img):
    ix = x_derivative(img)
    iy = y_derivative(img)
    return ix, iy

def single_image_derivative(img):
    derivative_finder_x = np.array([[0.25, -0.25]], dtype=np.float32)
    derivative_finder_y = derivative_finder_x.T
    ix = cv2.filter2D(src=img, ddepth=-1, kernel=derivative_finder_x)
    iy = cv2.filter2D(src=img, ddepth=-1, kernel=derivative_finder_y)
    return ix, iy


def get_laplacian(ix, iy):
    ixx = x_derivative(ix)
    iyy = y_derivative(iy)
    laplacian = ixx+iyy
    return laplacian

def load_image_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def main():
    path = 'data/mouse.jpeg'
    img = load_image_grayscale(path)
    ix, iy = find_image_derivatives(img)
    laplacian = get_laplacian(ix, iy)
    show_image(ix)
    show_image(iy)
    show_image(laplacian)


if __name__ == "__main__":
    main()