import cv2
import numpy as np
import matplotlib
import scipy
from scipy.signal import convolve2d
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def show_image_matlab(im1, sc=None, window_name='Image'):
    if sc is not None:
        im1 = np.clip((im1 - sc[0]) / (sc[1] - sc[0]) * 255, 0, 255).astype(np.uint8)
    else:
        # If image is not already in 8-bit format, normalize it
        if im1.dtype != np.uint8:
            im1 = cv2.normalize(im1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Show the image in a window
    cv2.imshow(window_name, im1)
    cv2.waitKey(0)

def show_image(img, title='title'):
    cv2.imshow(title, img)
    cv2.waitKey(0)

def get_laplacian(img):
    image = img.astype(np.float32)
    kx = np.array([[0.5, -0.5]], dtype=np.float32)
    ky = np.array([[-0.5], [0.5]], dtype=np.float32)
    ix = convolve2d(image, kx, mode='same', boundary='symm')
    iy = convolve2d(image, ky, mode='same', boundary='symm')
    d2x = convolve2d(ix, kx, mode='same', boundary='symm')
    d2y = convolve2d(iy, ky, mode='same', boundary='symm')
    laplacian = d2x + d2y
    return laplacian


def load_image_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

def create_test_image(size=100, thickness=3):
    image = np.zeros((size, size), dtype=np.uint8)
    cv2.line(image, (0,0), (size-1, size-1), color=255, thickness=thickness)
    return image

def load_and_show_image(image_path):
    image = load_image_grayscale(image_path)
    show_image(image)

def laplacian_cv2(image):
    return cv2.Laplacian(image, cv2.CV_64F, ksize=3)
def laplacian_from_image(image, cv=False):
    if cv:
        laplacian = laplacian_cv2(image)
    else:
        laplacian = get_laplacian(image)
    return laplacian

def print_image_details(img):
    print(f'Image shape: {img.shape}')
    return

def make_binary_image(image, threshold):
    binary = np.where(image > threshold, 255, 0).astype(np.uint8)
    return binary

def show_two_images(img1, img2, cmap='gray'):
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))

    axs[0].imshow(img1, cmap=cmap)
    axs[0].axis('off')
    axs[0].set_title('Image 1')

    axs[1].imshow(img2, cmap=cmap)
    axs[1].axis('off')
    axs[1].set_title('Image 2')

    plt.tight_layout()
    plt.show()

def show_image_and_binary_laplacian(image, cv=False):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = laplacian_from_image(image, cv)
    default_threshold = 10
    binary_laplacian = make_binary_image(laplacian, threshold=default_threshold)
    show_two_images(image, binary_laplacian)

def main():
    path_custom_image = 'exercise 3/data/mouse.jpeg'
    path_two_squares = 'exercise 3/ex3-files/simul_cont_squares.tif'
    path_cross = 'exercise 3/ex3-files/cross.tif'
    path_kofkaring = 'exercise 3/ex3-files/kofka_ring.tif'
    grayscale_image = load_image_grayscale(path_kofkaring)
    laplacian = laplacian_from_image(image=grayscale_image)
    show_image(laplacian, 'laplacian')
    print_image_details(grayscale_image)
    show_image_and_binary_laplacian(grayscale_image)





if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()