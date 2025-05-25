import cv2
import numpy as np
import matplotlib
# import scipy
import scipy.io as sio
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d
from scipy.special.cython_special import kl_div
import os

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

kx = np.array([[0.5, -0.5]], dtype=np.float32)
ky = np.array([[-0.5], [0.5]], dtype=np.float32)
plot_save_directory = 'exercise 3/plots'

def show_matlab(im1, sc=None):
    """
    Display an image in grayscale with aspect ratio preserved,
    optionally with a given intensity scale.

    Parameters:
        im1 : ndarray
            The image to display.
        sc : tuple or list, optional
            The intensity scale (vmin, vmax). If not provided, matplotlib auto-scales.
    """
    plt.figure()
    if sc is not None:
        plt.imshow(im1, cmap='gray', vmin=sc[0], vmax=sc[1])
    else:
        plt.imshow(im1, cmap='gray')
    plt.axis('image')
    plt.colorbar()
    plt.show()

def show_image(img, title='title'):
    cv2.imshow(title, img)
    cv2.waitKey(0)


def deriv2laplace(ix, iy):
    ix2 = convolve2d(ix, kx, mode='same')
    iy2 = convolve2d(iy, ky, mode='same')
    return ix2 + iy2

def get_image_derivatives(image):
    image = image.astype(np.float32)
    ix = convolve2d(image, kx, mode='full')
    iy = convolve2d(image, ky, mode='full')
    ix = ix[:,:-1]
    iy = iy[:-1,:]
    ix[0,:] = 0
    ix[-1,:] = 0
    ix[:,0] = 0
    ix[:,-1] = 0
    iy[0,:] = 0
    iy[-1,:] = 0
    iy[:,0] = 0
    iy[:,-1] = 0
    return ix, iy

def get_laplacian(img):
    image = img.astype(np.float32)
    ix, iy = get_image_derivatives(image)
    ix, iy = ix, iy
    d2x = convolve2d(ix, kx, mode='full')
    d2y = convolve2d(iy, ky, mode='full')
    laplacian = d2x + d2y
    return laplacian

def load_image_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
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
    binary = np.where(np.abs(image) > threshold, 255, 0).astype(np.uint8)
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

def get_image_log(image):
    image = image.astype(np.float32)
    log_image = np.log(image + 1e-6)
    return log_image

def calculate_norm(i1, i2):
    magnitude = np.sqrt(i1 ** 2 + i2 ** 2)
    return magnitude


def soft_thresh(x, t, s=5):
    """
    A soft threshold function.
    Parameters:
        x : numpy.ndarray
            Input array.
        t : float
            Threshold.
        s : float
            Softness parameter (default is 5).
    Returns:
        numpy.ndarray
    """
    y = 1 / (1 + np.exp(-s * (x - t)))
    return 1 - y

def two_squares(shadow_flag):
    """
    Generate a synthetic image of two squares, one potentially in shadow.
    Parameters:
        shadow_flag : int
            If 1, apply soft shadow with broader spread.
            If 0, apply tighter shadow.
    Returns:
        im : numpy.ndarray
            The resulting synthetic image.
    """
    R = np.ones((50, 50))
    R[29:40, 29:40] = 2  # MATLAB is 1-indexed; Python is 0-indexed
    R[9:20, 9:20] = 2

    x, y = np.meshgrid(np.arange(1, 51), np.arange(1, 51))
    rr = (x - 35) ** 2 + (y - 35) ** 2

    if shadow_flag == 1:
        L = 1 - 0.3 * soft_thresh(rr, 13**2, 0.05)
    else:
        L = 1 - 0.3 * soft_thresh(rr, 3**2, 0.04)

    im = R * L
    return im


def inv_del2(im_size):
    isize = 2 * max(im_size)
    K = np.zeros((isize, isize), dtype=np.float32)
    center = isize // 2
    K[center, center] = -4
    K[center + 1, center] = 1
    K[center - 1, center] = 1
    K[center, center + 1] = 1
    K[center, center - 1] = 1

    Khat = fft2(K / 4.0)

    Khat_safe = Khat.copy()
    Khat_safe[np.abs(Khat) < 1e-10] = 1
    invKhat = 1.0 / Khat_safe
    invKhat[np.abs(Khat) < 1e-10] = 0

    invK = np.real(ifft2(invKhat))

    shift_kernel = np.array([[1, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]], dtype=np.float32)
    invK = convolve2d(invK, shift_kernel, mode='same')
    return invK

def retinex_one_threshold(image, threshold):
    log_image = get_image_log(image)
    log_ix, log_iy = get_image_derivatives(log_image)
    log_der_norm = calculate_norm(log_ix, log_iy)
    mask = log_der_norm >= threshold
    filtered_ix = mask * log_ix
    filtered_iy = mask * log_iy
    reflectance_laplacian = deriv2laplace(filtered_ix, filtered_iy)
    inv_lap_k = inv_del2(image.shape)
    log_reflectance = convolve2d(reflectance_laplacian, inv_lap_k, mode='same')
    reflectance = np.exp(log_reflectance)
    illumination = image / (reflectance + 1e-8)
    return reflectance, illumination

def simplified_retinex_multiple_thresholds(image, thresholds):
    print(f'starting retinex')
    log_image = get_image_log(image)
    log_ix, log_iy = get_image_derivatives(log_image)
    log_der_norm = calculate_norm(log_ix, log_iy)
    kl_by_reflectance = {}
    if len(thresholds) > 1:
        for t in thresholds:
            reflectance, illumination = retinex_one_threshold(image, t)
            kl_by_reflectance[t] = (reflectance, illumination)
        return kl_by_reflectance
    if len(thresholds) == 1:
        reflectance, illumination = retinex_one_threshold(image, thresholds[0])
        return reflectance, illumination

def laplacian_question():
    path_custom_image = 'exercise 3/data/mouse.jpeg'
    path_two_squares = 'exercise 3/ex3-files/simul_cont_squares.tif'
    path_cross = 'exercise 3/ex3-files/cross.tif'
    path_kofkaring = 'exercise 3/ex3-files/kofka_ring.tif'
    grayscale_image = load_image_grayscale(path_two_squares)
    laplacian = laplacian_from_image(grayscale_image, cv=False)
    binary = make_binary_image(laplacian, threshold=10)
    show_two_images(grayscale_image, laplacian)

def plot_diagonal(image, title='Diagonal Intensity Profile'):

    diag = np.diagonal(image)

    plt.figure()
    plt.plot(diag, marker='o')
    plt.title(title)
    plt.xlabel('Pixel index along diagonal (x)')
    plt.ylabel('Intensity R[x, x]')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_3_images_and_diagonal_overlay(images, titles, reflectance, title, cmap='gray'):
    if len(images) != 3 or len(titles) != 3:
        raise ValueError("Expected exactly 3 images and 3 titles.")
    plot_path = os.path.join(plot_save_directory, f'{title}.jpeg')
    plt.figure(figsize=(12, 10))

    for i in range(3):
        plt.subplot(2, 2, i + 1)
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis('off')

    plt.subplot(2, 2, 4)
    # plt.imshow(reflectance, cmap=cmap)
    diag = np.diagonal(reflectance)
    plt.plot(np.arange(len(diag)), diag, color='red', linewidth=2, label='Diagonal R[x,x]')
    plt.title('Reflectance with Diagonal Overlay')
    plt.grid(True)
    plt.axis('off')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

def retinex_squares():
    titles = ['original', 'reflectance', 'illumination']
    grayscale_image = two_squares(1)
    threshold = 0.07
    reflectance, illumination = simplified_retinex_multiple_thresholds(grayscale_image, [threshold])
    show_3_images_and_diagonal_overlay([grayscale_image, reflectance, illumination],
                                       titles, reflectance, title=f'squares_t{str(threshold)}', cmap='gray')


def show_image_with_dots(image, x1, y1, x2, y2, dot_radius=5,
                         cmap='gray', title='Image with Red Dots'):
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap=cmap)
    plt.scatter([x1, x2], [y1, y2], s=dot_radius**2,
                edgecolors='red', facecolors='none', linewidths=2)
    plt.title(title)
    plt.axis('off')
    plt.show()
    print('finished')


def q9():
    mat = sio.loadmat('exercise 3/ex3-files/checkerShadow.mat')
    im1 = mat['im1']
    print(im1.shape)
    show_matlab(im1)
    coordinates = mat['x1'], mat['y1'], mat['x2'], mat['y2']
    coordinates = [c.item() for c in coordinates]
    x1, y1, x2, y2 = coordinates
    threshold = 0.07
    print(f'intensity: a:{im1[y1, x1]}, b:{im1[y2, x2]}')
    reflectance, illumination = simplified_retinex_multiple_thresholds(im1, [threshold])
    show_3_images_and_diagonal_overlay([im1, reflectance, illumination],
                                       ['original', 'reflectance', 'illumination'],
                                       reflectance, title=f'checker_shadow_t{threshold}', cmap='gray')

def q10():
    mat = sio.loadmat('exercise 3/ex3-files/runner.mat')
    im1 = mat['im1']
    print(im1.shape)
    show_matlab(im1)
    thresholds = [0.05, 0.1, 0.15]
    kl_by_reflectance = simplified_retinex_multiple_thresholds(im1, thresholds)
    for t in thresholds:
        reflectance, illumination = kl_by_reflectance[t]
        show_3_images_and_diagonal_overlay([im1, reflectance, illumination],
                                           ['original', 'reflectance', 'illumination'],
                                           reflectance, title=f'runner_t{str(t)}', cmap='gray')

def q11():
    mat = sio.loadmat('exercise 3/ex3-files/couch.mat')
    im1 = mat['im1']
    show_matlab(im1)
    thresholds = [0.01, 0.02, 0.03, 0.04]
    kl_by_reflectance = simplified_retinex_multiple_thresholds(im1, thresholds)
    for t in thresholds:
        reflectance, illumination = kl_by_reflectance[t]
        show_3_images_and_diagonal_overlay([im1, reflectance, illumination],
                                           ['original', 'reflectance', 'illumination'],
                                           reflectance, title=f'couch_t{str(t)}', cmap='gray')

def main():
    q11()

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()