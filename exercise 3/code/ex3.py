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

def image_derivatives(image):
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

def load_image_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
    return image

def create_test_image(size=100, thickness=3):
    image = np.zeros((size, size), dtype=np.uint8)
    cv2.line(image, (0,0), (size-1, size-1), color=255, thickness=thickness)
    return image

def make_binary_image(image, threshold):
    binary = np.where(np.abs(image) > threshold, 255, 0)
    return binary

def show_two_images(img1, img2, title, cmap='gray'):
    fig, axs = plt.subplots(2, 1, figsize=(5, 10))

    axs[0].imshow(img1, cmap=cmap)
    axs[0].axis('off')
    axs[0].set_title('Image 1')

    axs[1].imshow(img2, cmap=cmap)
    axs[1].axis('off')
    axs[1].set_title('Image 2')
    plot_path = os.path.join(plot_save_directory, f'{title}.jpg')
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f'saved to {plot_path}')
    plt.show()

def get_image_log(image):
    image = image.astype(np.float32)
    log_image = np.log(image + 1e-6)
    return log_image

def calculate_norm(i1, i2):
    magnitude = np.sqrt(i1 ** 2 + i2 ** 2)
    return magnitude


def soft_thresh(x, t, s=5):
    y = 1 / (1 + np.exp(-s * (x - t)))
    return 1 - y

def two_squares(shadow_flag):

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

def do_retinex(image, threshold):
    log_image = get_image_log(image)
    log_ix, log_iy = image_derivatives(log_image)
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

def do_retinex_multiple_thresholds(image, thresholds):
    print(f'starting retinex')
    kl_by_reflectance = {}
    if len(thresholds) > 1:
        for t in thresholds:
            reflectance, illumination = do_retinex(image, t)
            kl_by_reflectance[t] = (reflectance, illumination)
        return kl_by_reflectance
    if len(thresholds) == 1:
        reflectance, illumination = do_retinex(image, thresholds[0])
        return reflectance, illumination


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
    diag = np.diagonal(reflectance)
    plt.plot(np.arange(len(diag)), diag, color='red', linewidth=2, label='Diagonal R[x,x]')
    plt.title('Reflectance with Diagonal Overlay')
    plt.grid(True)
    plt.axis('off')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()

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

def save_plot(image, filename, title, cmap='gray', vmin=None, vmax=None):
    plt.figure(figsize=(6, 6))
    if image.ndim == 2:
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(image)
    file_path = os.path.join(plot_save_directory, f'{filename}.jpeg')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def q3():
    path_two_squares = 'exercise 3/ex3-files/simul_cont_squares.tif'
    grayscale_image = load_image_grayscale(path_two_squares)
    ix, iy = image_derivatives(grayscale_image)
    laplacian = deriv2laplace(ix, iy)
    threshold = 15
    binary = make_binary_image(laplacian, threshold=threshold)
    show_two_images(grayscale_image, binary, f'simultaneous_contrast_t{str(threshold)}')

def q4():
    path_cross = 'exercise 3/ex3-files/cross.tif'
    grayscale_image = load_image_grayscale(path_cross)
    ix, iy = image_derivatives(grayscale_image)
    laplacian = deriv2laplace(ix, iy)
    threshold = 4.5
    binary = make_binary_image(laplacian, threshold=threshold)
    show_two_images(grayscale_image, binary, f'cross_t{str(threshold)}')

def q5():
    path_kofkaring = 'exercise 3/ex3-files/kofka_ring.tif'
    grayscale_image = load_image_grayscale(path_kofkaring)
    ix, iy = image_derivatives(grayscale_image)
    laplacian = deriv2laplace(ix, iy)
    threshold = 17
    binary = make_binary_image(laplacian, threshold=threshold)
    show_two_images(grayscale_image, binary, f'kofka_ring_t{str(threshold)}')

def q8():
    titles = ['original', 'reflectance', 'illumination']
    flag = 2
    grayscale_image = two_squares(flag)
    threshold = 0.3
    reflectance, illumination = do_retinex_multiple_thresholds(grayscale_image, [threshold])
    show_3_images_and_diagonal_overlay([grayscale_image, reflectance, illumination],
                                       titles, reflectance, title=f'squares{flag}_t{str(threshold)}', cmap='gray')

def q9():
    mat = sio.loadmat('exercise 3/ex3-files/checkerShadow.mat')
    im1 = mat['im1']
    print(im1.shape)
    coordinates = mat['x1'], mat['y1'], mat['x2'], mat['y2']
    coordinates = [c.item() for c in coordinates]
    x1, y1, x2, y2 = coordinates
    threshold = 0.2
    print(f'original image intensity: a:{im1[y1, x1]}, b:{im1[y2, x2]}')
    reflectance, illumination = do_retinex_multiple_thresholds(im1, [threshold])
    save_plot(reflectance, f'checkerShadow_t{str(threshold)}', f'reflectance_t{str(threshold)}')
    print(f'reflectance image intensity: a:{reflectance[y1, x1]}, b:{reflectance[y2, x2]}')

def q10():
    mat = sio.loadmat('exercise 3/ex3-files/runner.mat')
    im1 = mat['im1']
    print(im1.shape)
    thresholds = [0.05, 0.1, 0.15]
    kl_by_reflectance = do_retinex_multiple_thresholds(im1, thresholds)
    for t in thresholds:
        reflectance, illumination = kl_by_reflectance[t]
        save_plot(reflectance, f'runner_t{str(t)}', f'reflectance_t{str(t)}')

def q11():
    mat = sio.loadmat('exercise 3/ex3-files/couch.mat')
    im1 = mat['im1']
    thresholds = [0.01, 0.02, 0.03, 0.04]
    kl_by_reflectance = do_retinex_multiple_thresholds(im1, thresholds)
    for t in thresholds:
        reflectance, illumination = kl_by_reflectance[t]
        save_plot(reflectance, f'couch_t{str(t)}', f'reflectance_t{str(t)}')

def main():
    q11()

if __name__ == "__main__":
    main()
