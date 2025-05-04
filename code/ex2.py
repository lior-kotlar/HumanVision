import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import signal
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve

PATCHED = True
X_DERIVATIVE_IDX = 0
Y_DERIVATIVE_IDX = 1
PATCH_SIZE = 50
FRAME1_FILE_NAME = 'data/frame1.jpg'
FRAME2_FILE_NAME = 'data/frame2.jpg'
WINDOW_SIZE = 3
EIGEN_THRESHOLD = 0.001
CONDITION_THRESHOLD = 1000
FRAME_SKIP = 3


def get_patch(image, patch_center):
    half_window = PATCH_SIZE // 2
    template = image[patch_center[1] - half_window:patch_center[1] + half_window + 1,
               patch_center[0] - half_window:patch_center[0] + half_window + 1]
    return template


def compute_error_image(i_target, i_shifted):
    error = i_target - i_shifted
    return error


def warp_image(img, tx, ty, mask):
    m = np.float32([[1, 0, tx],
                    [0, 1, ty]])

    shifted = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]))
    height, width = img.shape[:2]
    if mask is None:
        mask = np.ones((height, width), np.uint8)

    warp_mask = cv2.warpAffine(mask, m, (width, height), flags=cv2.INTER_NEAREST)

    return shifted, warp_mask


def warp_matlab(Im, v):
    row_num, col_num = Im.shape

    # Define grid in MATLAB-style coordinates (1-based indexing and flipped rows)
    x = np.arange(1, col_num + 1)
    y = np.arange(row_num, 0, -1)  # Flip vertically to match MATLAB's flipud

    # Create the interpolator
    interpolator = RegularGridInterpolator((y, x), Im, bounds_error=False, fill_value=np.nan)

    # Create meshgrid of coordinates
    xx, yy = np.meshgrid(x, y)

    # Shift coordinates
    coords = np.stack([(yy + v[1]).ravel(), (xx + v[0]).ravel()], axis=-1)

    # Interpolate
    Iw = interpolator(coords).reshape(Im.shape)

    # Create warp mask (1 where not NaN, 0 where NaN)
    warpMask = ~np.isnan(Iw)
    Iw[~warpMask] = 0
    warpMask = warpMask.astype(np.float32)

    return Iw, warpMask


def show_image(img, title):
    cv2.imshow(title, img)
    cv2.waitKey(0)


def show_image_plt(img):
    plt.imshow(img, cmap='gray')


def get_time_derivative(img1, img2):
    time_kernel = np.array([[0.25, 0.25],
                            [0.25, 0.25]], dtype=np.float32)

    img1_smoothed = cv2.filter2D(img1, -1, time_kernel)
    img2_smoothed = cv2.filter2D(img2, -1, time_kernel)

    temporal_derivative = img2_smoothed - img1_smoothed

    return temporal_derivative


def get_time_derivative_plt(img1, img2):
    filter_t1 = np.array([[-0.25, -0.25],
                          [-0.25, -0.25]])

    filter_t2 = np.array([[0.25, 0.25],
                          [0.25, 0.25]])
    it1 = signal.convolve2d(img1, filter_t1, mode='same')
    it2 = signal.convolve2d(img2, filter_t2, mode='same')
    it = it1 + it2
    return it


def single_image_derivative(img):
    derivative_finder_x = np.array([[0.25, -0.25],
                                    [0.25, -0.25]], dtype=np.float32)
    derivative_finder_y = derivative_finder_x.T
    ix = cv2.filter2D(src=img, ddepth=-1, kernel=derivative_finder_x)
    iy = cv2.filter2D(src=img, ddepth=-1, kernel=derivative_finder_y)
    return ix, iy


def single_image_derivative_plt(img):
    derivative_finder_x = np.array([[0.25, -0.25],
                                    [0.25, -0.25]], dtype=np.float32)
    derivative_finder_y = derivative_finder_x.T
    ix = signal.convolve2d(img, derivative_finder_x, mode='same')
    iy = signal.convolve2d(img, derivative_finder_y, mode='same')
    return ix, iy


def image_derivatives(img1, img2):
    img1_derivatives = single_image_derivative(img1)
    img2_derivatives = single_image_derivative(img2)
    ix = img1_derivatives[0] + img2_derivatives[0]
    iy = img1_derivatives[1] + img2_derivatives[1]
    time_derivative = get_time_derivative(img1, img2)
    return ix, iy, time_derivative


def load_images(path1, path2, grayscale=True):
    image1 = cv2.imread(path1)
    image2 = cv2.imread(path2)
    if grayscale:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    return image1, image2


def read_2_frames(cap, skip):
    ret1, frame1 = cap.read()
    ret_ = None
    if ret1:
        for _ in range(skip):
            ret_, frame_ = cap.read()
            if ret_ is None:
                break
    if ret_ or skip == 0:
        ret2, frame2 = cap.read()
        if ret2:
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            return frame1, frame2
    print('no more frames')
    return None, None


def extract_two_frames(video_file_path, skip=FRAME_SKIP, save=True):
    cap = cv2.VideoCapture(video_file_path)
    frame1, frame2 = read_2_frames(cap, skip)
    if frame1 is None or frame2 is None:
        print('cant read two frames')
        exit(1)
    cv2.imwrite(FRAME1_FILE_NAME, frame1)
    cv2.imwrite(FRAME2_FILE_NAME, frame2)
    return frame1, frame2


def get_patch_center(image):
    coords = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            coords.append((x, y))
            # cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
            print(f"Clicked at: ({x}, {y})")
            cv2.destroyAllWindows()

    # Show image
    window_name = 'Click to select patch center'
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        cv2.imshow(window_name, image)
        key = cv2.waitKey(1) & 0xFF
        if len(coords) > 0 or key == 27:  # ESC key to exit without clicking
            break

    cv2.destroyAllWindows()
    return coords[0] if coords else None


def analyze_window(fx, fy, ft, lmda):
    fx_flat = fx.flatten()
    fy_flat = fy.flatten()
    ft_flat = ft.flatten()
    ft_vec = np.array([ft_flat]).T
    at = np.vstack((fx_flat, fy_flat))
    a = at.T
    ata = np.matmul(at, a)
    eigen_values = np.linalg.eig(ata).eigenvalues
    if np.min(eigen_values) < EIGEN_THRESHOLD:
        return None
    cond_number = np.linalg.cond(ata)

    if cond_number > CONDITION_THRESHOLD:
        return None
    regularization_matrix = ata + lmda * np.eye(2)
    v = np.linalg.pinv(regularization_matrix) @ a.T @ ft_vec
    return v


def crop_window(ix, iy, it, mask, window_center):
    full_image_height, full_image_width = it.shape
    x_center, y_center = window_center
    shul = WINDOW_SIZE // 2
    if x_center < shul or x_center > full_image_width - shul - 1 or \
            y_center < shul or y_center > full_image_height - shul - 1:
        print('window out of bounds')
        return None

    ix_window = ix[y_center - shul: y_center + shul + 1, x_center - shul:x_center + shul + 1]
    iy_window = iy[y_center - shul: y_center + shul + 1, x_center - shul:x_center + shul + 1]
    it_window = it[y_center - shul: y_center + shul + 1, x_center - shul:x_center + shul + 1]
    mask_window = mask[y_center - shul: y_center + shul + 1, x_center - shul: x_center + shul + 1]

    return ix_window, iy_window, it_window, mask_window


def blur_downsample(image, kernel_size=5, sigma=1.0):

    g1d = cv2.getGaussianKernel(ksize=kernel_size, sigma=sigma)
    kernel = g1d @ g1d.T

    if image.ndim == 3:
        blurred = np.stack([cv2.filter2D(image[:, :, c], -1, kernel) for c in range(image.shape[2])], axis=2)
    else:
        blurred = cv2.filter2D(image, -1, kernel)
    downsampled = blurred[::2, ::2] if image.ndim == 2 else blurred[::2, ::2, :]

    return downsampled


def lk_alg(i1, i2, lmda = 0.0, mask=None, v_initial=[0, 0], num_iterations=2):
    accumulated_vector = v_initial
    previous_iteration_guess = [0, 0]
    i1x, i1y = single_image_derivative_plt(i1)
    height, width = i1.shape
    mask_prev = mask if mask else np.ones((height, width))
    for iter in range(num_iterations):
        print(f'iteration: {iter}')
        x_shift, y_shift = accumulated_vector
        print(f'x_shift: {x_shift}, y_shift: {y_shift}')
        warped_image, warp_mask = warp_image(i2, x_shift, y_shift, mask)
        # warped_image, warp_mask = warp_matlab(i2, (x_shift, y_shift))
        new_mask = mask_prev * warp_mask
        show_image(i1, 'i1')
        show_image(warped_image, 'warped_image')
        show_image(new_mask, 'new_mask')
        it = get_time_derivative_plt(i1, warped_image)
        i2x, i2y = single_image_derivative_plt(warped_image)
        ix = i1x + i2x
        iy = i1y + i2y
        velocity_vectors = []
        shul = WINDOW_SIZE // 2
        for x_window_center in range(shul, width - shul - 1):
            for y_window_center in range(shul, height - shul - 1):
                ix_window, iy_window, it_window, mask_window = crop_window(ix, iy, it, new_mask, (x_window_center,y_window_center))
                mask_flat = mask_window.flatten().astype(np.uint8)
                if np.any(mask_flat == 0):
                    continue
                v = analyze_window(ix_window, iy_window, it_window, lmda=lmda)
                if v is not None:
                    velocity_vectors.append(v)

        ux = np.mean(velocity_vectors[:][0])
        uy = np.mean(velocity_vectors[:][1])

        print(f'u = ({ux}, {uy})')

        previous_iteration_guess = [ux, uy]

        accumulated_vector[1] += ux
        accumulated_vector[0] += uy

        print(f'accumulated vector: {accumulated_vector}')

    return accumulated_vector


def full_lk_alg(i1, i2, lmda=0.001, mask=None, num_iterations=5):
    d = [0, 0]
    patch_center = get_patch_center(i1)
    if PATCHED:
        i1 = get_patch(i1, patch_center)
        i2 = get_patch(i2, patch_center)
    i1_downsampled = blur_downsample(i1, kernel_size=5, sigma=lmda)
    i2_downsampled = blur_downsample(i2, kernel_size=5, sigma=lmda)

    initial_guess = lk_alg(i1_downsampled, i2_downsampled, lmda, None, [0, 0], 1)
    initial_guess = [2*p for p in initial_guess]

    print(f'initial guess: {initial_guess}')

    final_vector = lk_alg(i1, i2, lmda=lmda, mask=None, v_initial=initial_guess, num_iterations=num_iterations)

    print(f'final vector: {final_vector}')



def main():
    if WINDOW_SIZE % 2 == 0:
        print('window size needs to be odd')
        exit(1)
    mode = sys.argv[1]
    if mode == 'v':
        print("mode v")
        frame1, frame2 = extract_two_frames(sys.argv[2])
    else:
        frame1, frame2 = load_images(FRAME1_FILE_NAME, FRAME2_FILE_NAME)

    full_lk_alg(frame1, frame2)


if __name__ == '__main__':
    main()
