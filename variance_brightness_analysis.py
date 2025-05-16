import numpy as np
from matplotlib import pyplot as plt
from raw_utils import *
from alignment import *
import rawpy
from dataset_navigation import get_image_paths
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib



def compute_local_variance_single_channel(image, kernel_size=5):
    image = image.astype(np.float64)
    # cv2 blur calculates local mean using a box filter
    mean = cv2.blur(image, (kernel_size, kernel_size))
    mean_sq = cv2.blur(image**2, (kernel_size, kernel_size))
    variance = mean_sq - mean**2
    return variance

def compute_local_variance(image, kernel_size=5):
    R_variance = compute_local_variance_single_channel(image[:, :, 0], kernel_size)
    G1_variance = compute_local_variance_single_channel(image[:, :, 1], kernel_size)
    B_variance = compute_local_variance_single_channel(image[:, :, 2], kernel_size)
    G2_variance = compute_local_variance_single_channel(image[:, :, 3], kernel_size)

    return np.stack((R_variance, G1_variance, B_variance, G2_variance), axis=2)

def compute_local_mean_single_channel(image, kernel_size=5):
    image = image.astype(np.float64)
    # cv2 blur calculates local mean using a box filter
    mean = cv2.blur(image, (kernel_size, kernel_size))
    return mean

def compute_local_mean(image, kernel_size=5):
    R_variance = compute_local_mean_single_channel(image[:, :, 0], kernel_size)
    G1_variance = compute_local_mean_single_channel(image[:, :, 1], kernel_size)
    B_variance = compute_local_mean_single_channel(image[:, :, 2], kernel_size)
    G2_variance = compute_local_mean_single_channel(image[:, :, 3], kernel_size)

    return np.stack((R_variance, G1_variance, B_variance, G2_variance), axis=2)


def create_array_per_pair(image, variance_difference):
    image = image.flatten()
    variance_difference = variance_difference.flatten()
    if image.shape != variance_difference.shape:
        raise ValueError("Inputs must have the same shape after flattening.")
    paired_array = np.stack((image, variance_difference), axis=1)
    return paired_array

def average_y_per_x(paired_array, threshold=0):
    x = paired_array[:, 0]
    y = paired_array[:, 1]
    unique_x, inverse_indices = np.unique(x, return_inverse=True)
    sum_y = np.bincount(inverse_indices, weights=y)
    count_y = np.bincount(inverse_indices)
    avg_y = sum_y / count_y
    # Apply threshold
    mask = count_y >= threshold
    filtered_x = unique_x[mask]
    filtered_avg_y = avg_y[mask]
    return np.stack((filtered_x, filtered_avg_y), axis=1)

def average_y_per_x_binned(paired_array, num_bins=100, threshold=0):
    x = paired_array[:, 0]
    y = paired_array[:, 1]

    # Create bins
    x_min, x_max = x.min(), x.max()
    bins = np.linspace(x_min, x_max, num_bins + 1)

    # Assign each x to a bin
    bin_indices = np.digitize(x, bins) - 1  # shift to 0-based index

    # Remove out-of-range values
    valid_mask = (bin_indices >= 0) & (bin_indices < num_bins)
    bin_indices = bin_indices[valid_mask]
    y = y[valid_mask]

    # Compute average y per bin
    sum_y = np.bincount(bin_indices, weights=y, minlength=num_bins)
    count_y = np.bincount(bin_indices, minlength=num_bins)
    avg_y = np.divide(sum_y, count_y, out=np.zeros_like(sum_y), where=count_y > 0)

    # Apply threshold
    mask = count_y >= threshold
    bin_centers = (bins[:-1] + bins[1:]) / 2
    filtered_x = bin_centers[mask]
    filtered_avg_y = avg_y[mask]

    return np.stack((filtered_x, filtered_avg_y), axis=1)

#
# def get_brightness_to_variance_difference():
#     # n = 2
#     # diffused_image_paths = paths_dict['filter_long_exp'][:n]
#     # clear_image_paths = paths_dict['long_exp'][:n]
#     paths_dict = get_image_paths()
#
#     diffused_image_paths = paths_dict['filter_long_exp']
#     clear_image_paths = paths_dict['long_exp']
#
#     brightness_to_variance_difference_r = np.zeros((0,2))
#     brightness_to_variance_r = np.zeros((0,2))
#     for diffused_image_path, clear_image_path in tqdm(zip(diffused_image_paths, clear_image_paths), total=len(diffused_image_paths)):
#         clear, diffused = align_and_crop_raw_images(clear_image_path, diffused_image_path)
#         clear_channels = clear['channels']
#         diffused_channels = diffused['channels']
#         clear_variance = compute_local_variance(clear_channels)
#         diffused_variance = compute_local_variance(diffused_channels)
#         variance_difference = (diffused_variance - clear_variance)
#         brightness_to_variance_difference_r = np.append(brightness_to_variance_difference_r, create_array_per_pair(clear_channels[:,:,0], variance_difference[:,:,0]), axis=0)
#
#     return brightness_to_variance_difference_r
#
# def get_brightness_to_variance():
#     paths_dict = get_image_paths()
#
#     # n = 2
#     # diffused_image_paths = paths_dict['filter_long_exp'][:n]
#     # clear_image_paths = paths_dict['long_exp'][:n]
#
#     clear_image_paths = paths_dict['long_exp']
#
#     brightness_to_variance_r = np.zeros((0,2))
#     for clear_image_path in tqdm(clear_image_paths, total=len(clear_image_paths)):
#         clear_channels = pack_raw(rawpy.imread(clear_image_path))
#         clear_variance = compute_local_variance(clear_channels)
#         brightness_to_variance_r = np.append(brightness_to_variance_r, create_array_per_pair(clear_channels[:,:,0], clear_variance[:,:,0]), axis=0)
#     return brightness_to_variance_r
#


def _process_variance_difference_pair(pair):
    diffused_path, clear_path = pair
    try:
        clear, diffused = align_and_crop_raw_images(clear_path, diffused_path)
    except Exception as e:
        print(f"Error processing pair {pair}: {e}")
        return np.zeros((0, 2))
    clear_ch = clear['channels']
    diff_ch = diffused['channels']
    clear_var = compute_local_variance(clear_ch)
    diff_var = compute_local_variance(diff_ch)
    return create_array_per_pair(
        clear_ch[:, :, 0],
        diff_var[:, :, 0] - clear_var[:, :, 0]
    )


def get_brightness_to_variance_difference(num_workers=None):
    """
    Parallel version of get_brightness_to_variance_difference.
    Set num_workers to control number of processes (default: CPU count).
    """
    paths_dict = get_image_paths()
    pairs = list(zip(paths_dict['filter_long_exp'], paths_dict['long_exp']))
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # executor.map yields results in order; wrap for progress bar
        results = list(tqdm(
            executor.map(_process_variance_difference_pair, pairs),
            total=len(pairs)
        ))
    # Stack all arrays into a single (N,2) array
    return np.vstack(results)


def _process_brightness_to_variance(image_path):
    raw = rawpy.imread(image_path)
    clear_ch = pack_raw(raw)
    clear_var = compute_local_variance(clear_ch)
    return create_array_per_pair(
        clear_ch[:, :, 0],
        clear_var[:, :, 0]
    )

def get_brightness_to_variance(num_workers=None):
    """
    Parallel version of get_brightness_to_variance.
    """
    paths_dict = get_image_paths()
    paths = list(paths_dict['long_exp'])
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(_process_brightness_to_variance, paths),
            total=len(paths)
        ))
    return np.vstack(results)

# if __name__ == '__main__':
#
#     # Get brightness to variance difference
#     brightness_to_variance_difference = get_brightness_to_variance_difference()
#     brightness_to_variance_difference = average_y_per_x_binned(brightness_to_variance_difference, num_bins=100, threshold=10)
#     plt.plot(brightness_to_variance_difference[:, 0], brightness_to_variance_difference[:, 1])
#     plt.title("Brightness to Variance Difference")
#     plt.xlabel("Brightness")
#     plt.ylabel("Variance Difference")
#     plt.show()
#
#     # Get brightness to variance
#     brightness_to_variance = get_brightness_to_variance()
#     brightness_to_variance = average_y_per_x_binned(brightness_to_variance, num_bins=100, threshold=10)
#     plt.plot(brightness_to_variance[:, 0], brightness_to_variance[:, 1])
#     plt.title("Brightness to Variance")
#     plt.xlabel("Brightness")
#     plt.ylabel("Variance")
#     plt.show()