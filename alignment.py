import rawpy

import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
import os
from raw_utils import *


raw_dir = os.path.join('.', 'first-dataset', 'first-dataset-RAW')


def get_image_alignment_transform(img1, img2):
    orb = cv2.ORB_create(nfeatures=5000)
    # sift = cv2.SIFT.create()

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, mask=None)
    kp2, des2 = orb.detectAndCompute(img2, mask=None)
    # kp1, des1 = sift.detectAndCompute(img1, mask=None)
    # kp2, des2 = sift.detectAndCompute(img2, mask=None)
    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use RANSAC to find homography
    if len(matches) > 10:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        return H
    else:
        raise ValueError("Not enough matches found.")

def apply_transform(img1, img2, H):
    aligned_img = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
    return aligned_img

def align_and_crop_raw_images(path1, path2):
    raw1 = rawpy.imread(path1)
    raw2 = rawpy.imread(path2)

    packed1_scaled = pack_raw(raw1, normalize=True)
    packed2_scaled = pack_raw(raw2, normalize=True)
    packed1_unscaled = pack_raw(raw1, normalize=False)
    packed2_unscaled = pack_raw(raw2, normalize=False)

    aligned_channels = []
    projection_matrices = []
    
    # Calculate projections for each color channel individually
    for c in range(4):
        img1 = packed1_scaled[:, :, c]
        img2 = packed2_scaled[:, :, c]
        projection_matrix = get_image_alignment_transform(img1, img2)
        projection_matrices.append(projection_matrix)

    projection_matrix = np.stack(projection_matrices, axis=2).mean(axis=2)

    for c in range(4):
        img1 = packed1_unscaled[:, :, c]
        img2 = packed2_unscaled[:, :, c]
        aligned_ch = apply_transform(img1, img2, projection_matrix)
        aligned_channels.append(aligned_ch)

    aligned_channels = np.stack(aligned_channels, axis=2, dtype=np.float32)

    # TODO: check projection matrices
    # TODO: crop original image
    # TODO: intensity alignment

    cropped_aligned, cropped_original = crop_zero_sides(image1=aligned_channels, image2=packed1_unscaled)

    aligned_image_raw, aligned_image_array = unpack_raw(raw2, cropped_aligned)
    original_image_raw, original_image_array = unpack_raw(raw1, cropped_original)
    return_original = {"raw": original_image_raw, "mosaic_array": original_image_array, "channels": cropped_original}
    return_aligned = {"raw": aligned_image_raw, "mosaic_array": aligned_image_array, "chan nels": cropped_aligned}

    return return_original, return_aligned

def align_and_crop_multiple_raw_images(reference_path, other_paths):
    import rawpy

    def load_and_pack(path):
        raw = rawpy.imread(path)
        packed_scaled = pack_raw(raw, normalize=True)
        packed_unscaled = pack_raw(raw, normalize=False)
        return raw, packed_scaled, packed_unscaled

    def crop_common_nonzero_region(*images):
        assert all(img.shape[:2] == images[0].shape[:2] for img in images), "All images must have same size"
        print([img.shape for img in images])
        masks = [np.any(img != 0, axis=-1) if img.ndim == 3 else img != 0 for img in images]
        common_mask = np.logical_and.reduce(masks)
        coords = np.argwhere(common_mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0) + 1
        return tuple(img[y0:y1, x0:x1, ...] for img in images)

    # Load reference
    ref_raw, ref_scaled, ref_unscaled = load_and_pack(reference_path)

    aligned_images = [ref_unscaled]
    raw_objects = [ref_raw]
    unscaled_images = [ref_unscaled]

    # Align each other image to reference
    for path in other_paths:
        raw, scaled, unscaled = load_and_pack(path)
        raw_objects.append(raw)

        projection_matrices = []
        for c in range(4):
            proj = get_image_alignment_transform(ref_scaled[:, :, c], scaled[:, :, c])
            projection_matrices.append(proj)
        avg_proj = np.stack(projection_matrices, axis=2).mean(axis=2)

        aligned_channels = []
        for c in range(4):
            ch = apply_transform(unscaled[:, :, c], ref_unscaled[:, :, c], avg_proj)
            aligned_channels.append(ch)
        aligned_img = np.stack(aligned_channels, axis=2, dtype=np.float32)

        aligned_images.append(aligned_img)
        unscaled_images.append(unscaled)

    aligned_channels_ref = []
    # for c in range(4):
    #     aligned_ch = apply_transform(ref_unscaled[:, :, c], ref_unscaled[:, :, c], np.eye(3))
    #     aligned_channels_ref.append(aligned_ch)
    # ref_aligned = np.stack(aligned_channels_ref, axis=2, dtype=np.float32)
    # aligned_images = [ref_aligned] + aligned_images
    # Crop all aligned images + reference to common valid region
    cropped_images = crop_common_nonzero_region(*unscaled_images)
    cropped_aligned = crop_common_nonzero_region(*aligned_images)

    # First image is reference
    result = []
    ref_raw_unpacked, ref_array = unpack_raw(raw_objects[0], cropped_images[0])
    # result.append({"raw": ref_raw_unpacked, "mosaic_array": ref_array, "channels": cropped_images[0]})

    for i in range(len(cropped_aligned)):
        raw_unpacked, array = unpack_raw(raw_objects[i], cropped_aligned[i])
        result.append({"raw": raw_unpacked, "mosaic_array": array, "channels": cropped_aligned[i]})

    return result
def align_images_raw(path1, path2):
    """Aligns img2 to img1 channel by channel"""
    raw1 = rawpy.imread(path1)
    raw2 = rawpy.imread(path2)

    packed1_scaled = pack_raw(raw1, normalize=True)
    packed2_scaled = pack_raw(raw2, normalize=True)
    packed1_unscaled = pack_raw(raw1, normalize=False)
    packed2_unscaled = pack_raw(raw2, normalize=False)

    aligned_channels = []
    projection_matrices = []
    
    # Calculate projections for each color channel individually
    for c in range(4):
        img1 = packed1_scaled[:, :, c]
        img2 = packed2_scaled[:, :, c]
        projection_matrix = get_image_alignment_transform(img1, img2)
        projection_matrices.append(projection_matrix)

    projection_matrix = np.stack(projection_matrices, axis=2).mean(axis=2)

    for c in range(4):
        img1 = packed1_unscaled[:, :, c]
        img2 = packed2_unscaled[:, :, c]
        aligned_ch = apply_transform(img1, img2, projection_matrix)
        aligned_channels.append(aligned_ch)

    aligned_channels = np.stack(aligned_channels, axis=2, dtype=np.float32)

    # TODO: check projection matrices
    # TODO: crop original image
    # TODO: intensity alignment

    raw = rawpy.imread(path1)
    aligned_image_raw, aligned_image_array = unpack_raw(raw, aligned_channels)


    return aligned_image_raw, aligned_image_array, aligned_channels



def align_images_single_channel(img1, img2):
    """Aligns img2 to img1 using ORB feature matching and RANSAC."""
    orb = cv2.ORB_create(5000)

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # Sort by distance

    # Use RANSAC to find homography
    if len(matches) > 10:
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        # Warp img2 to align with img1 (preserving colors)
        aligned_img = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))
        
        return aligned_img, H
    else:
        raise ValueError("Not enough matches found.")

def grayscale_from_raw(raw):

    rgb = raw.postprocess()  # Process RAW file to RGB
    # image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    gray_downscaled = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return  gray_downscaled

def crop_zero_sides(image1, image2):
    """
    Crops the sides of the images where pixel values are zero in image1.
    The same crop is applied to image2 to maintain alignment.

    Args:
        image1 (numpy.ndarray): The first image (used to determine the crop).
        image2 (numpy.ndarray): The second image (to apply the same crop).

    Returns:
        cropped_image1 (numpy.ndarray): Cropped version of image1.
        cropped_image2 (numpy.ndarray): Cropped version of image2.
    """
    # Find non-zero rows and columns in image1
    non_zero_rows = np.any(image1 != 0, axis=(1, 2))
    non_zero_cols = np.any(image1 != 0, axis=(0, 2))

    # Determine the cropping bounds
    row_start, row_end = np.where(non_zero_rows)[0][[0, -1]]
    col_start, col_end = np.where(non_zero_cols)[0][[0, -1]]

    # Crop both images using the same bounds
    cropped_image1 = image1[row_start:row_end + 1, col_start:col_end + 1, :]
    cropped_image2 = image2[row_start:row_end + 1, col_start:col_end + 1, :]

    return cropped_image1, cropped_image2

def crop_common_nonzero_region(*images):
    """
    Crops all images to the region where all have non-zero pixels.

    Args:
        *images: Numpy arrays of shape (H, W, C) or (H, W)

    Returns:
        Tuple of cropped images.
    """
    assert all(img.shape[:2] == images[0].shape[:2] for img in images), "All images must have same spatial size"

    # Convert all images to mask of non-zero pixels
    masks = [np.any(img != 0, axis=-1) if img.ndim == 3 else img != 0 for img in images]

    # Find common valid region
    common_mask = np.logical_and.reduce(masks)

    # Get bounding box of common non-zero region
    coords = np.argwhere(common_mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1

    return tuple(img[y0:y1, x0:x1, ...] for img in images)

def homography_error_matrix(homographies):
    """
    Given a list of 3x3 homography matrices, computes an NxN error matrix where
    error_matrix[i, j] is the Frobenius norm of (I - H_j^-1 * H_i).

    :param homographies: List of N homography matrices (each 3x3).
    :return: NxN numpy array of error values.
    """
    N = len(homographies)
    error_mat = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N):
        H_i = homographies[i]
        for j in range(N):
            H_j = homographies[j]
            
            # Attempt to invert H_j. If singular, treat the error as infinite.
            try:
                H_j_inv = np.linalg.inv(H_j)
            except np.linalg.LinAlgError:
                error_mat[i, j] = np.inf
                continue
            
            # Combined transform from j to i
            H_combined = np.dot(H_j_inv, H_i)
            
            # Measure difference from the identity
            diff = np.eye(3) - H_combined
            error_mat[i, j] = np.linalg.norm(diff, ord='fro')
    
    return error_mat

def MSE_error_matrix(homographies):
    """
    Given a list of 3x3 homography matrices, computes an NxN error matrix where
    error_matrix[i, j] is the average L2 norm of (H_j - H_i).

    :param homographies: List of N homography matrices (each 3x3).
    :return: NxN numpy array of error values.
    """
    N = len(homographies)
    error_mat = np.zeros((N, N), dtype=np.float64)
    
    for i in range(N):
        H_i = homographies[i]
        for j in range(N):
            H_j = homographies[j]
            diff = H_i - H_j
            error_mat[i, j] = np.linalg.norm(diff, ord=2)
    
    return error_mat

    
    

