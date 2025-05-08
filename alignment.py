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

    # Detect keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, mask=None)
    kp2, des2 = orb.detectAndCompute(img2, mask=None)

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

def align_and_crop_raw_images(path1, path2, align_intensity=True):
    print(f"Reading {path1} and {path2}")
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

    cropped_aligned, cropped_original = crop_zero_sides(image1=aligned_channels, image2=packed1_unscaled)
    cropped_original = cropped_original.astype(np.float32)

    if align_intensity:
      cropped_aligned_mean = cropped_aligned.mean()
      cropped_original_mean = cropped_original.mean()
      global_mean = (cropped_aligned_mean + cropped_original_mean) / 2

      cropped_aligned = (cropped_aligned.astype(np.float32) - cropped_aligned_mean + global_mean).astype(np.uint16)
      cropped_original = (cropped_original.astype(np.float32) - cropped_original_mean + global_mean).astype(np.uint16)
    else:
      cropped_aligned = cropped_aligned.astype(np.uint16)
      cropped_original = cropped_original.astype(np.uint16)

    aligned_image_raw, aligned_image_array = unpack_raw(raw2, cropped_aligned)
    original_image_raw, original_image_array = unpack_raw(raw1, cropped_original)
    return_original = {"raw": original_image_raw, "mosaic_array": original_image_array, "channels": cropped_original}
    return_aligned = {"raw": aligned_image_raw, "mosaic_array": aligned_image_array, "channels": cropped_aligned}

    return return_original, return_aligned

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

    
    

