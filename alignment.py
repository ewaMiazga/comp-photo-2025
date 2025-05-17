import rawpy

import cv2
import numpy as np
import rawpy
import matplotlib.pyplot as plt
import os
from raw_utils import *


def get_image_alignment_transform(img1, img2):
    orb = cv2.ORB_create(nfeatures=5000)
    # sift = cv2.SIFT.create()

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

def align_and_crop_raw_images(path1, path2):
    print(f"Reading {path1} and {path2}")
    raw1 = rawpy.imread(path1).raw_image_visible
    raw2 = rawpy.imread(path2).raw_image_visible

    packed1 = pack_raw(raw1, normalize=True).astype(np.float32)
    packed2 = pack_raw(raw2, normalize=True).astype(np.float32)

    # Calculate projection matrix for green channel and apply it
    try: 
        projection_matrix = get_image_alignment_transform(
            (packed1[:, :, 1] * 255).astype(np.uint8),
            (packed2[:, :, 1] * 255).astype(np.uint8),
        )
    except Exception as e:
        print(f"Exception when getting transform: {str(e)}")
        return None

    aligned_channels = [
        apply_transform(packed1[:, :, c], packed2[:, :, c], projection_matrix) for c in range(4)
    ]
    aligned_channels = np.stack(aligned_channels, axis=2, dtype=np.float32)

    cropped_aligned, cropped_original = crop_zero_sides(image1=aligned_channels, image2=packed1)

    unpacked_aligned = unpack_raw(cropped_aligned)
    unpacked_original = unpack_raw(cropped_original)

    result_aligned = {"raw": unpacked_aligned, "mosaic": cropped_aligned, "rgb": demosaic_bilinear(unpacked_aligned)}
    result_original = {"raw": unpacked_original, "mosaic": cropped_original,  "rgb": demosaic_bilinear(unpacked_original)}

    return result_original, result_aligned


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

def crop_zero_sides(image1, image2, shift=64):
    """
    Crops the sides of the images
    """

    row_start, row_end = shift, -shift
    col_start, col_end = shift, -shift

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

    
    

