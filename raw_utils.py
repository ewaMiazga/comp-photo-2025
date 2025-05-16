import numpy as np
from scipy.ndimage.filters import convolve


def pack_raw(raw, black_level: int = 512, white_level: int = 16383, normalize: bool = False):
    """Packs Bayer image to 4 channels"""
    img = np.expand_dims(raw, axis=2)

    out = np.concatenate((img[0::2, 0::2], # R
                          img[0::2, 1::2], # G
                          img[1::2, 1::2], # B
                          img[1::2, 0::2], # G
                          ), axis=2)
    
    if normalize:
        out = (out - black_level) / (white_level - black_level)

    return out

def unpack_raw(packed_img, black_level=512, white_level=16383, denormalize=False):
    H = 2 * packed_img.shape[0]
    W = 2 * packed_img.shape[1]

    if denormalize:
        packed_img = packed_img / 255.0 * (white_level - black_level) + black_level

    out = np.zeros((H, W), dtype=np.uint16 if denormalize else np.float32)
    out[0:H:2, 0:W:2] = packed_img[:, :, 0]
    out[0:H:2, 1:W:2] = packed_img[:, :, 1]
    out[1:H:2, 1:W:2] = packed_img[:, :, 2]
    out[1:H:2, 0:W:2] = packed_img[:, :, 3]

    return out

def white_balance_gray_world(raw):
    red = raw[0::2, 0::2]
    green = (raw[0::2, 1::2] + raw[1::2, 0::2]) / 2 # Take the average of green_top and green_bottom
    blue = raw[1::2, 1::2] #

    mu_red = np.average(red)
    mu_green = np.average(green) # calculate the average of the green color
    mu_blue = np.average(blue) # calculate the average of the blue color

    whitebalance_coefs = [mu_green/mu_red, 1.0, mu_green/mu_blue]
    print("Gray World White Balancing Coefficients: ", whitebalance_coefs)

    whitebalanced_mosaic = np.array(raw)
    whitebalanced_mosaic[0::2, 0::2] *= whitebalance_coefs[0]     # scale red
    whitebalanced_mosaic[0::2, 1::2] *= whitebalance_coefs[1]     # scale the top green
    whitebalanced_mosaic[1::2, 0::2] *= whitebalance_coefs[1]     # scale the bottom green
    whitebalanced_mosaic[1::2, 1::2] *= whitebalance_coefs[2]     # scale blue

    return whitebalanced_mosaic

def demosaic_bilinear(raw):
    H, W = raw.shape
    channels = {channel: np.zeros((H, W)) for channel in 'RGB'}
    for channel, (y, x) in zip('RGGB', [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = raw[y::2, x::2]

    R_mosaic, G_mosaic, B_mosaic = tuple(channels[c] for c in 'RGB')

    # convolution kernels
    # green
    H_G = np.asarray(
        [[0.0, 1.0, 0.0],
        [1.0, 4.0, 1.0],
        [0.0, 1.0, 0.0]]) / 4.0
    # red/blue
    H_RB = np.asarray(
        [[1.0, 2.0, 1.0],
        [2.0, 4.0, 2.0],
        [1.0, 2.0, 1.0]]) / 4.0

    R = convolve(R_mosaic, H_RB) # Apply the H_RB filter on R_mosaic
    G = convolve(G_mosaic, H_G) # Apply the H_G filter on G_mosaic
    B = convolve(B_mosaic, H_RB) # Apply the H_RB filter on B_mosaic
    rgb_image = np.dstack((R,G,B)) # Stack R, G, B into a 3 channel image

    return rgb_image

