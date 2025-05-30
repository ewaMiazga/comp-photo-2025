from utils.raw_utils import unpack_raw, demosaic_bilinear
import numpy as np
class PostProcessor:
    def __init__(self, gray_world_constants, gamma=2.2):
        self.gray_world_constants = gray_world_constants
        self.gamma = gamma

    def postprocess(self, image, gray_world_constants=None):
        rgb = demosaic_bilinear(unpack_raw(image))

        scales = self.gray_world_constants if gray_world_constants is None else gray_world_constants
        # apply scales
        rgb = rgb * scales[None,None,:]

        rgb = np.clip(rgb, 0, 1)
        # Apply gamma correction
        rgb = np.power(rgb, 1 / self.gamma)
        # Normalize to [0, 1]
        # rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
        return rgb

def get_gray_world_constants(image):
    """
    Calculate gray world constants for white balancing.

    Args:
        image (numpy.ndarray): Input image in RGB format.

    Returns:
        tuple: Mean values for each channel (R, G, B).
    """
    demosaiced = demosaic_bilinear(unpack_raw(image))
    mu = demosaiced.mean(axis=(0, 1))  # [μ_R, μ_G, μ_B]
    mu_gray = mu.mean()           # gray reference
    scales = mu_gray / mu          # [s_R, s_G, s_B]
    return scales