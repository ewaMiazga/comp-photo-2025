

import rawpy
import numpy as np
def pack_raw(raw: rawpy.rawpy, black_level: int =512, white_level: int =16383, normalize:bool=False):
    """Packs Bayer image to 4 channels"""

    img = raw.raw_image_visible
    img = np.expand_dims(img, axis=2)

    out = np.concatenate((img[0::2, 0::2], # R
                          img[0::2, 1::2], # G
                          img[1::2, 1::2], # B
                          img[1::2, 0::2], # G
                          ), axis=2)
    
    if normalize:
        out = (out - black_level) / (white_level - black_level) * 255.0
        out = out.astype(np.uint8)

    return out

def unpack_raw(raw, packed_img, black_level=512, white_level=16383, denormalize=False):
    
    H = 2 * packed_img.shape[0]
    W = 2 * packed_img.shape[1]

    if denormalize:
        packed_img = packed_img / 255.0 * (white_level - black_level) + black_level
        packed_img = packed_img.astype(np.uint16)

    out = np.zeros((H, W), dtype=np.uint16)
    out[0:H:2, 0:W:2] = packed_img[:, :, 0]
    out[0:H:2, 1:W:2] = packed_img[:, :, 1]
    out[1:H:2, 1:W:2] = packed_img[:, :, 2]
    out[1:H:2, 0:W:2] = packed_img[:, :, 3]
    
    raw.raw_image_visible[0:H:2, 0:W:2] = packed_img[:, :, 0]
    raw.raw_image_visible[0:H:2, 1:W:2] = packed_img[:, :, 1]
    raw.raw_image_visible[1:H:2, 1:W:2] = packed_img[:, :, 2]
    raw.raw_image_visible[1:H:2, 0:W:2] = packed_img[:, :, 3]  
    raw.raw_image_visible[H:,:] = packed_img.max()
    raw.raw_image_visible[:,W:] = packed_img.max()
    return raw, out

