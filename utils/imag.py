
import numpy as np
from PIL import Image as pil_image

def save_depthmaps(depth, min_depth, max_depth, conf_mask_original, depth_path, orig_size, vggt_fixed_resolution):
    orig_h, orig_w = orig_size

    # filter depth with confidence mask
    depth[~conf_mask_original] = max_depth + 1.0  # set low-confidence depth to max_depth + 1

    # calculate the scale factor used to resize original image to square image
    scale_factor = vggt_fixed_resolution / max(orig_h, orig_w)
    # calculate the padding added to the original image to make it square
    # after resizing, the padding should also be scaled
    pad_h = (vggt_fixed_resolution - int(orig_h * scale_factor)) // 2
    pad_w = (vggt_fixed_resolution - int(orig_w * scale_factor)) // 2

    depth = depth[pad_w: pad_w + int(orig_w * scale_factor), pad_h: pad_h + int(orig_h * scale_factor)]
    
    # normalize depth for visualization
    depth_img = (1.0-(depth - min_depth) / (max_depth - min_depth))
    depth_img = np.clip(depth_img, 0.0, 1.0)
    depth_img = (depth_img * 255.0).astype(np.uint8)
    depth_img = np.repeat(depth_img, 3, axis=-1)
    depth_img = pil_image.fromarray(depth_img)

    depth_img.save(depth_path)


def average_images(images):
    # all textures have the same size and an alpha channel,
    # average all pixels with alpha > 0 together 
    # any value with alpha > 0 is considered valid
    
    images_size = images[0].shape[:2]
    agregated = np.zeros((images_size[0], images_size[1], 4), dtype=np.float32)
    count = np.zeros((images_size[0], images_size[1]), dtype=np.float32)

    for texture in images:
        alpha_mask = texture[:, :, 3] > 0
        agregated[alpha_mask] += texture[alpha_mask]
        count[alpha_mask] += 1.0

    # Avoid division by zero
    count[count == 0] = 1.0
    averaged = (agregated / count[:, :, None]).astype(np.uint8)

    # if the alpha is > 0 set alpha to 255
    averaged[averaged[:, :, 3] > 0, 3] = 255
    return averaged