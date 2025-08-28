# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F

import PIL.Image as pil_image

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
import trimesh


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

# Set device and dtype

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Run VGGT with a COLMAP scene bundle conversion")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--conf_thres_value", type=float, default=2.5, help="Confidence threshold value for depth filtering (wo BA)"
    )
    return parser.parse_args()


def run_VGGT(images, fixed_resolution: int = None):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

     # Run VGGT for camera and depth estimation
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    model = model.to(device)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(fixed_resolution, fixed_resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]

        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def get_image_path_list(scene_dir):
    image_dir = os.path.join(scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    image_path_list = sorted(image_path_list)
    return image_path_list


def to_Colmap(scene_dir, conf_thres_value):

    image_path_list = get_image_path_list(scene_dir)

    # Load images and original coordinates
    # Load Image in 1024, while running VGGT with 518
    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)

    # Run VGGT to estimate camera and depth
    # Run with 518x518 images
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(images, vggt_fixed_resolution)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # Save to COLMAP format
    max_points_for_colmap = 100000  # randomly sample 3D points
    image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
    num_frames, height, width, _ = points_3d.shape

    points_rgb = F.interpolate(
        images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    conf_mask = depth_conf >= conf_thres_value
    conf_mask_original = conf_mask.copy()
    print(conf_mask.shape, "is the shape of confidence mask")
    # at most writing 100000 3d points to colmap reconstruction object
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    print(conf_mask.sum(), "3D points are kept after confidence filtering")

    # get depth min and max after confidence filtering
    depth_values = depth_map[conf_mask]
    min_depth = float(depth_values.min())
    max_depth = float(depth_values.max())

    # Export depth map images
    depth_dir = os.path.join(scene_dir, "depth")
    os.makedirs(depth_dir, exist_ok=True)
    for i, depth in enumerate(depth_map):

        # filter depth with confidence mask
        depth = depth.copy()
        depth[~conf_mask_original[i]] = max_depth + 1.0  # set low-confidence depth to max_depth + 1

        # rotate depth to original image size
        depth = np.rot90(depth, k=3)  # rotate to match original orientation

        # place depth in original image size
        orig_h, orig_w = original_coords[i, -2:].cpu().numpy().astype(int)

        # calculate the scale factor used to resize original image to square image
        scale_factor = vggt_fixed_resolution / max(orig_h, orig_w)
        # calculate the padding added to the original image to make it square
        # after resizing, the padding should also be scaled
        pad_h = (vggt_fixed_resolution - int(orig_h * scale_factor)) // 2
        pad_w = (vggt_fixed_resolution - int(orig_w * scale_factor)) // 2

        depth = depth[pad_h: pad_h + int(orig_h * scale_factor), pad_w: pad_w + int(orig_w * scale_factor)]
        
        # normalize depth for visualization
        depth_img = (1.0-(depth - min_depth) / (max_depth - min_depth))
        depth_img = np.clip(depth_img, 0.0, 1.0)
        depth_img = (depth_img * 255.0).astype(np.uint8)
        depth_img = np.repeat(depth_img, 3, axis=-1)
        depth_img = pil_image.fromarray(depth_img)

        basename = os.path.basename(image_path_list[i])
        basename = os.path.splitext(basename)[0]
        depth_img.save(os.path.join(depth_dir, f"{basename}.png"))

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format")
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=False,
        camera_type="SIMPLE_PINHOLE",
    )

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        image_path_list,
        original_coords.cpu().numpy(),
        img_size=vggt_fixed_resolution,
        shift_point2d_to_original_res=True,
    )

    print(f"Saving reconstruction to {scene_dir}/sparse")
    sparse_reconstruction_dir = os.path.join(scene_dir, "sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(scene_dir, "sparse/points.ply"))

    return True


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = os.path.basename(image_paths[pyimageid - 1])

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[pyimageid - 1, :2]

            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

    return reconstruction


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():

        # Set seed for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)  # for multi-GPU
        print(f"Setting seed as: {args.seed}")

        to_Colmap(args.scene_dir, args.conf_thres_value)


# Work in Progress (WIP)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
