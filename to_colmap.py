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

import argparse

import torch
import torch.nn.functional as F

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track

from utils.mesh import Mesh
import PIL.Image as pil_image
import cv2
from utils.face_tracker import image_to_nodes, nodes_faces, nodes_uvs, unwarp_texture

# Set device and dtype

device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser(description="Run VGGT with a COLMAP scene bundle conversion")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--confidence_threshold", type=float, default=1.25, help="Confidence threshold value for depth filtering")
    parser.add_argument("--export_depth", action="store_true", default=False, help="Whether to export depth maps")
    parser.add_argument("--export_mask_texture", action="store_true", default=False, help="Whether to export mask texture maps")
    return parser.parse_args()


def run_VGGT(images, fixed_resolution: int = None, predictions_file: str = None):

    if predictions_file and os.path.exists(predictions_file):
        print(f"predictions.npz already exists, loading...")
        predictions = np.load(predictions_file)
        extrinsic = predictions["extrinsic"]
        intrinsic = predictions["intrinsic"]
        depth_map = predictions["depth_map"]
        depth_conf = predictions["depth_conf"]
        return extrinsic, intrinsic, depth_map, depth_conf
    
    else:
            
        # Input checks:
        # Ensure the input is a 4D tensor with shape [B, 3, H, W]
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

            # # Predict Point Maps
            # point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

        extrinsic = extrinsic.squeeze(0).cpu().numpy()
        intrinsic = intrinsic.squeeze(0).cpu().numpy()
        depth_map = depth_map.squeeze(0).cpu().numpy()
        depth_conf = depth_conf.squeeze(0).cpu().numpy()

        # Save predicions to numpy files
        if predictions_file:
            predictions = {
                "extrinsic": extrinsic,
                "intrinsic": intrinsic,
                "depth_map": depth_map,
                "depth_conf": depth_conf
            }
            np.savez_compressed(predictions_file, **predictions)
            print(f"Saved predictions to {predictions_file}")

        return extrinsic, intrinsic, depth_map, depth_conf


def get_image_path_list(scene_dir):
    image_dir = os.path.join(scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    image_path_list = sorted(image_path_list)
    return image_path_list


def to_colmap(scene_dir, confidence_threshold: float = 2.5, vggt_fixed_resolution: int = 518, img_load_resolution: int = 1024, max_points_for_colmap: int = 100000):
    image_path_list = get_image_path_list(scene_dir)

    # Load images as 1024x1024 square images, while preserving original coordinates
    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)

    predictions_file = os.path.join(scene_dir, "predictions.npz")
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(images, vggt_fixed_resolution, predictions_file)

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # Save to COLMAP format
    image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
    num_frames, height, width, _ = points_3d.shape
    print(f"Number of images: {num_frames}, Image size: {width}x{height}")

    # 3D points are in 518x518 resolution, so scale rgb accordingly and get rgb values
    points_rgb = F.interpolate(images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False)
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    # (S, H, W, 3), with x, y coordinates and frame indices
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    conf_mask = depth_conf >= confidence_threshold
    conf_mask_original = conf_mask.copy()

    # at most writing 100000 3d points to colmap reconstruction object
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    # get depth min and max after confidence filtering
    depth_values = depth_map[conf_mask]
    min_depth = float(depth_values.min())
    max_depth = float(depth_values.max())

    min_depth_confidence = float(depth_conf[conf_mask_original].min())
    max_depth_confidence = float(depth_conf[conf_mask_original].max())
    print(f"Depth range: {min_depth:.4f} - {max_depth:.4f}")
    print(f"Confidence range: {min_depth_confidence:.4f} - {max_depth_confidence:.4f}")

    # Export depth map images
    if args.export_depth:
        depth_dir = os.path.join(scene_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

    mask_dir = os.path.join(scene_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    faces = nodes_faces()
    uvs = nodes_uvs()

    agregate_masks_node_positions = []
    agregate_masks_node_colors = []
    agregate_masks_normalized_weights = []

    for i, depth in enumerate(depth_map):
        image_path = image_path_list[i]

        # place depth in original image size
        orig_h, orig_w = original_coords[i, -2:].cpu().numpy().astype(int)

        img = cv2.cvtColor(cv2.imread(image_path_list[i]), cv2.COLOR_BGR2RGB) 
        mask_nodes = image_to_nodes(img)
        if mask_nodes is not None:
            mask_obj_path = os.path.join(mask_dir, f"{os.path.basename(image_path_list[i]).split('.')[0]}.obj")

            # Get original image dimensions for coordinate transformation
            top_left = original_coords[i, :2].cpu().numpy()  # padding offset in processed image
            scale_factor = vggt_fixed_resolution / max(orig_h, orig_w)  # scale used to resize to 518x518
            
            # Convert normalized MediaPipe coordinates to VGGT image coordinates (518x518)
            # MediaPipe x,y are normalized [0,1], so scale to image dimensions
            mask_nodes_image = mask_nodes.copy()
            mask_nodes_image[:, 0] *= orig_w * scale_factor  # scale x to resized image width
            mask_nodes_image[:, 1] *= orig_h * scale_factor  # scale y to resized image height
            
            # Add padding offset to account for centering in 518x518 square
            pad_h = (vggt_fixed_resolution - int(orig_h * scale_factor)) // 2
            pad_w = (vggt_fixed_resolution - int(orig_w * scale_factor)) // 2
            mask_nodes_image[:, 0] += pad_w
            mask_nodes_image[:, 1] += pad_h
            
            # Sample depth from VGGT depth map at face landmark positions
            # Clamp coordinates to valid range
            u_coords = np.clip(mask_nodes_image[:, 0].astype(int), 0, vggt_fixed_resolution - 1)
            v_coords = np.clip(mask_nodes_image[:, 1].astype(int), 0, vggt_fixed_resolution - 1)
            
            # Get depth values from VGGT depth map
            sampled_depths = depth[v_coords, u_coords]
            
            # Replace MediaPipe z with VGGT depth for consistent scale
            # Ensure sampled_depths is 1D array to match the slice shape
            if sampled_depths.ndim > 1:
                sampled_depths = sampled_depths.flatten()
            mask_nodes_image[:, 2] = sampled_depths
            
            # Transform from image coordinates to camera coordinates using intrinsics
            fx, fy = intrinsic[i, 0, 0], intrinsic[i, 1, 1]
            cx, cy = intrinsic[i, 0, 2], intrinsic[i, 1, 2]
            
            # Unproject to 3D camera coordinates
            mask_nodes_camera = np.zeros_like(mask_nodes_image)
            mask_nodes_camera[:, 0] = (mask_nodes_image[:, 0] - cx) * mask_nodes_image[:, 2] / fx
            mask_nodes_camera[:, 1] = (mask_nodes_image[:, 1] - cy) * mask_nodes_image[:, 2] / fy
            mask_nodes_camera[:, 2] = mask_nodes_image[:, 2]
            
            # Transform from camera coordinates to world coordinates using extrinsics
            # extrinsic transforms world to camera, so we need the inverse
            R = extrinsic[i, :3, :3]
            t = extrinsic[i, :3, 3]
            
            # Transform: world = R.T @ (camera - t)
            # First subtract translation, then apply rotation transpose
            mask_nodes_world = (R.T @ (mask_nodes_camera - t).T).T

            # extract vertex colors from point_rgb
            mask_nodes_colors = points_rgb[i][v_coords, u_coords]
            agregate_masks_node_colors.append(mask_nodes_colors)

            if args.export_mask_texture:
                # Export texture map for the mask
                texture = unwarp_texture(img, mask_nodes)
                texture_path = os.path.splitext(mask_obj_path)[0] + '_texture.png'
                cv2.imwrite(texture_path, cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))

            mask = Mesh()
            mask.addVertices(mask_nodes_world)
            mask.addTexCoords(uvs)
            mask.addTriangles(faces)
            mask.addColors(mask_nodes_colors)
            mask.smoothNormals()
            mask.toObj(mask_obj_path)

            # Get confidence values at face landmark positions
            landmark_depth_confidence = depth_conf[i][v_coords, u_coords]

            agregate_masks_node_positions.append(mask_nodes_world)
            normalized_node_weights = (landmark_depth_confidence - min_depth_confidence) / (max_depth_confidence - min_depth_confidence + 1e-8)
            agregate_masks_normalized_weights.append(normalized_node_weights)


        if args.export_depth:
            depth = depth.copy()
            # filter depth with confidence mask
            depth[~conf_mask_original[i]] = max_depth + 1.0  # set low-confidence depth to max_depth + 1

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

            basename = os.path.basename(image_path)
            basename = os.path.splitext(basename)[0]
            depth_img.save(os.path.join(depth_dir, f"{basename}.png"))


    best_mask_world_points = []
    best_mask_colors = []
    if len(agregate_masks_node_positions) == 1:
        best_mask_world_points = agregate_masks_node_positions[0]
        best_mask_colors = agregate_masks_node_colors[0]

    elif len(agregate_masks_node_positions) > 1:
        
        #  convert agregate_masks_node_positions to numpy array
        agregate_masks_node_positions = np.array(agregate_masks_node_positions)  # (N_images, N_landmarks, 3)
        agregate_masks_normalized_weights = np.array(agregate_masks_normalized_weights)  # (N_images, N_landmarks)

        # return the image index number with highest normalize weight for each landmark
        best_image_indices = np.argmax(agregate_masks_normalized_weights, axis=0)
        best_masks_normalized_weights = agregate_masks_normalized_weights[best_image_indices, np.arange(len(best_image_indices))]

        #  Average the position of each node so the higher the weight the more it influences the final position BUT first filter very low weights
        for landmark_idx in range(agregate_masks_node_positions.shape[1]):
            weights = agregate_masks_normalized_weights[:, landmark_idx]
            positions = agregate_masks_node_positions[:, landmark_idx, :]

            # filter very low weights
            weight_threshold = (args.confidence_threshold - min_depth_confidence) / (max_depth_confidence - min_depth_confidence + 1e-8)
            valid = weights >= weight_threshold
            if np.sum(valid) == 0:
                # if no valid weights, just take the position from the image with highest weight
                best_image_idx = np.argmax(weights)
                best_mask_world_points.append(positions[best_image_idx])
                best_mask_colors.append(agregate_masks_node_colors[best_image_idx][landmark_idx])
            else:
                weights = weights[valid]
                positions = positions[valid]
                weighted_position = np.average(positions, axis=0, weights=weights)
                best_mask_world_points.append(weighted_position)
                # also average colors
                colors = np.array(agregate_masks_node_colors)[valid, landmark_idx, :]
                weighted_color = np.average(colors, axis=0, weights=weights)
                best_mask_colors.append(weighted_color)

        best_mask_world_points = np.array(best_mask_world_points)
        best_mask_colors = np.array(best_mask_colors).astype(np.uint8)

        # filter those points with very low weights
        weight_threshold = (args.confidence_threshold - min_depth_confidence) / (max_depth_confidence - min_depth_confidence + 1e-8)
        # weight_threshold = 0.1; #max(0.1, weight_threshold)  # at least 0.1
        weight_mask = best_masks_normalized_weights >= weight_threshold

        # filter the best mask world points
        filtered_mask_world_points = best_mask_world_points[weight_mask]

        filtered_mask_colors = best_mask_colors[weight_mask]

        # reorganize faces and uvs based on the weight mask
        old_to_new_idx = np.full(len(best_mask_world_points), -1, dtype=int)
        old_to_new_idx[weight_mask] = np.arange(np.sum(weight_mask))

        # Filter faces - keep only faces where ALL vertices are in the confident set
        valid_faces = []
        for face in faces:
            # Check if all vertices in this face are in the confident set
            if all(weight_mask[v] for v in face if v < len(weight_mask)):
                # Remap vertex indices to new filtered array
                new_face = [old_to_new_idx[v] for v in face if v < len(weight_mask)]
                if all(idx >= 0 for idx in new_face):  # All vertices successfully mapped
                    valid_faces.append(new_face)
        
        faces_filtered = np.array(valid_faces) if valid_faces else np.empty((0, 3), dtype=int)
        uvs_filtered = uvs[weight_mask] if len(uvs) == len(best_mask_world_points) else uvs[:len(filtered_mask_world_points)]

        # Only create mesh if we have enough confident landmarks and faces
        agregated_mask_mesh = Mesh()
        agregated_mask_mesh.addVertices(filtered_mask_world_points)
        agregated_mask_mesh.addTexCoords(uvs_filtered)
        agregated_mask_mesh.addTriangles(faces_filtered)
        agregated_mask_mesh.addColors(filtered_mask_colors)
        # agregated_mask_mesh.toPly(os.path.join(scene_dir, f"mask.ply"))
        agregated_mask_mesh.smoothNormals()
        agregated_mask_mesh.toObj(os.path.join(scene_dir, f"mask.obj"))


    # filter points
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
    point_cloud = Mesh()
    point_cloud.addVertices(points_3d)
    point_cloud.addColors(points_rgb)
    point_cloud.toPly(os.path.join(scene_dir, "sparse/points.ply"))

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

        to_colmap(args.scene_dir, args.confidence_threshold)


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
