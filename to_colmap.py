# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import random
import numpy as np
import glob
import os

import argparse

import cv2
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

from utils.fs import get_image_path_list
from utils.mesh import Mesh
from utils.imag import save_depthmaps, average_images
from utils.geom import convert_from_orthographic_to_perspective, assign_uv, estimate_normals, pcl_to_textures, mesh_reconstruction, agregate_mask_mesh
from utils.scene import colmap_to_csv, rename_colmap_recons_and_rescale_camera
from utils.face_tracker import image_to_nodes, nodes_faces, nodes_uvs, unwarp_texture

# Set device and dtype
device = "cuda" if torch.cuda.is_available() else "cpu"
texture_size = 512

def parse_args():
    parser = argparse.ArgumentParser(description="Run VGGT with a COLMAP scene bundle conversion")
    parser.add_argument("--scene_dir", type=str, required=True, help="Directory containing the scene images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--confidence_threshold", type=float, default=1.25, help="Confidence threshold value for depth filtering")
    parser.add_argument("--export_depth", action="store_true", default=False, help="Whether to export depth maps")
    parser.add_argument("--export_mask_texture", action="store_true", default=False, help="Whether to export mask texture maps")
    parser.add_argument("--subdivide_mask", type=int, default=0, help="Amount of times to subdivide the mask mesh")
    parser.add_argument("--replace_folder", type=str, default=None, help="Folder to replace existing images")
    parser.add_argument("--mesh_reconstruction", action="store_true", default=False, help="Whether to run mesh reconstruction with Open3D")
    parser.add_argument("--export_pcl_as_texture", action="store_true", default=False, help="Whether to export point cloud as textured mesh")
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


def to_colmap(scene_dir, confidence_threshold: float = 2.5, vggt_fixed_resolution: int = 518, img_load_resolution: int = 1024, max_points_for_colmap: int = texture_size * texture_size):
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
    agregate_masks_texture = []

    for i, depth in enumerate(depth_map):
        image_path = image_path_list[i]

        # place depth in original image size
        orig_h, orig_w = original_coords[i, -2:].cpu().numpy().astype(int)

        image_path = image_path_list[i]
        
        img = cv2.cvtColor(cv2.imread(image_path_list[i]), cv2.COLOR_BGR2RGB) 
        mask_nodes = image_to_nodes(img)
        if mask_nodes is not None:
            mask_obj_path = os.path.join(mask_dir, f"{os.path.basename(image_path_list[i]).split('.')[0]}.obj")

            mask_nodes_world, u_coords, v_coords = convert_from_orthographic_to_perspective(mask_nodes, (orig_w, orig_h), vggt_fixed_resolution, depth, intrinsic[i], extrinsic[i])
            agregate_masks_node_positions.append(mask_nodes_world)

            # extract vertex colors from point_rgb
            mask_nodes_colors = points_rgb[i][v_coords, u_coords]
            agregate_masks_node_colors.append(mask_nodes_colors)

            landmark_depth_confidence = depth_conf[i][v_coords, u_coords]
            normalized_node_weights = (landmark_depth_confidence - min_depth_confidence) / (max_depth_confidence - min_depth_confidence + 1e-8)
            agregate_masks_normalized_weights.append(normalized_node_weights)

            if args.export_mask_texture:
                # Export texture map for the mask
                texture = unwarp_texture(img, mask_nodes, filter_occluded=True)
                texture_path = os.path.splitext(mask_obj_path)[0] + '_texture.png'
                cv2.imwrite(texture_path, cv2.cvtColor(texture, cv2.COLOR_RGBA2BGRA))
                agregate_masks_texture.append(texture)

            mask = Mesh()
            mask.addVertices(mask_nodes_world)
            mask.addTexCoords(uvs)
            mask.addTriangles(faces)
            mask.addColors(mask_nodes_colors)
            mask.smoothNormals()
            mask.toObj(mask_obj_path)
            print(f"Saved mask mesh to {mask_obj_path}")
            
        if args.export_depth:
            basename = os.path.basename(image_path)
            basename = os.path.splitext(basename)[0]
            depth_path = os.path.join(depth_dir, f"{basename}.png")
            save_depthmaps(depth.copy(), min_depth, max_depth, conf_mask_original[i], depth_path, (orig_h, orig_w), vggt_fixed_resolution)

    if args.replace_folder:
        # reload points_rgb with images from the replace folder that match the base name (not the extension)
        replace_image_paths = glob.glob(os.path.join(args.replace_folder, "*"))
        replace_image_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in replace_image_paths}
        for i, image_path in enumerate(image_path_list):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            if base_name in replace_image_dict:
                replace_img = cv2.cvtColor(cv2.imread(replace_image_dict[base_name]), cv2.COLOR_BGR2RGB)
                replace_img = cv2.resize(replace_img, (vggt_fixed_resolution, vggt_fixed_resolution))
                points_rgb[i] = replace_img
                print(f"Replaced colors for {base_name} from {replace_image_dict[base_name]}")


    # filter points
    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    # assing some random uvs to the points
    points_uvs = assign_uv(points_3d, texture_size)
    points_normals = estimate_normals(points_3d)

    # if open3d is installed, use it to estimate normals and reconstruct mesh
    if args.mesh_reconstruction:
        mesh_reconstruction(scene_dir, points_3d, points_rgb)
        
    # Agregate mask textures
    if args.export_mask_texture and len(agregate_masks_texture) > 0:
        averaged_texture = average_images(agregate_masks_texture)
        texture_path = os.path.join(scene_dir, "mask_texture.png")
        cv2.imwrite(texture_path, cv2.cvtColor(averaged_texture, cv2.COLOR_RGBA2BGRA))
        print(f"Saved aggregated mask texture to {texture_path}")

    # Agregate mask mesh
    weight_threshold = (args.confidence_threshold - min_depth_confidence) / (max_depth_confidence - min_depth_confidence + 1e-8)
    agregated_mask_mesh = agregate_mask_mesh(
                            agregate_masks_node_positions, agregate_masks_node_colors, agregate_masks_normalized_weights,
                            weight_threshold,
                            faces, uvs, points_3d, points_rgb, 
                            args.subdivide_mask
                        )
    agregated_mask_mesh.toPly(os.path.join(scene_dir, f"mask.ply"))
    print(f"Saved aggregated mask mesh to {os.path.join(scene_dir, f'mask.ply')}")

    if args.export_pcl_as_texture:
        pcl_to_textures(scene_dir, texture_size, points_3d, points_rgb, points_uvs, points_normals)

        
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
    # reconstruction.write_text(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    point_cloud = Mesh()
    point_cloud.addVertices(points_3d)
    point_cloud.addColors(points_rgb)
    point_cloud.addTexCoords(points_uvs)
    point_cloud.toPly(os.path.join(scene_dir, "sparse/points.ply"))

    return True

  
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
        colmap_to_csv(args.scene_dir, os.path.join(args.scene_dir, "cameras.csv"))


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
