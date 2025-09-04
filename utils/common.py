import os
import glob
import copy

import numpy as np
import cv2
import PIL.Image as pil_image

import tqdm

from .mesh import Mesh
from .colmap import read_model

def get_image_path_list(scene_dir):
    image_dir = os.path.join(scene_dir, "images")
    image_path_list = glob.glob(os.path.join(image_dir, "*"))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    image_path_list = sorted(image_path_list)
    return image_path_list

def assign_uv(points_3d, texture_size=512):
    points_uvs = np.zeros_like(points_3d[:, :2])
    # create a new set of uvs that fit in the 512x512 texture
    for i in range(texture_size * texture_size):
        u = (i % texture_size) / (texture_size - 1)
        v = (i // texture_size) / (texture_size - 1)
        points_uvs[i] = np.array([u, v])
        if i >= points_3d.shape[0]:
            break

    return points_uvs


def estimate_normals(points_3d):
     # try to compute normals
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        # estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(10)

        return np.asarray(pcd.normals)
    except ImportError:
        print("Open3D is not installed, skipping normal estimation.")
    return None


def export_mesh_reconstruction(scene_dir, points_3d, points_rgb):
    print("Running mesh reconstruction with Open3D")
    try:
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(points_rgb / 255.0)

        # voxel downsample
        pcd = pcd.voxel_down_sample(voxel_size=0.005)
        pcd.remove_non_finite_points()
        pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        # estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(10)

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]

        # smooth mesh surface
        mesh = mesh.filter_smooth_simple(number_of_iterations=3)

        # remove low density vertices
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        mesh.remove_unreferenced_vertices()

        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(scene_dir, "sparse/points_mesh.ply"), mesh)

    except ImportError:
        print("Open3D is not installed, skipping mesh reconstruction.")


def export_as_texture(scene_dir, texture_size, points_3d, points_rgb, points_uvs, points_normals=None):
    texture_xyz_path = os.path.join(scene_dir, "sparse/points_xyz.png")
    texture_xyz_32bit_path = os.path.join(scene_dir, "sparse/points_xyz.exr")
    texture_rgb_path = os.path.join(scene_dir, "sparse/points_rgb.png")
    texture_normals_path = os.path.join(scene_dir, "sparse/points_normals.png")

    # calculate the bounding box of the point cloud
    min_bound = points_3d.min(axis=0)
    max_bound = points_3d.max(axis=0)
    bbox = np.array([min_bound, max_bound])
    print(f"Point cloud bounding box size: {max_bound - min_bound}")

    # normalize their postions 
    points_3d_normalized = (points_3d - min_bound) / (max_bound - min_bound + 1e-8)

    texture_xyz = np.zeros((texture_size, texture_size, 4), dtype=np.uint16)
    texture_xyz_32bit = np.zeros((texture_size, texture_size, 3), dtype=np.float32)
    texture_rgb = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
    texture_normals = np.zeros((texture_size, texture_size, 4), dtype=np.uint8)
    
    # create a 2 512x512 texture and save their values as 16bit png where X is R, Y is G and Z is B and the color values
    for i in range(points_3d_normalized.shape[0]):
        u = int(points_uvs[i, 0] * 511)
        v = int(points_uvs[i, 1] * 511)

        texture_xyz[v, u, 0] = int(points_3d_normalized[i, 0] * 65535)
        texture_xyz[v, u, 1] = int(points_3d_normalized[i, 1] * 65535)
        texture_xyz[v, u, 2] = int(points_3d_normalized[i, 2] * 65535)
        texture_xyz[v, u, 3] = 65535

        texture_xyz_32bit[v, u, 0] = points_3d[i, 0]
        texture_xyz_32bit[v, u, 1] = points_3d[i, 1]
        texture_xyz_32bit[v, u, 2] = points_3d[i, 2]

        if points_normals is not None:
            texture_normals[v, u, 0] = int((points_normals[i, 0] * 0.5 + 0.5) * 255)
            texture_normals[v, u, 1] = int((points_normals[i, 1] * 0.5 + 0.5) * 255)
            texture_normals[v, u, 2] = int((points_normals[i, 2] * 0.5 + 0.5) * 255)
            texture_normals[v, u, 3] = 255

        texture_rgb[v, u, 0] = int(points_rgb[i, 0])
        texture_rgb[v, u, 1] = int(points_rgb[i, 1])
        texture_rgb[v, u, 2] = int(points_rgb[i, 2])
        texture_rgb[v, u, 3] = 255

        points_uvs[i, 0] = u
        points_uvs[i, 1] = v
    
    cv2.imwrite(texture_xyz_path, cv2.cvtColor(texture_xyz, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(texture_rgb_path, cv2.cvtColor(texture_rgb, cv2.COLOR_RGBA2BGRA))
    if points_normals is not None:
        cv2.imwrite(texture_normals_path, cv2.cvtColor(texture_normals, cv2.COLOR_RGBA2BGRA))
    try:
        import OpenEXR
        import Imath

        header = OpenEXR.Header(texture_size, texture_size)
        header['channels'] = {
            'R': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'G': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT)),
            'B': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
        }


        exr_file = OpenEXR.OutputFile(texture_xyz_32bit_path, header)
        exr_file.writePixels({
            'R': texture_xyz_32bit[:, :, 0].astype(np.float32).tobytes(),
            'G': texture_xyz_32bit[:, :, 1].astype(np.float32).tobytes(),
            'B': texture_xyz_32bit[:, :, 2].astype(np.float32).tobytes()
        })
        exr_file.close()

    except ImportError:
        print("OpenEXR not installed, cannot save 32bit EXR texture")


def convert_from_orthographic_to_perspective(mask_nodes, orig_size, vggt_fixed_resolution, 
                                             depth, intrinsic, extrinsic):
    orig_w, orig_h = orig_size
    scale_factor = vggt_fixed_resolution / max(orig_h, orig_w)

    # Convert 2D mask nodes from orthographic to perspective projection
    # This is a placeholder function and needs to be implemented

    # MediaPipe landmarks are in normalized coordinates [0,1] with orthographic projection
    # We need to convert them to perspective projection in VGGT camera space
    
    # First, convert normalized MediaPipe coordinates to original image pixel coordinates
    mask_nodes_pixel = mask_nodes.copy()
    mask_nodes_pixel[:, 0] *= orig_w  # scale x to original image width
    mask_nodes_pixel[:, 1] *= orig_h  # scale y to original image height
    
    # MediaPipe z is in relative depth units, we need to scale it to match VGGT depth scale
    # For now, we'll sample VGGT depth at the 2D landmark positions
    
    # Convert original image coordinates to VGGT processing coordinates (518x518)
    mask_nodes_vggt = mask_nodes_pixel.copy()
    mask_nodes_vggt[:, 0] *= scale_factor  # scale x to resized image width  
    mask_nodes_vggt[:, 1] *= scale_factor  # scale y to resized image height
    
    # Add padding offset to account for centering in 518x518 square
    pad_h = (vggt_fixed_resolution - int(orig_h * scale_factor)) // 2
    pad_w = (vggt_fixed_resolution - int(orig_w * scale_factor)) // 2
    mask_nodes_vggt[:, 0] += pad_w
    mask_nodes_vggt[:, 1] += pad_h
    
    # Sample depth from VGGT depth map at face landmark positions
    # Clamp coordinates to valid range
    u_coords = np.clip(mask_nodes_vggt[:, 0].astype(int), 0, vggt_fixed_resolution - 1)
    v_coords = np.clip(mask_nodes_vggt[:, 1].astype(int), 0, vggt_fixed_resolution - 1)
    
    # Get depth values from VGGT depth map
    sampled_depths = depth[v_coords, u_coords]
    
    # Ensure sampled_depths is 1D array to match the slice shape
    if sampled_depths.ndim > 1:
        sampled_depths = sampled_depths.flatten()
    
    # Now we have the correct depth scale from VGGT, but we need to properly handle
    # the orthographic to perspective conversion
    
    # MediaPipe provides orthographic projection, so we need to convert to perspective
    # The key insight: MediaPipe landmarks represent normalized face coordinates
    # We need to project them onto the depth plane determined by VGGT
    
    # Use VGGT camera parameters to unproject the 2D landmarks to 3D
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    # Unproject VGGT image coordinates to 3D camera coordinates
    mask_nodes_camera = np.zeros_like(mask_nodes_vggt)
    mask_nodes_camera[:, 0] = (mask_nodes_vggt[:, 0] - cx) * sampled_depths / fx
    mask_nodes_camera[:, 1] = (mask_nodes_vggt[:, 1] - cy) * sampled_depths / fy
    mask_nodes_camera[:, 2] = sampled_depths
    
    # However, we need to account for the fact that MediaPipe landmarks have relative
    # positioning that may not exactly match the perspective projection
    # Apply a correction based on MediaPipe's orthographic nature
    
    # Get the median depth to use as reference depth plane
    median_depth = np.median(sampled_depths)
    
    # MediaPipe landmarks are relative to face center, so we need to adjust
    # the x,y coordinates based on the relative MediaPipe z-coordinate
    for j in range(len(mask_nodes)):
        # MediaPipe z represents relative depth from face plane
        # Scale the x,y displacement based on this relative depth
        relative_z = mask_nodes[j, 2]  # MediaPipe relative z
        depth_scale = 1.0 + (relative_z - 0.5) * 0.1  # Subtle depth-based scaling
        
        # Apply depth-based correction to the unprojected coordinates
        mask_nodes_camera[j, 0] *= depth_scale
        mask_nodes_camera[j, 1] *= depth_scale
    
    # Transform from camera coordinates to world coordinates using extrinsics
    # extrinsic transforms world to camera, so we need the inverse
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    
    # Transform: world = R.T @ (camera - t)
    # First subtract translation, then apply rotation transpose
    mask_nodes_world = (R.T @ (mask_nodes_camera - t).T).T
    return mask_nodes_world, u_coords, v_coords


def agregate_mask_texture(agregate_masks_texture):
    # all textures have the same size and an alpha channel,
    # average all pixels with alpha > 0 together 
    # any value with alpha > 0 is considered valid
    
    texture_size = agregate_masks_texture[0].shape[:2]
    aggregated_texture = np.zeros((texture_size[0], texture_size[1], 4), dtype=np.float32)
    count_texture = np.zeros((texture_size[0], texture_size[1]), dtype=np.float32)

    for texture in agregate_masks_texture:
        alpha_mask = texture[:, :, 3] > 0
        aggregated_texture[alpha_mask] += texture[alpha_mask]
        count_texture[alpha_mask] += 1.0

    # Avoid division by zero
    count_texture[count_texture == 0] = 1.0
    averaged_texture = (aggregated_texture / count_texture[:, :, None]).astype(np.uint8)

    # if the alpha is > 0 set alpha to 255
    averaged_texture[averaged_texture[:, :, 3] > 0, 3] = 255
    return averaged_texture


def agregate_mask_mesh( agregate_masks_node_positions, agregate_masks_node_colors, agregate_masks_normalized_weights,
                        weight_threshold,
                        faces, uvs, points_3d, points_rgb, 
                        subdivide_mask):

     # Initialize best mask points and colors
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

        if subdivide_mask > 0:
            # subdivide mesh at each vertex of points_3d and points_rgb (with threshold=0.01)
            total_points = len(points_3d)
            indices = np.arange(total_points)

            # mix the indices
            np.random.shuffle(indices)

            # add tqdm progress bar
            counter = 0
            with tqdm.tqdm(total=subdivide_mask) as pbar:
                for i in indices:
                    point = points_3d[i]
                    color = points_rgb[i]

                    rta = agregated_mask_mesh.subdivideAt(point, color=color, threshold=0.01)

                    if len(rta) > 0:
                        counter += 1
                        pbar.update(1)

                    if i >= total_points or counter >= total_points or counter >= subdivide_mask:
                        break

                if counter % 10 == 0:
                    agregated_mask_mesh.isotropicRemesh(max_iterations=1)

        # smooth normals
        agregated_mask_mesh.smoothNormals()

        return agregated_mask_mesh


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


def colmap_to_csv(scene_dir, output_csv):
    sparse_folder = os.path.join(scene_dir, "sparse")
    rgba_folder = os.path.join(scene_dir, "images")
    expected_N = len(os.listdir(rgba_folder))

    if os.path.exists(os.path.join(sparse_folder, "0")):
        sparse_folder = os.path.join(sparse_folder, "0")

    cameras, model_images, points3D = read_model(path=sparse_folder)

    keys = list(model_images.keys())
    keys = sorted(keys, key=lambda x: model_images[x].name)

    if expected_N is not None:
        assert len(keys) == expected_N

    camkey = model_images[keys[0]].camera_id

    cam = cameras[camkey]
    params = cam.params

    model = cam.model
    width = params[0]
    height = params[1]

    focal_length = None
    principal_point = None
    if cam.model == "SIMPLE_PINHOLE":
        focal_length = params[0]
        principal_point = params[:2].tolist()
    elif cam.model == "PINHOLE":
        focal_length = params[0]
        principal_point = params[:2].tolist()
    field_of_view = 2 * np.arctan(0.5 * params[1] / focal_length) * 180 / np.pi

    # assert cam.model in ["RADIAL", "SIMPLE_RADIAL"]
    K = np.array([[params[0], 0.0, params[1]],
                  [0.0, params[0], params[2]],
                  [0.0,       0.0,       1.0]])

    Rs = np.stack([model_images[k].qvec2rotmat() for k in keys])
    ts = np.stack([model_images[k].tvec for k in keys])

    N = Rs.shape[0]
    params = params[:3][None].repeat(N, axis=0)
    Rs = Rs.reshape(N, 9)

    lines = np.concatenate((params, Rs, ts), axis=1)

    np.savetxt(output_csv, lines, delimiter=",", newline="\n",
               header=(",".join(["f", "ox", "oy"]+
                                [f"R[{i//3},{i%3}]" for i in range(9)]+
                                [f"t[{i}]" for i in range(3)])))
