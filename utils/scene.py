import os
import copy
import numpy as np

from .colmap import read_model

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
