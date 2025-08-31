import argparse
import os
import numpy as np
import cv2

# Facetracker
from utils.face_tracker import image_to_nodes, nodes_uvs, nodes_faces
import trimesh


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run TEED model')

    # Data parameters
    parser.add_argument('--input',
                        type=str,
                        default='data',
                        help='the path to the directory with the input data for validation.')

    parser.add_argument('--output',
                        type=str,
                        default='./',
                        help='the path to output the results.')

    args = parser.parse_args()
    return args


def run(input_image_path, output_image_path, args):
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB) 

    points = image_to_nodes(image)
    if points is None:
        print(f"No face detected in {input_image_path}. Skipping.")
        return

    # faces, edges = nodes_faces_edges();
    faces = nodes_faces();
    uvs = nodes_uvs();

    mesh = trimesh.Trimesh(vertices=points, faces=faces, uvs=uvs, process=False)
    # add uvs 
    texture_image = cv2.cvtColor(cv2.imread("uvs.png"), cv2.COLOR_BGR2RGB) 
    material = trimesh.visual.texture.SimpleMaterial(image=texture_image)
    mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uvs, image=texture_image, material=material)

    mesh.export(output_image_path, file_type='ply')
    print(f"Saved textured mesh to {output_image_path}")


if __name__ == '__main__':
    args = parse_args()
    
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_image_path = os.path.join(args.input, filename)
                output_image_path = os.path.join(args.output, os.path.splitext(filename)[0] + '.ply')
                run(input_image_path=input_image_path,
                    output_image_path=output_image_path,
                    args=args)
    else:
        run(input_image_path=args.input,
            output_image_path=os.path.join(args.output, 'output.ply'),
            args=args)