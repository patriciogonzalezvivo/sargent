import argparse
import os
import numpy as np
import cv2

# Facetracker
from utils.face_tracker import image_to_nodes, nodes_uvs, nodes_faces, nodes_colors, unwarp_texture
from utils.mesh import Mesh
from utils.material import Material


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run TEED model')

    # Data parameters
    parser.add_argument('--input',
                        type=str,
                        default='data',
                        help='the path to the directory with the input data for validation.')
    
    parser.add_argument('--extract_texture',
                        action='store_true',
                        help='whether to extract texture map from the input image.')

    parser.add_argument('--output',
                        type=str,
                        default='./',
                        help='the path to output the results.')

    args = parser.parse_args()
    return args


def run(input_image_path, output_path, args):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB) 

    points = image_to_nodes(image)
    if points is None:
        print(f"No face detected in {input_image_path}. Skipping.")
        return

    faces = nodes_faces();
    uvs = nodes_uvs();
    colors = nodes_colors(points, image)

    if args.extract_texture:
        texture = unwarp_texture(image, points)
        texture_path = os.path.splitext(output_path)[0] + '_texture.png'
        cv2.imwrite(texture_path, cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))

    # flip X and Y axist of points positions to match OBJ coordinate system
    points[:, 0] = -points[:, 0]
    points[:, 1] = -points[:, 1]

    mesh = Mesh()
    mesh.addVertices(points)
    mesh.addTexCoords(uvs)
    mesh.addTriangles(faces)
    mesh.addColors(colors)
    mesh.smoothNormals()
    mesh.toObj(output_path)

    print(f"Saved mesh to {output_path}")


if __name__ == '__main__':
    args = parse_args()
    
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_image_path = os.path.join(args.input, filename)
                output_path = os.path.join(args.output, os.path.splitext(filename)[0] + '.obj')
                run(input_image_path=input_image_path,
                    output_path=output_path,
                    args=args)
    else:
        run(input_image_path=args.input,
            output_path=os.path.join(args.output, 'output.obj'),
            args=args)