import argparse
import os

import cv2
import numpy as np
import torch

# from moge.model.v1 import MoGeModel
from moge.model.v2 import MoGeModel # Let's try MoGe-2

from utils.mesh import Mesh

device = torch.device("cuda")

model = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run MoGe model')

    # Data parameters
    parser.add_argument('--input',
                        type=str,
                        default='data',
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--output',
                        type=str,
                        default='depth',
                        help='the path to output the results.')

    args = parser.parse_args()
    return args


def init_model():
    global model

    if model is not None:
        return model

    # Load the model from huggingface hub (or load from local).
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
    return model


def to_depth(input_image_path, output_image_path, args):
    global model
    if model is None:
        model = init_model()

    # Read the input image and convert to tensor (3, H, W) with RGB values normalized to [0, 1]
    input_image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)                       
    input_image = torch.tensor(input_image / 255, dtype=torch.float32, device=device).permute(2, 0, 1)    

    # Infer 
    output = model.infer(input_image)

    intrinsics = output['intrinsics'].cpu().numpy()
    mask = output['mask'].cpu().numpy()
    depthmap = output['depth'].cpu().numpy()
    normalmap = output['normal'].cpu().numpy()

    v_mask = mask.reshape(-1) > 0.5
    vertices = output['points'].cpu().numpy().reshape(-1, 3)
     # get colors and normals for the valid points
    v_colors = input_image.cpu().numpy().transpose(1, 2, 0).reshape(-1, 3) * 255
    v_normals = normalmap.reshape(-1, 3)

    
    # extract 2D uvs by normalizing x,y by the image width and height
    h, w = input_image.shape[1:3]

    print("width x height", w, "x", h)
    print("masks", mask.shape)
    print("depthmap", depthmap.shape)
    print("normalmap", normalmap.shape)

    v_texcoords = []
    for u in range(w):
        for v in range(h):
            v_texcoords.append([u / w, v / h])
    v_texcoords = np.array(v_texcoords)

    vertices = vertices[v_mask]
    v_colors = v_colors[v_mask]
    v_normals = v_normals[v_mask]
    v_texcoords = v_texcoords[v_mask]

    mesh = Mesh()
    mesh.addVertices(vertices)
    mesh.addColors(v_colors)
    mesh.addNormals(v_normals)
    mesh.addTexCoords(v_texcoords)
    
    # Ensure the output path has a proper extension
    if not output_image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        output_image_path = output_image_path + '.png'
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_image_path) if os.path.dirname(output_image_path) else '.', exist_ok=True)
    
    output_basename = os.path.splitext(output_image_path)[0]
    mesh.toPly(output_basename + '.ply')

    # save depth map
    depth_min = np.min(depthmap[mask > 0.5])
    depth_max = np.max(depthmap[mask > 0.5])
    depthmap = np.clip(depthmap, depth_min, depth_max)
    depthmap = 1.0 - (depthmap - depth_min) / (depth_max - depth_min + 1e-8)
    depthmap = (depthmap * 255).astype(np.uint8)
    cv2.imwrite(output_image_path, depthmap)


    # save normal map
    normalmap = ((normalmap * 0.5) + 0.5) * 255
    normalmap = normalmap.astype(np.uint8)
    cv2.imwrite(output_basename + '_normal.png', cv2.cvtColor(normalmap, cv2.COLOR_RGB2BGR))

    # save mask
    mask = (mask * 255).astype(np.uint8)
    cv2.imwrite(output_basename + '_mask.png', mask)


if __name__ == '__main__':
    args = parse_args()
    
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_image_path = os.path.join(args.input, filename)
                output_image_path = os.path.join(args.output, os.path.splitext(filename)[0] + '.png')
                os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                to_depth(   input_image_path=input_image_path,
                            output_image_path=output_image_path,
                            args=args)
    else:
        # Ensure output has proper extension for single file case
        if not args.output.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            args.output = args.output + '.png'
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
        
        to_depth(   input_image_path=args.input,
                    output_image_path=args.output,
                    args=args)
