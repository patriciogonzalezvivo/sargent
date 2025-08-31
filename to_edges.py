import argparse
import os
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING']="0"

import cv2
import torch

# TEED (Tiny Edge-Enhancement Distillation) model
from utils.ted import TED
device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')
checkpoint_path = 'checkpoints/teed.pth'
model = None

# Edge to Vector
from utils.AxiSurface.AxiSurface import AxiSurface, Path, convert
from skimage.morphology import skeletonize

# Facetracker
from utils.face_tracker import image_to_guidelines


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
                        default='results',
                        help='the path to output the results.')

    parser.add_argument('--up_scale',
                        type=bool,
                        default=True, # for Upsale test set in 30%
                        help='True: up scale x1.5 test image')  # Just for test

    parser.add_argument('--img_width',
                        type=int,
                        default=512,
                        help='Image width for testing.')
    parser.add_argument('--img_height',
                        type=int,
                        default=512,
                        help='Image height for testing.')
    
    parser.add_argument('--use_gpu',type=int,
                        default=0, help='use GPU')
    
    parser.add_argument('--mean',
                        default=[104.007, 116.669, 122.679, 137.86],
                        type=float)
    
    parser.add_argument('--threshold',
                        type=float,
                        default=0.97,
                        help='The threshold to convert the probability map to binary map.')
    
    parser.add_argument('--scale',
                        type=float,
                        default=0.65,
                        help='scale the input image')
    
    parser.add_argument('--margin',
                        type=int,
                        default=10,
                        help='margin for the input image')
    
    parser.add_argument('--epsilon',
                        type=float,
                        default=0.0001,
                        help='epsilon factor for approxPolyDP')

    args = parser.parse_args()
    return args


def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def init_model():
    global model

    if model is not None:
        return model
    
    model = TED().to(device)

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def extract_contours(image, threshold=0.5, scale=1.0, translate=(0,0), epsilon_factor=0.0001):
    # convert image to gray if it is not
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    _, binary = cv2.threshold(gray, int(255 * threshold), 255, cv2.THRESH_BINARY)

    #invert binary image
    binary = cv2.bitwise_not(binary)

    skeleton = skeletonize(binary // 255)
    skeleton_display = (skeleton * 255).astype(np.uint8)

    # 2. Find Contours
    contours, _ = cv2.findContours(skeleton_display, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    vector_paths = []
    for contour in contours:
        # 3. Approximate Contours
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # 4. Scale and Translate
        points = [(point[0][0] * scale + translate[0], point[0][1] * scale + translate[1]) for point in approx]
        if len(points) > 1:
            vector_paths.append(points)

    return Path(vector_paths)


def extract_coorners(gray, scale=1.0, translate=(0,0)):
    gray_with = gray.shape[1]
    gray_height = gray.shape[0]
    return [(0*scale+translate[0],0*scale+translate[1]),
            (gray_with*scale+translate[0],0*scale+translate[1]),
            (0*scale+translate[0],gray_height*scale+translate[1]),
            (gray_with*scale+translate[0],gray_height*scale+translate[1])]


def image_to_edge(input_image, mean_bgr=[104.007, 116.669, 122.679, 137.86], resize_input=False):
    global model
    if model is None:
        model = init_model()
    
    if device.type == 'cuda':
        torch.cuda.synchronize()

    image = np.array(input_image, dtype=np.float32)

    image -= mean_bgr if len(mean_bgr) == 3 else mean_bgr[:3]
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image.copy()).float().unsqueeze(0).to(device)
        
    preds = model(image, single_test=resize_input)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    fuse_num = len(preds)

    # Fuse predictions
    fuse = None
    if fuse_num > 0:
        fuse = torch.sigmoid(preds[0]).cpu().detach().numpy()
        fuse = np.squeeze(fuse)
        fuse = np.uint8(image_normalization(fuse))
        fuse = cv2.bitwise_not(fuse)

        for i in range(1, fuse_num):
            tmp = torch.sigmoid(preds[i]).cpu().detach().numpy()
            tmp = np.squeeze(tmp)
            tmp = np.uint8(image_normalization(tmp))
            tmp = cv2.bitwise_not(tmp)
            if fuse is None:
                fuse = tmp
            else:
                fuse = np.maximum(fuse, tmp)

    torch.cuda.empty_cache()

    return fuse


def run(input_image_path, output_image_path, args):

    # prepare paths and parameters
    output_filename = output_image_path.split('.')[0]
    output_png_path = output_filename + '.png'
    output_svg_path = output_filename + '.svg'
    img_size = (args.img_height, args.img_width)
    
    # load image
    image = cv2.imread(input_image_path, cv2.IMREAD_COLOR)
    image_original = image.copy()
    image_original_size = [image_original.shape[0], image_original.shape[1]]

    # Pre-process image
    if args.up_scale:
        # For TEED BIPBRIlight Upscale
        image = cv2.resize(image,(0,0),fx=1.3,fy=1.3)

    if image.shape[0] < img_size[0] or image.shape[1] < img_size[1]:
        #TEED BIPED standard proposal if you want speed up the test, comment this block
        image = cv2.resize(image, (0, 0), fx=1.5, fy=1.5)

    # if it's too large, downscale the max dimension to the min of img_height, img_width
    if image.shape[0] > img_size[0] or image.shape[1] > img_size[1]:
        if image.shape[0] >= image.shape[1]:
            scale = img_size[0] / image.shape[0]
        else:
            scale = img_size[1] / image.shape[1]
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        
    # Make sure images and labels are divisible by 2^4=16
    if image.shape[0] % 8 != 0 or image.shape[1] % 8 != 0:
        img_width = ((image.shape[1] // 8) + 1) * 8
        img_height = ((image.shape[0] // 8) + 1) * 8
        image = cv2.resize(image, (img_width, img_height))

    img_height, img_width = image.shape[0], image.shape[1]

    print(f"{input_image_path}: {image_original_size} -> {(img_height, img_width)}")

    # Predict SotA edge detection
    edge = image_to_edge(image, mean_bgr=args.mean)
    
    # Thresholding to make the edge more clear
    threshold_value = int(256.0 * args.threshold)
    edge[edge > threshold_value] = 255.0
    edge[edge <= threshold_value] = 0.0

    # # Save edge map
    # cv2.imwrite(output_png_path, edge)

    margin = [args.margin, args.margin]
    marks = 5

    vector_paths = extract_contours(edge, scale=args.scale, translate=margin, epsilon_factor=args.epsilon)
    coorners = extract_coorners(edge, scale=args.scale, translate=margin)

    width = coorners[1][0] - coorners[0][0]
    height = coorners[3][1] - coorners[0][1]

    # add intermediate points between coorners 
    p1 = coorners[0]
    p2 = coorners[1]
    p3 = coorners[2]
    p4 = coorners[3]
    inter_num = 5
    for i in range(1, inter_num):
        coorners.append( (p1[0]+(p2[0]-p1[0])*i/inter_num, p1[1]+(p2[1]-p1[1])*i/inter_num) )
        coorners.append( (p3[0]+(p4[0]-p3[0])*i/inter_num, p3[1]+(p4[1]-p3[1])*i/inter_num) )
        coorners.append( (p1[0]+(p3[0]-p1[0])*i/inter_num, p1[1]+(p3[1]-p1[1])*i/inter_num) )
        coorners.append( (p2[0]+(p4[0]-p2[0])*i/inter_num, p2[1]+(p4[1]-p2[1])*i/inter_num) )

    guidelines = image_to_guidelines(image_original,
                                     scale=(width, height),
                                     translate=margin)

    axi = AxiSurface(size='12in x 16in')

    for coorner in coorners:
        axi.circle( center=coorner, radius=marks*0.5)
        axi.line( [coorner[0]-marks, coorner[1]], [coorner[0]+marks, coorner[1]])
        axi.line( [coorner[0], coorner[1]-marks], [coorner[0], coorner[1]+marks])

    axi.path( Path(vector_paths).getSimplify(2.0).getSorted())
    axi.path(Path( guidelines ).getSorted())

    axi.toSVG(output_svg_path)


if __name__ == '__main__':
    args = parse_args()
    
    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        for filename in os.listdir(args.input):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                input_image_path = os.path.join(args.input, filename)
                output_image_path = os.path.join(args.output, os.path.splitext(filename)[0] + '.png')
                run(input_image_path=input_image_path,
                    output_image_path=output_image_path,
                    args=args)
    else:
        run(input_image_path=args.input,
            output_image_path=args.output,
            args=args)
