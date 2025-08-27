"""
Hello, welcome on board,
"""
from __future__ import print_function

import argparse
import os
import time
import numpy as np
os.environ['CUDA_LAUNCH_BLOCKING']="0"

import cv2
import torch
from torch.utils.data import Dataset, DataLoader

from ted import TED # TEED architecture


class TestDataset(Dataset):
    def __init__(self,
                 data_root='data',
                 img_height=512,
                 img_width=512,
                 up_scale=False,
                 mean_test=[104.007, 116.669, 122.679, 137.86],
                 ):

        self.data_root = data_root
        self.up_scale = up_scale
        self.mean_bgr = mean_test if len(mean_test) == 3 else mean_test[:3]
        self.img_height = img_height
        self.img_width = img_width
        self.data_index = self._build_index()

    def _build_index(self):
        sample_indices = []
        # for single image testing
        images_path = os.listdir(self.data_root)
        labels_path = None
        sample_indices = [images_path, labels_path]
        return sample_indices

    def __len__(self):
        return len(self.data_index[0])

    def __getitem__(self, idx):
        # get data sample
        # image_path, label_path = self.data_index[idx]
        if self.data_index[1] is None:
            image_path = self.data_index[0][idx] if len(self.data_index[0]) > 1 else self.data_index[0][idx - 1]
        else:
            image_path = self.data_index[idx][0]

        img_name = os.path.basename(image_path)
        file_name = os.path.splitext(img_name)[0] + ".png"

        img_dir = self.data_root

        # load data
        image = cv2.imread(os.path.join(img_dir, image_path), cv2.IMREAD_COLOR)
        label = None

        im_shape = [image.shape[0], image.shape[1]]
        image, label = self.transform(img=image, gt=label)

        return dict(images=image, labels=label, file_names=file_name, image_shape=im_shape)

    def transform(self, img, gt):
        # gt[gt< 51] = 0 # test without gt discrimination
        # up scale test image
        if self.up_scale:
            # For TEED BIPBRIlight Upscale
            img = cv2.resize(img,(0,0),fx=1.3,fy=1.3)

        if img.shape[0] < 512 or img.shape[1] < 512:
            #TEED BIPED standard proposal if you want speed up the test, comment this block
            img = cv2.resize(img, (0, 0), fx=1.5, fy=1.5)

        # if it's too large, downscale the max dimension to the min of img_height, img_width
        if img.shape[0] > self.img_height or img.shape[1] > self.img_width:
            if img.shape[0] >= img.shape[1]:
                scale = self.img_height / img.shape[0]
            else:
                scale = self.img_width / img.shape[1]
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            
        # Make sure images and labels are divisible by 2^4=16
        if img.shape[0] % 8 != 0 or img.shape[1] % 8 != 0:
            img_width = ((img.shape[1] // 8) + 1) * 8
            img_height = ((img.shape[0] // 8) + 1) * 8
            img = cv2.resize(img, (img_width, img_height))
        else:
            pass

        img = np.array(img, dtype=np.float32)

        img -= self.mean_bgr
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img.copy()).float()

        gt = np.array(gt, dtype=np.float32)
        if len(gt.shape) == 3:
            gt = gt[:, :, 0]
        gt /= 255.
        gt = torch.from_numpy(np.array([gt])).float()

        return img, gt


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


def save_image_batch_to_disk(tensor, output_dir, file_names, img_shape=None):
    os.makedirs(output_dir, exist_ok=True)
    
    tensor2=None
    tmp_img2 = None

    # 255.0 * (1.0 - em_a)
    edge_maps = []
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)
    # print(f"tensor shape: {tensor.shape}")

    image_shape = [x.cpu().detach().numpy() for x in img_shape]
    # (H, W) -> (W, H)
    image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

    assert len(image_shape) == len(file_names)

    idx = 0
    for i_shape, file_name in zip(image_shape, file_names):
        tmp = tensor[:, idx, ...]
        tmp2 = tensor2[:, idx, ...] if tensor2 is not None else None
        # tmp = np.transpose(np.squeeze(tmp), [0, 1, 2])
        tmp = np.squeeze(tmp)
        tmp2 = np.squeeze(tmp2) if tensor2 is not None else None

        # Iterate our all 7 NN outputs for a particular image
        preds = []
        fuse_num = tmp.shape[0]-1
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img = np.uint8(image_normalization(tmp_img))
            tmp_img = cv2.bitwise_not(tmp_img)
            # tmp_img[tmp_img < 0.0] = 0.0
            # tmp_img = 255.0 * (1.0 - tmp_img)
            if tmp2 is not None:
                tmp_img2 = tmp2[i]
                tmp_img2 = np.uint8(image_normalization(tmp_img2))
                tmp_img2 = cv2.bitwise_not(tmp_img2)

            # Resize prediction to match input image size
            if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
                tmp_img2 = cv2.resize(tmp_img2, (i_shape[0], i_shape[1])) if tmp2 is not None else None

            if tmp2 is not None:
                tmp_mask = np.logical_and(tmp_img>128,tmp_img2<128)
                tmp_img= np.where(tmp_mask, tmp_img2, tmp_img)
                preds.append(tmp_img)

            else:
                preds.append(tmp_img)

            if i == fuse_num:
                # print('fuse num',tmp.shape[0], fuse_num, i)
                fuse = tmp_img
                fuse = fuse.astype(np.uint8)
                if tmp_img2 is not None:
                    fuse2 = tmp_img2
                    fuse2 = fuse2.astype(np.uint8)
                    # fuse = fuse-fuse2
                    fuse_mask=np.logical_and(fuse>128,fuse2<128)
                    fuse = np.where(fuse_mask,fuse2, fuse)

                    # print(fuse.shape, fuse_mask.shape)

        # Save predicted edge maps
        average = np.array(preds, dtype=np.float32)
        average = np.uint8(np.mean(average, axis=0))
        output_file_name = os.path.join(output_dir, file_name)
        cv2.imwrite(output_file_name, fuse)

        idx += 1


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='TEED model')
    parser.add_argument('--choose_test_data',
                        type=int,
                        default=-1,     # UDED=15
                        help='Choose a dataset for testing: 0 - 15')

    # Data parameters
    parser.add_argument('--input_dir',
                        type=str,
                        default='data',
                        help='the path to the directory with the input data for validation.')
    parser.add_argument('--output_dir',
                        type=str,
                        default='results',
                        help='the path to output the results.')

    parser.add_argument('--test_list',
                        type=str,
                        default=None,
                        help='Dataset sample indices list.')
    parser.add_argument('--up_scale',
                        type=bool,
                        default=False, # for Upsale test set in 30%
                        help='True: up scale x1.5 test image')  # Just for test

    parser.add_argument('--checkpoint_data',
                        type=str,
                        default='checkpoints/teed.pth',# 37 for biped 60 MDBD
                        help='Checkpoint path.')
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
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        help='The number of workers for the dataloaders.')
    parser.add_argument('--mean_test',
                        default=[104.007, 116.669, 122.679, 137.86],
                        type=float)

    args = parser.parse_args()
    return args


def run_TEED(checkpoint_path, input_dir, output_dir, img_size=(512, 512), mean_test=[104.007, 116.669, 122.679, 137.86], num_workers=8, resize_input=False, up_scale=False):
    # torch.autograd.set_detect_anomaly(True)

    # Get computing device
    device = torch.device('cpu' if torch.cuda.device_count() == 0 else 'cuda')

    # Instantiate model and move it to the computing device
    model = TED().to(device)

    dataset = TestDataset(  input_dir,
                            img_width=img_size[0],
                            img_height=img_size[1],
                            up_scale=up_scale,
                            mean_test=mean_test)
    
    print(dataset)
                              
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=num_workers)


    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint filte note found: {checkpoint_path}")
    
    print(f"Restoring weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        print(f"output_dir: {output_dir}")
        for _, sample_batched in enumerate(dataloader):
            images = sample_batched['images'].to(device)
            file_names = sample_batched['file_names']
            image_shape = sample_batched['image_shape']

            print(f"{file_names}: {images.shape}")
            if device.type == 'cuda':
                torch.cuda.synchronize()

            preds = model(images, single_test=resize_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()

            save_image_batch_to_disk(preds,
                                     output_dir, # output_dir
                                     file_names,
                                     image_shape)
            
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # os.system(" ".join(command))
    args = parse_args()
    run_TEED(   checkpoint_path=args.checkpoint_data,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                img_size=(args.img_width, args.img_height),
                up_scale=args.up_scale,
                mean_test=args.mean_test)
