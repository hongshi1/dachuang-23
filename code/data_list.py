# from __future__ import print_function, division

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path

#数据读取加载器


def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], float(val.split()[1])) for val in image_list]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:  # 以二进制格式打开一个文件用于只读
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)

def convert_to_ast_vector_path(image_paths):
    ast_vector_paths = []
    for path in image_paths:
        ast_path = path.replace('data/img/grb_img', 'data/embedding')
        ast_path = os.path.splitext(ast_path)[0] + '.npy'
        ast_vector_paths.append(ast_path)
    return ast_vector_paths


class ImageList(object):
    """
    A generic data loader where the images are arranged in a specific way.
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        # Convert image paths to AST vector paths
        ast_vector_list = convert_to_ast_vector_path(image_list)

        # Make dataset for images
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            root = '../data/'
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                'png')))

        self.imgs = imgs
        self.ast_vectors = ast_vector_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, ast_vector, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        # Load AST vector
        ast_vector = np.load(self.ast_vectors[index])

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img,target,path, torch.from_numpy(ast_vector).float()

    def __len__(self):
        return len(self.imgs)


def ClassSamplingImageList(image_list, transform, return_keys=False):
    data = open(image_list).readlines()
    label_dict = {}
    for line in data:
        label_dict[int(line.split()[1])] = []
    for line in data:
        label_dict[int(line.split()[1])].append(line)
    all_image_list = {}
    for i in label_dict.keys():
        all_image_list[i] = ImageList(label_dict[i], transform=transform)
    if return_keys:
        return all_image_list, label_dict.keys()
    else:
        return all_image_list
