# from __future__ import print_function, division

import torch
import numpy as np
from PIL import Image
import torch.utils.data as data
import os
import os.path

def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([float(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], float(val.split()[1])) for val in image_list]
    return images

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

def convert_to_vec_path(image_paths):
    vec_paths = []
    for path in image_paths:
        vec_path = path[0].replace('data/img', 'data/imgVec')
        vec_path = os.path.splitext(vec_path)[0] + '.npy'
        vec_paths.append(vec_path)
    return vec_paths

def convert_to_ast_vector_path(image_paths):
    ast_vector_paths = []
    for path in image_paths:
        ast_path = path[0].replace('data/img/grb_img', 'data/embedding')
        ast_path = os.path.splitext(ast_path)[0] + '.npy'
        ast_vector_paths.append(ast_path)
    return ast_vector_paths

class ImageList(object):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list, labels)
        vec_list = convert_to_vec_path(imgs)
        ast_vector_list = convert_to_ast_vector_path(imgs)
        if len(imgs) == 0:
            root = '../data/'
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                "Supported image extensions are: " + ",".join('png')))

        self.imgs = imgs
        self.vecs = vec_list
        self.ast_vectors = ast_vector_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        # Load the vector for the image
        img_vec = np.load(self.vecs[index])
        img_vec = torch.from_numpy(img_vec).float()

        # Load AST vector
        ast_vector = np.load(self.ast_vectors[index])
        ast_vector = torch.from_numpy(ast_vector).float()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, path, ast_vector, img_vec

    def __len__(self):
        return len(self.imgs)

# ... [rest of the code]



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
