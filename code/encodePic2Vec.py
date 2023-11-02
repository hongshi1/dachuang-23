import os
import glob
import torch
import random
from torchvision import transforms, models
from PIL import Image, ImageFilter
import numpy as np

# Gaussian blur augmentation class
class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# Define the complex augmentation
def get_complex_augmentation():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    return transforms.Compose(augmentation)

def process_images_updated(model, source_dir='../data/img/', target_dir='../data/imgVec/'):
    # 设置图像预处理
    transform = get_complex_augmentation()

    # 遍历源目录下的所有.png文件
    for img_path in glob.glob(os.path.join(source_dir, '**', '*.png'), recursive=True):
        # 读取图像
        image = Image.open(img_path).convert("RGB")
        image = transform(image).unsqueeze(0)

        # 使用模型处理图像
        with torch.no_grad():
            vec = model(image).numpy()

        # 保存向量到目标目录
        rel_path = os.path.relpath(img_path, source_dir)
        save_path = os.path.join(target_dir, os.path.splitext(rel_path)[0] + '.npy')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, vec)

# 加载模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(in_features=2048, out_features=128)

# 加载检查点
checkpoint_path = '../model/checkpoint_0160.pth.tar'
if os.path.isfile(checkpoint_path):
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # rename moco pre-trained keys
    state_dict = checkpoint["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        # retain only encoder_q up to before the embedding layer
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            # remove prefix
            new_key = k[len("module.encoder_q."):]
            new_state_dict[new_key] = v

    # Load the new state dict
    model.load_state_dict(new_state_dict, strict=False)
    print("=> loaded pre-trained model from '{}'".format(checkpoint_path))
else:
    print("=> no checkpoint found at '{}'".format(checkpoint_path))

model = model.to(device)
model.eval()

# 开始处理
process_images_updated(model)
