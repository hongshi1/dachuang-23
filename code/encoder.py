import torch.nn as nn
import torchvision.models as models

#我写的图像编码器,将图像编码为100维向量


class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # Load the pretrained ResNet-50 model
        resnet = models.resnet50(pretrained=True)
        # Remove the last fully connected layer (classifier)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Add custom layers to get 100-dimensional output
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Res18Encoder(nn.Module):
    def __init__(self):
        super(Res18Encoder, self).__init__()
        # Load the pretrained ResNet-18 model
        resnet = models.resnet18(pretrained=True)
        # Remove the last fully connected layer (classifier)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Add custom layers to get 100-dimensional output
        # Notice the input dimension for the first Linear layer is now 512, not 2048
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 100)
        )

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
