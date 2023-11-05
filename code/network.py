import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
from torch.autograd import Function
import sklearn
import sklearn.preprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time


#各种各样的网络文件
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        # print(x.size())
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out

class NormalAlex(nn.Module):
    def __init__(self):
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        # end = time.perf_counter()
        return x

    def output_num(self):
        return self.__in_features


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class SilenceLayer(torch.autograd.Function):
    def __init__(self):
        pass

    def forward(self, input):
        return input * 1.0

    def backward(self, gradOutput):
        return 0 * gradOutput


# convnet without the last layer
class AlexNetFc(nn.Module):
    def __init__(self):
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i), model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

        self.atten = Self_Attn(256, 'relu')

    def forward(self, x):
        x = self.features(x)
        x = self.atten(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        # end = time.clock()
        end = time.perf_counter()
        return x

    def output_num(self):
        return self.__in_features


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
                self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, queries):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Scaled dot-product attention
        attention = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        attention = attention / (self.embed_size ** (1 / 2))
        attention = torch.nn.functional.softmax(attention, dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class AttentionModel(nn.Module):
    def __init__(self, input_dim=100, embed_size=20, heads=2):
        super(AttentionModel, self).__init__()

        self.attention = SelfAttention(embed_size=embed_size, heads=heads)

        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, embed_size)

        self.__in_features = embed_size

    def forward(self, x):
        # Transform input to "sequence" form
        x = x.view(x.size(0), 10, -1)

        # Apply self attention
        x = self.attention(x, x, x)

        # Flatten the output for fully connected layer
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

    def output_num(self):
        return self.__in_features

class ResNet18Fc(nn.Module):
    def __init__(self):
        super(ResNet18Fc, self).__init__()
        model_resnet18 = models.resnet18(pretrained=True)
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool
        self.__in_features = model_resnet18.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet34Fc(nn.Module):
    def __init__(self):
        super(ResNet34Fc, self).__init__()
        model_resnet34 = models.resnet34(pretrained=True)
        self.conv1 = model_resnet34.conv1
        self.bn1 = model_resnet34.bn1
        self.relu = model_resnet34.relu
        self.maxpool = model_resnet34.maxpool
        self.layer1 = model_resnet34.layer1
        self.layer2 = model_resnet34.layer2
        self.layer3 = model_resnet34.layer3
        self.layer4 = model_resnet34.layer4
        self.avgpool = model_resnet34.avgpool
        self.__in_features = model_resnet34.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet101Fc(nn.Module):
    def __init__(self):
        super(ResNet101Fc, self).__init__()
        model_resnet101 = models.resnet101(pretrained=True)
        self.conv1 = model_resnet101.conv1
        self.bn1 = model_resnet101.bn1
        self.relu = model_resnet101.relu
        self.maxpool = model_resnet101.maxpool
        self.layer1 = model_resnet101.layer1
        self.layer2 = model_resnet101.layer2
        self.layer3 = model_resnet101.layer3
        self.layer4 = model_resnet101.layer4
        self.avgpool = model_resnet101.avgpool
        self.__in_features = model_resnet101.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class ResNet152Fc(nn.Module):
    def __init__(self):
        super(ResNet152Fc, self).__init__()
        model_resnet152 = models.resnet152(pretrained=True)
        self.conv1 = model_resnet152.conv1
        self.bn1 = model_resnet152.bn1
        self.relu = model_resnet152.relu
        self.maxpool = model_resnet152.maxpool
        self.layer1 = model_resnet152.layer1
        self.layer2 = model_resnet152.layer2
        self.layer3 = model_resnet152.layer3
        self.layer4 = model_resnet152.layer4
        self.avgpool = model_resnet152.avgpool
        self.__in_features = model_resnet152.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


import torch.nn as nn
import torchvision.models as models


class RCANBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(RCANBlock, self).__init__()

        # Residual Group
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        # Channel Attention
        y = self.avg_pool(x).view(x.size(0), -1)
        y = self.fc(y).view(x.size(0), x.size(1), 1, 1)
        x = x * y.expand_as(x)

        return res + x


class RCAN(nn.Module):
    def __init__(self):
        super(RCAN, self).__init__()
        model_resnet152 = models.resnet152(pretrained=True)

        self.conv1 = model_resnet152.conv1
        self.bn1 = model_resnet152.bn1
        self.relu = model_resnet152.relu
        self.maxpool = model_resnet152.maxpool
        self.layer1 = model_resnet152.layer1
        self.layer2 = model_resnet152.layer2
        self.layer3 = model_resnet152.layer3
        self.layer4 = model_resnet152.layer4
        self.avgpool = model_resnet152.avgpool
        self.fc = model_resnet152.fc

        # Remove the fully connected layer of ResNet152
        self.fc = nn.Identity()

        # Add RCAN blocks
        self.rcan_block1 = RCANBlock(256)  # Adjust the number of channels as needed
        self.rcan_block2 = RCANBlock(512)
        self.rcan_block3 = RCANBlock(1024)
        self.rcan_block4 = RCANBlock(2048)

        # Output dimension matches the in features of ResNet152
        self.__in_features = 2048  # ResNet152's in features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x1 = self.rcan_block1(x1)
        x2 = self.rcan_block2(x2)
        x3 = self.rcan_block3(x3)
        x4 = self.rcan_block4(x4)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)

        return x

    def output_num(self):
        return self.__in_features


class RCANBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(RCANBlock, self).__init__()

        # Residual Group
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Channel Attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        # Channel Attention
        y = self.avg_pool(x).view(x.size(0), -1)
        y = self.fc(y).view(x.size(0), x.size(1), 1, 1)
        x = x * y.expand_as(x)

        return res + x


class RCAN(nn.Module):
    def __init__(self):
        super(RCAN, self).__init__()
        model_resnet152 = models.resnet152(pretrained=True)

        self.conv1 = model_resnet152.conv1
        self.bn1 = model_resnet152.bn1
        self.relu = model_resnet152.relu
        self.maxpool = model_resnet152.maxpool
        self.layer1 = model_resnet152.layer1
        self.layer2 = model_resnet152.layer2
        self.layer3 = model_resnet152.layer3
        self.layer4 = model_resnet152.layer4
        self.avgpool = model_resnet152.avgpool
        self.fc = model_resnet152.fc

        # Remove the fully connected layer of ResNet152
        self.fc = nn.Identity()

        # Add RCAN blocks
        self.rcan_block1 = RCANBlock(256)  # Adjust the number of channels as needed
        self.rcan_block2 = RCANBlock(512)
        self.rcan_block3 = RCANBlock(1024)
        self.rcan_block4 = RCANBlock(2048)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x1 = self.rcan_block1(x1)
        x2 = self.rcan_block2(x2)
        x3 = self.rcan_block3(x3)
        x4 = self.rcan_block4(x4)

        x = self.avgpool(x4)
        x = x.view(x.size(0), -1)

        return x

    def output_num(self):
        return 2048  # Adjust based on your specific ResNet model


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 2))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


class My_ResNet152Fc(nn.Module):
    def __init__(self):
        super(My_ResNet152Fc, self).__init__()
        model_resnet152 = models.resnet50(pretrained=True)
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', model_resnet152.conv1)
        self.feature.add_module('f_bn1', model_resnet152.bn1)
        self.feature.add_module('f_relu1', model_resnet152.relu)
        self.feature.add_module('f_pool1', model_resnet152.maxpool)
        self.feature.add_module('f_layer1', model_resnet152.layer1)
        self.feature.add_module('f_layer2', model_resnet152.layer2)
        self.feature.add_module('f_layer3', model_resnet152.layer3)
        self.feature.add_module('f_layer4', model_resnet152.layer4)
        self.feature.add_module('f_avgpool', model_resnet152.avgpool)
        self.__in_features = model_resnet152.fc.in_features

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.__in_features, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 2))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.__in_features, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax())

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 224, 224)
        # print(input_data.size())
        feature = self.feature(input_data)
        feature = feature.view(feature.size(0), -1)
        # print(feature.size())
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

# First, let's define these new models

# 1. New ResNet model (My_ResNet)
class My_ResNet(nn.Module):
    def __init__(self, in_features=200):
        super(My_ResNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU()
        )
        self.class_classifier = nn.Sequential(
            nn.Linear(2048, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(2048, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, x, alpha=0.5):
        feature = self.feature(x)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output

    def output_num(self):
        return 2048

# 2. New LSTM model (My_LSTM)
class My_LSTM(nn.Module):
    def __init__(self, in_features=200):
        super(My_LSTM, self).__init__()
        self.lstm = nn.LSTM(in_features, 1024, batch_first=True)
        self.class_classifier = nn.Sequential(
            nn.Linear(1024, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(1024, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, alpha=0.5):
        h0 = torch.zeros(1, x.size(0), 1024).to(x.device)  # initial hidden state
        c0 = torch.zeros(1, x.size(0), 1024).to(x.device)  # initial cell state
        out, _ = self.lstm(x.view(x.size(0), 1, -1), (h0, c0))  # LSTM output
        feature = out[:, -1, :]
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output

    def output_num(self):
        return 1024

# 3. New Transformer model (My_Transformer)
class My_Transformer(nn.Module):
    def __init__(self, in_features=200):
        super(My_Transformer, self).__init__()
        self.d_model = in_features  # Store the d_model value
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.d_model, nhead=10), num_layers=2)
        self.class_classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x, alpha=0.5):
        feature = self.transformer(x.view(x.size(0), 1, -1))
        feature = feature.view(feature.size(0), -1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output

    def output_num(self):
        return self.d_model  # Return the stored d_model value


class SimpleRegressor(nn.Module):
    def __init__(self, in_features=248, out_features=1):
        super(SimpleRegressor, self).__init__()

        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )


    def forward(self, x):
        feature = self.feature_extractor(x)
        return feature

    def output_num(self):
        return 64  # Feature dimensionality


#

# Add these models to the existing network_dict



network_dict = {"AlexNet": AlexNetFc, "ResNet18": ResNet18Fc, "ResNet34": ResNet34Fc, "ResNet50": ResNet50Fc,
                "ResNet101": ResNet101Fc, "ResNet152": ResNet152Fc,"RCAN": RCAN,
                "MyNet": My_ResNet152Fc,"NormalAlex":NormalAlex,"AttentionModel": AttentionModel}
network_dict["My_ResNet"] = My_ResNet
network_dict["My_LSTM"] = My_LSTM
network_dict["My_Transformer"] = My_Transformer
network_dict["SimpleRegressor"] = SimpleRegressor