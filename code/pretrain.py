# 训练+测试


import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
import cv2
import numpy as np

torch.manual_seed(1)  # 使用随机化种子使神经网络的初始化每次都相同

# 超参数
EPOCH = 10  # 训练整批数据的次数
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = True  # 表示还没有下载数据集，如果数据集下载好了就写False


# 用class类来建立CNN模型
class Net(nn.Module):  # 我们建立的Net继承nn.Module这个模块

    def __init__(self):
        super(Net, self).__init__()  # 第二、三行都是python类继承的基本操作,此写法应该是python2.7的继承格式,但python3里写这个好像也可以
        self.conv1 = nn.Conv2d(3, 6, 5)  # 添加第一个卷积层,调用了nn里面的Conv2d（）
        self.pool = nn.MaxPool2d(2, 2)  # 最大池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 同样是卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 接着三个全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):  # 这里定义前向传播的方法，为什么没有定义反向传播的方法呢？这其实就涉及到torch.autograd模块了，
        # 但说实话这部分网络定义的部分还没有用到autograd的知识，所以后面遇到了再讲
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))  # F是torch.nn.functional的别名，这里调用了relu函数 F.relu()
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # .view( )是一个tensor的方法，使得tensor改变size但是元素的总数是不变的。
        #  第一个参数-1是说这个参数由另一个参数确定， 比如矩阵在元素总数一定的情况下，确定列数就能确定行数。
        #  那么为什么这里只关心列数不关心行数呢，因为马上就要进入全连接层了，而全连接层说白了就是矩阵乘法，
        #  你会发现第一个全连接层的首参数是16*5*5，所以要保证能够相乘，在矩阵乘法之前就要把x调到正确的size
        # 更多的Tensor方法参考Tensor: http://pytorch.org/docs/0.3.0/tensors.html
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def dataloader():
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # 下载CIFAR10数据集
    train_data = torchvision.datasets.CIFAR10(
        root='./data/',  # 保存或提取的位置  会放在当前文件夹中
        train=True,  # true说明是用于训练的数据，false说明是用于测试的数据
        transform=transforms,  # 转换PIL.Image or numpy.ndarray
        download=DOWNLOAD_MNIST,  # 已经下载了就不需要下载了
    )

    test_data = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transforms,
    )

    return train_data, test_data


def get_text_data(test_data):
    # 进行测试
    testloader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=2)
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return images, labels, classes


def get_cnn_net():
    # 返回不同的模型，后面是在EPOCH=10时，该模型的准确率
    # net = Net()                            #64%
    # net = torchvision.models.vgg11()      #73%
    # net = torchvision.models.vgg11(pretrained=True)     #73%
    # net = torchvision.models.googlenet()  #76%
    net = torchvision.models.resnet18()  # 77%
    # net = torchvision.models.resnet18(pretrained=True)   #80%
    # for param in net.parameters():#nn.Module有成员函数parameters()
    # param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    # net.fc = nn.Linear(512, 1000)#resnet18中有self.fc，作为前向过程的最后一层。
    # net = torchvision.models.resnet50()   #71%
    # net = torchvision.models.resnet50(pretrained=True)   #80%
    print(net)
    return net


def train(train_data):
    # 批训练 50个samples
    # Torch中的DataLoader是用来包装数据的工具，它能帮我们有效迭代数据，这样就可以进行批训练
    train_loader = Data.DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True  # 是否打乱数据，一般都打乱
    )

    # 获取 cnn 网络
    net = get_cnn_net()
    # 模型加载到gpu中
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    # 优化器选择Adam
    # optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    # 仅更新模型的fc层，默认情况下更新所有的参数
    optimizer = torch.optim.Adam(net.fc.parameters(), lr=LR)

    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()  # 目标标签是one-hotted
    # 把x和y 都放入Variable中，然后放入cnn中计算output，最后再计算误差
    # 开始训练
    for epoch in range(EPOCH):
        running_loss = 0.0
        for step, (inputs, labels) in enumerate(train_loader):  # 分配batch data
            # 数据加载到gpu中
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = net(inputs)  # 先将数据放到cnn中计算output
            ### 梯度下降算法 ###
            loss = loss_func(output, labels)  # 损失函数，输出和真实标签的loss，二者位置不可颠倒
            optimizer.zero_grad()  # 清除之前学到的梯度的参数
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 应用梯度（权重更新）
            ### 梯度下降算法 ###
            # 数据加载到cpu中
            loss = loss.to('cpu')
            running_loss += loss.item()
            if step % 50 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, step + 1, running_loss / 50))
                running_loss = 0.0
                # 保存模型
    torch.save(net.state_dict(), 'test_cifar_gpu.pkl')


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def predict(test_data):
    # 获取测试数据集
    # images,labels,classes= get_text_data(test_data)
    # print images
    # imshow(torchvision.utils.make_grid(images))
    # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

    testloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    # 获取cnn网络
    net = get_cnn_net()
    # 加载模型
    net.load_state_dict(torch.load('test_cifar_gpu.pkl'))
    # 设置为推理模式
    net.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 模型加载到gpu中
    net = net.to(device)
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # 数据加载到gpu中
            images = images.to(device)
            labels = labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # 数据加载回cpu
            outputs = outputs.to('cpu')
            labels = labels.to('cpu')
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


if __name__ == "__main__":
    # 数据加载
    train_data, test_data = dataloader()

    # 训练（先训练再预测）
    train(train_data)
    # 预测（预测前将训练注释掉，否则会再训练一遍）
    predict(test_data)
