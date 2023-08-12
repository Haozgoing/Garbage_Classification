import time

import numpy as np
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

use_gpu = torch.cuda.is_available()
if use_gpu:
    print('gpu可以使用')
else:
    print('gpu不可以使用')

# 超参数
epochs = 50  # 训练次数
batch_size = 6  # 批处理大小
number_workers = 0  # 多线程数目
model = 'model.pt'  # 存储训练好的模型名称

# 对加载的图像进行归一化处理，全部改为32x32x3大小的图像
data_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载数据
trainset = torchvision.datasets.ImageFolder(root='datasets/train/', transform=data_transform)
trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=number_workers, drop_last=True)
testset = torchvision.datasets.ImageFolder(root='datasets/test/', transform=data_transform)
testloader = DataLoader(testset, batch_size, shuffle=False, num_workers=number_workers)

# 类别数
classes = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')
# 获取一个批次的数据
dataloader = iter(trainloader)
images, labels = dataloader.next()
# 将图像数据标准化到 [0, 1] 范围
images = (images - images.min()) / (images.max() - images.min())

def imshow(img):
    plt.imshow(np.transpose(img, (1, 2, 0)))
    # plt.show()
imshow(torchvision.utils.make_grid(images))
# 查看图像数据的shape 6x3x32x42
print(f'单张图片的shape:{images[0].shape}')
print(f'一批数据中的shape{images.shape}')

# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(560, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 6)
        )

    def forward(self, x):
        x = self.layers(x)
        return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         #卷积
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
#         #池化
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         #卷积
#         self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
#         #全连接
#         self.fc1 = nn.Linear(in_features=16 * 5 * 5,out_features=120)
#         self.fc2 = nn.Linear(in_features=120, out_features=84)
#         self.fc3 = nn.Linear(in_features=84, out_features=6)
#
#     def forward(self, x):
#         x = self.pool1(F.relu(self.conv1(x)))
#         x = self.pool1(F.relu(self.conv2(x)))
#         #改变张量维度
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
def train():
    net = Net()
    # 指定设置 cuda表示GPU ，cpu表示CPU
    device = torch.device('cuda' if use_gpu else 'cpu')
    # 将模型和数据移到指定设备上
    net.to(device)
    print('开始训练')

    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    loss_count = []
    train_accuracy_count = []
    test_accuracy_count = []
    diff_count = []
    correct = 0  # 预测正确图片数
    total = 0  # 总共图片数

    train_correct = 0
    train_total = 0

    torch.set_num_threads(8)
    start = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # 输入数据
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward+backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # 更新参数
            optimizer.step()

            # 打印log信息
            # loss是一个scalar，需要使用loss.item()来获取数值不能使用loss[0]
            running_loss += loss.item()
            if i % 50 == 49:  # 每50个batch打印一下训练状态
                loss_count.append(running_loss / 50)
                # 计算测试集上的准确率
                for data in testloader:
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    predicted = torch.argmax(outputs, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()

                # 计算训练集上的准确率
                for train_data in trainloader:
                    images, labels = train_data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    predicted = torch.argmax(outputs, dim=1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum()

                test_accuracy_count.append((100 * correct) / total)
                train_accuracy_count.append((100 * train_correct) / train_total)
                Diff = (100 * train_correct / train_total) - (100 * correct / total)
                diff_count.append(Diff)

                print('[%d, %5d] loss: %.3f test_accuracy: %.3f train_accuracy:%.3f' \
                      % (epoch+1, i+1, running_loss / 50, 100*correct/total, 100*train_correct/train_total))

                correct = 0
                total = 0
                train_correct = 0
                train_total = 0
                running_loss = 0.0

        torch.save(net, model)
        end = time.time()
        print('训练完毕! 总耗时:%d秒' % (end - start))

        # 创建一个 2x2 的子图布局
        plt.figure(figsize=(12, 8))

        # 子图1：CNN Loss
        plt.subplot(2, 2, 1)
        plt.plot(loss_count, label='Loss')
        plt.legend()
        plt.title('CNN Loss')

        # 子图2：CNN Test Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(test_accuracy_count, label='Test_Accuracy')
        plt.legend()
        plt.title('CNN Test Accuracy')

        # 子图3：CNN Train Accuracy
        plt.subplot(2, 2, 3)
        plt.plot(train_accuracy_count, label='Train_Accuracy')
        plt.legend()
        plt.title('CNN Train Accuracy')

        # 子图4：CNN Diff Accuracy
        plt.subplot(2, 2, 4)
        plt.plot(diff_count, label='Diff_Accuracy')
        plt.legend()
        plt.title('CNN Diff Accuracy')

        # 调整子图之间的距离和布局
        plt.tight_layout()

        # 显示所有子图
        plt.show()


def test():
    correct = 0  # 预测正确的图片数
    total = 0  # 总共的图片数
    print('开始检测')

    net = torch.load(model)
    device = torch.device('cuda' if use_gpu else 'cpu')
    net.to(device)
    net.eval() # 进入评估模式

    # 由于测试的时候不需要求导，可以暂时关闭autograd，提高速度，节约内存
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('测试集中的准确率为:%d %%' % (100 * correct / total))

train()
# test()
