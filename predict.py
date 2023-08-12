import torch
import os
from torch import nn
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

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

# 加载之前训练好的模型
model = torch.load('model1.pt', map_location=torch.device('cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()  # 进入评估模式

# 设置图像目录和目标大小
image_dir = 'validation/v2'  # 替换为您的validation目录路径
target_size = (32, 42)

# 初始化一个空的张量来保存图像数据
batch_images = torch.zeros((6, 3, target_size[0], target_size[1]))

# 使用PIL库加载、调整大小并转换图像为张量
for i in range(6):
    image_path = os.path.join(image_dir, f'{chr(ord("a") + i)}.png')
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    batch_images[i] = image_tensor

print('组合后的张量形状:', batch_images.shape)

# 将张量数据转换为NumPy数组，并调整通道顺序
# batch_images_np = batch_images.numpy()
# batch_images_np = batch_images_np.transpose((0, 2, 3, 1))  # 调整通道顺序为(batch, height, width, channels)
# 显示组合后的图片
# plt.figure(figsize=(12, 8))
# for i in range(6):
#     plt.subplot(2, 3, i+1)
#     plt.imshow(batch_images_np[i])
#     plt.title(f'Image {chr(ord("a") + i)}')
#     plt.axis('off')
# plt.tight_layout()
# plt.show()

# 将组合后的图像张量移动到适当的设备
batch_images = batch_images.to(device)

# 使用模型进行预测
with torch.no_grad():
    outputs = model(batch_images)
    predicted_labels = torch.argmax(outputs, dim=1)

# 将预测结果转换为NumPy数组
predicted_labels_np = predicted_labels.cpu().numpy()

# 定义类别标签
class_labels = ('cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash')

# 打印预测结果
for i, label_idx in enumerate(predicted_labels_np):
    predicted_class = class_labels[label_idx]
    print(f'Image {chr(ord("a") + i)} is predicted as {predicted_class}')